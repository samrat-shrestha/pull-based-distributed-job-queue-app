"""
Websocket worker for the DGX.

Holds one persistent websocket to the broker, receives pushed jobs, runs them
via Ollama (or a registered custom handler), and reports results back over the
same socket. Reconnects with exponential backoff.

Design notes -- the non-obvious parts:

1. CONCURRENCY IS BOUNDED AND ADVERTISED.
   Jobs run as independent asyncio tasks so the receive loop never blocks, but
   a semaphore caps how many run at once, and the worker tells the broker its
   capacity in the `hello` frame on connect. The broker should push at most
   that many outstanding jobs. Excess work stays `queued` in Postgres where
   it is visible and re-prioritisable, rather than piling up invisibly in this
   process or inside Ollama's internal queue.

2. LEASES ARE RENEWED, NOT JUST SET.
   Under concurrency, jobs share GPU bandwidth and take longer than they did
   when the worker ran one at a time. A lease sized for serial execution will
   expire mid-job, the broker's sweeper will requeue a job that is still
   running, and the GPU ends up doing the same work twice. Each running job
   therefore sends a `renew` frame every LEASE_RENEW_INTERVAL seconds. This
   requires broker support -- see BROKER REQUIREMENTS at the bottom.

3. ONE WRITER OWNS THE SOCKET.
   All outbound frames go through an asyncio.Queue drained by a single writer
   task, so concurrent job completions and heartbeats can never interleave
   partial frames regardless of the websockets version in use.

4. NOTHING SURVIVES A DISCONNECT.
   On connection loss, in-flight tasks get a short grace period to finish and
   report, then are cancelled outright. A task that outlived its connection
   would hold a dead socket, fail to report, and race the broker's requeue --
   producing duplicate execution. The broker requeues on disconnect anyway,
   so cancelling is both correct and cheaper.

Needs: pip install websockets httpx
Run:   python -u worker_ws.py        (-u matters: systemd block-buffers stdout)
"""
import asyncio
import contextlib
import json
import logging
import os
import signal
import socket

import httpx
import websockets

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("worker")

WS_URL = os.getenv("WS_URL", "wss://<baseurl>/infer/v1/ws")
OLLAMA = os.getenv("OLLAMA_URL", "http://localhost:11434")
WORKER_ID = os.getenv("WORKER_ID", socket.gethostname())
WORKER_TOKEN = os.getenv("WORKER_TOKEN", "")

# Models this worker will accept. Empty = accept anything the broker sends.
# Set this once you run more than one worker (e.g. an Ollama worker and a
# vLLM worker) so the broker's claim query can filter with
#   WHERE model = ANY($capabilities)
CAPABILITIES = [m for m in os.getenv("CAPABILITIES", "").split(",") if m]

# How many jobs to run at once. Should match what the GPU can actually serve:
# roughly OLLAMA_NUM_PARALLEL x OLLAMA_MAX_LOADED_MODELS, capped by memory.
# Higher just moves queueing from Postgres into Ollama, which is worse.
MAX_CONCURRENT = int(os.getenv("MAX_CONCURRENT", "2"))

PING_INTERVAL = 20            # app-level keepalive to the broker
LEASE_RENEW_INTERVAL = 15     # must be comfortably under the broker's lease TTL
OLLAMA_TIMEOUT = 600          # ceiling for one inference call
DRAIN_TIMEOUT = 5             # grace period for in-flight jobs on disconnect
RECONNECT_MIN_DELAY = 1
RECONNECT_MAX_DELAY = 30

http_client: httpx.AsyncClient | None = None
shutdown = asyncio.Event()    # set by SIGTERM/SIGINT for a clean systemd stop


# --------------------------------------------------------------------------
# Model handlers
# --------------------------------------------------------------------------

async def run_ollama(job: dict) -> dict:
    model = job["model"]
    inp = job["input"]
    params = job.get("params") or {}
    messages = inp.get("messages")

    if messages:
        payload = {"model": model, "messages": messages, "stream": False}
        endpoint = "/api/chat"
    else:
        payload = {"model": model, "prompt": inp.get("prompt", ""), "stream": False}
        endpoint = "/api/generate"
    if params:
        payload["options"] = dict(params)

    async with httpx.AsyncClient(timeout=300) as client:
        r = await client.post(f"{OLLAMA}{endpoint}", json=payload)
        r.raise_for_status()
    data = r.json()
    text = (data.get("message") or {}).get("content", "") if messages else data.get("response", "")
    return {"response": text, "model": model, "eval_count": data.get("eval_count")}

# alongside the existing OLLAMA constant
OLLAMA_DOCKER = os.getenv("OLLAMA_DOCKER_URL", "http://localhost:11435")


async def run_ollama_docker(job: dict) -> dict:
    """Same as run_ollama but targets the containerized Ollama instance."""
    model = job["model"].removesuffix("-docker")   # routing label vs. real model name
    prompt = job["input"].get("prompt", "")
    params = job.get("params") or {}
    payload = {"model": model, "prompt": prompt, "stream": False}
    if params:
        payload["options"] = dict(params)
    r = await http_client.post(f"{OLLAMA_DOCKER}/api/generate", json=payload,
                                timeout=OLLAMA_TIMEOUT)
    r.raise_for_status()
    data = r.json()
    return {"response": data.get("response", ""), "model": model,
            "eval_count": data.get("eval_count"), "source": "docker"}

HANDLERS = {
    # "flood-predict": run_flood_model,   # future non-Ollama models go here
    "gemma3:270m-docker": run_ollama_docker,   # routes only this model tag to the container
}


async def run_job(job: dict) -> dict:
    handler = HANDLERS.get(job["model"], run_ollama)
    return await handler(job)


# --------------------------------------------------------------------------
# Connection-scoped state
#
# Everything below is recreated per connection. Nothing is shared across
# reconnects, which is what keeps a dead socket from leaking into a live one.
# --------------------------------------------------------------------------

class Connection:
    def __init__(self, ws):
        self.ws = ws
        self.outbox: asyncio.Queue[str] = asyncio.Queue()
        self.sem = asyncio.Semaphore(MAX_CONCURRENT)
        self.tasks: set[asyncio.Task] = set()
        self.closing = False

    def send(self, msg: dict) -> None:
        """Queue a frame for the writer task. Never blocks, never raises --
        if the socket is gone the frames are simply dropped and the broker
        requeues on disconnect."""
        if not self.closing:
            self.outbox.put_nowait(json.dumps(msg))

    async def writer(self) -> None:
        """Sole owner of ws.send(). Serialises all outbound traffic."""
        while True:
            raw = await self.outbox.get()
            try:
                await self.ws.send(raw)
            except Exception:
                logger.warning("send failed -- connection is going away")
                return

    async def heartbeat(self) -> None:
        while True:
            await asyncio.sleep(PING_INTERVAL)
            self.send({"type": "ping", "worker_id": WORKER_ID})

    async def lease_renewer(self, job_id: str) -> None:
        """Keeps the broker from requeueing a job that is still running."""
        while True:
            await asyncio.sleep(LEASE_RENEW_INTERVAL)
            self.send({"type": "renew", "id": job_id, "worker_id": WORKER_ID})

    async def process_job(self, job: dict) -> None:
        job_id = job["id"]

        # The semaphore is acquired INSIDE the task rather than before
        # creating it, so the receive loop is never blocked. Jobs beyond
        # capacity wait here briefly -- but the broker shouldn't be sending
        # them in the first place if it respects the advertised capacity.
        async with self.sem:
            if self.closing:
                logger.info("job %s dropped -- connection closing", job_id)
                return

            logger.info(
                "job %s started: model=%r task=%r", job_id, job["model"], job.get("task")
            )
            renewer = asyncio.create_task(self.lease_renewer(job_id))
            try:
                result = await run_job(job)
                self.send({"type": "complete", "id": job_id, "result": result})
                logger.info("job %s done", job_id)
            except asyncio.CancelledError:
                # Connection died underneath us. Don't report anything -- the
                # broker already requeued this on disconnect, and a late
                # `fail` could clobber a fresh attempt.
                logger.warning("job %s cancelled -- broker will requeue", job_id)
                raise
            except httpx.HTTPStatusError as e:
                body = e.response.text[:500]
                logger.error("job %s: ollama returned %s: %s",
                             job_id, e.response.status_code, body)
                self.send({"type": "fail", "id": job_id,
                           "error": f"ollama {e.response.status_code}: {body}"})
            except httpx.TimeoutException:
                logger.error("job %s timed out after %ss", job_id, OLLAMA_TIMEOUT)
                self.send({"type": "fail", "id": job_id,
                           "error": f"inference timed out after {OLLAMA_TIMEOUT}s"})
            except Exception as e:
                logger.exception("job %s failed", job_id)
                self.send({"type": "fail", "id": job_id, "error": f"{type(e).__name__}: {e}"})
            finally:
                renewer.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await renewer

    def spawn(self, job: dict) -> None:
        task = asyncio.create_task(self.process_job(job))
        self.tasks.add(task)
        task.add_done_callback(self.tasks.discard)

    async def drain(self) -> None:
        """Give in-flight jobs a moment to finish, then cancel the rest.
        Leaving them running would mean duplicate GPU work after requeue."""
        self.closing = True
        if not self.tasks:
            return
        logger.info("closing with %d job(s) in flight", len(self.tasks))
        done, pending = await asyncio.wait(self.tasks, timeout=DRAIN_TIMEOUT)
        for t in pending:
            t.cancel()
        if pending:
            logger.warning("cancelled %d unfinished job(s)", len(pending))
            await asyncio.gather(*pending, return_exceptions=True)


# --------------------------------------------------------------------------
# Connection lifecycle
# --------------------------------------------------------------------------

async def handle_connection() -> None:
    url = f"{WS_URL}?worker_id={WORKER_ID}"
    headers = {"Authorization": f"Bearer {WORKER_TOKEN}"} if WORKER_TOKEN else {}

    async with websockets.connect(
        url,
        additional_headers=headers,   # `extra_headers` on websockets < 14
        ping_interval=20,             # protocol-level keepalive; catches dead
        ping_timeout=20,              # NAT/proxy connections the app ping won't
        max_size=16 * 1024 * 1024,
        open_timeout=15,
    ) as ws:
        conn = Connection(ws)
        logger.info(
            "worker %s connected (capacity=%d, capabilities=%s)",
            WORKER_ID, MAX_CONCURRENT, CAPABILITIES or "any",
        )

        # Tell the broker what we can take. Without this it has no basis for
        # pushing more than one job at a time.
        conn.send({
            "type": "hello",
            "worker_id": WORKER_ID,
            "capacity": MAX_CONCURRENT,
            "capabilities": CAPABILITIES,
        })

        writer = asyncio.create_task(conn.writer())
        hb = asyncio.create_task(conn.heartbeat())
        shutdown_watch = asyncio.create_task(shutdown.wait())

        try:
            receiver = asyncio.create_task(receive_loop(conn))
            done, _ = await asyncio.wait(
                {receiver, shutdown_watch, writer},
                return_when=asyncio.FIRST_COMPLETED,
            )
            if shutdown_watch in done:
                logger.info("shutdown signal -- draining")
            receiver.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await receiver
        finally:
            hb.cancel()
            shutdown_watch.cancel()
            await conn.drain()
            # let queued completions flush before the socket closes
            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait_for(conn.outbox.join(), timeout=2)
            writer.cancel()
            for t in (hb, writer, shutdown_watch):
                with contextlib.suppress(asyncio.CancelledError):
                    await t


async def receive_loop(conn: Connection) -> None:
    async for raw in conn.ws:
        try:
            msg = json.loads(raw)
        except json.JSONDecodeError:
            logger.warning("ignoring malformed frame: %.200s", raw)
            continue

        mtype = msg.get("type")
        if mtype in ("pong", "ack", "renewed"):
            continue
        if mtype != "job":
            logger.debug("ignoring frame type %r", mtype)
            continue

        if not msg.get("id") or not msg.get("model"):
            logger.warning("ignoring job frame missing id/model: %.200s", raw)
            continue

        # Defensive: if the broker's capability filter is misconfigured, fail
        # fast and let it reroute rather than erroring inside Ollama.
        if CAPABILITIES and msg["model"] not in CAPABILITIES:
            logger.warning("rejecting job %s -- model %r not in capabilities",
                           msg["id"], msg["model"])
            conn.send({"type": "fail", "id": msg["id"],
                       "error": f"worker cannot serve model {msg['model']!r}"})
            continue

        conn.spawn(msg)


async def main() -> None:
    delay = RECONNECT_MIN_DELAY
    while not shutdown.is_set():
        try:
            await handle_connection()
            if shutdown.is_set():
                break
            delay = RECONNECT_MIN_DELAY
            logger.warning("connection closed cleanly -- reconnecting in %.1fs", delay)
        except websockets.InvalidStatus as e:
            # A 403 here is almost always nginx, not the app: missing
            # Upgrade/Connection headers, or the /ws location block ordered
            # after the general one. Check with:
            #   sudo nginx -T | grep -A 15 "location /lab/infer/v1/ws"
            # and isolate by curling the app directly, bypassing nginx.
            logger.error("handshake rejected (%s) -- check nginx ws config", e)
        except (websockets.ConnectionClosed, OSError) as e:
            logger.warning("connection lost: %s", e)
        except Exception:
            logger.exception("unexpected error in worker loop")

        if shutdown.is_set():
            break
        logger.info("reconnecting in %.1fs", delay)
        with contextlib.suppress(asyncio.TimeoutError):
            await asyncio.wait_for(shutdown.wait(), timeout=delay)
        delay = min(delay * 2, RECONNECT_MAX_DELAY)

    logger.info("worker stopped")


def _install_signal_handlers() -> None:
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        with contextlib.suppress(NotImplementedError):
            loop.add_signal_handler(sig, shutdown.set)


async def _run() -> None:
    global http_client
    _install_signal_handlers()
    # Pool must exceed MAX_CONCURRENT or jobs serialise on HTTP connections --
    # reintroducing the exact bottleneck this rewrite removes.
    limits = httpx.Limits(
        max_connections=MAX_CONCURRENT * 2 + 4,
        max_keepalive_connections=MAX_CONCURRENT * 2,
    )
    async with httpx.AsyncClient(limits=limits) as client:
        http_client = client
        await main()


if __name__ == "__main__":
    asyncio.run(_run())


# --------------------------------------------------------------------------
# BROKER REQUIREMENTS (main_ws.py)
#
# This worker is useless without matching changes on the server side:
#
# 1. Handle the `hello` frame. Store capacity and capabilities per connected
#    worker. Until the broker knows capacity is > 1 it will keep pushing one
#    job at a time and none of this concurrency will be exercised.
#
# 2. Push up to `capacity` outstanding jobs per worker. If dispatch is
#    currently gated on a per-worker `busy` flag cleared by complete/fail,
#    replace it with a counter compared against capacity.
#
# 3. Handle the `renew` frame:
#       UPDATE jobs SET lease_expires_at = now() + interval '60 seconds'
#       WHERE id = $1 AND worker_id = $2 AND status = 'running'
#    The lease TTL must exceed LEASE_RENEW_INTERVAL with margin -- 60s against
#    a 15s renew gives three missed renewals before a requeue.
#
# 4. Filter claims by capability once more than one worker exists:
#       WHERE status = 'queued' AND (model = ANY($1::text[]) OR $1 = '{}')
#       ORDER BY priority DESC, created_at ASC
#       FOR UPDATE SKIP LOCKED
#
# 5. On disconnect, requeue every job still `running` for that worker_id --
#    this worker deliberately does not report failures for cancelled jobs.
# --------------------------------------------------------------------------