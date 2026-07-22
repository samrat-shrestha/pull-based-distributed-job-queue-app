import asyncio
import json
import logging
import os
import uuid
from contextlib import asynccontextmanager

import asyncpg
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("infer-ws")

DATABASE_URL = os.environ.get(
    "DATABASE_URL"
)
SYNC_HOLD_SECONDS = 60    # how long submit_job will wait before giving up

# one Event per in-flight job_id -- set when the WS receiver gets a
# complete/fail message, so submit_job() can wake up the moment the DGX is
# done, instead of polling.
# NOTE: lives in process memory. If you ever run more than one uvicorn worker
# process, this breaks -- fix at that point: Postgres LISTEN/NOTIFY.
DONE_EVENTS: dict[str, asyncio.Event] = {}

# signaled whenever a new job is queued, so the dispatcher wakes instantly
# instead of polling Postgres on a timer.
NEW_JOB = asyncio.Event()

API = "/infer/v1"   # versioned base path


class SubmitRequest(BaseModel):
    model: str
    input: dict                              # the payload, e.g. {"prompt": "..."}
    task: str | None = None
    params: dict = Field(default_factory=dict)
    response_type: str = "sync"              # sync | async | stream
    priority: int = 0
    app_id: str = "unknown"


LEASE_SWEEP_INTERVAL_SECONDS = 30   # how often to check for stale 'running' jobs


async def lease_sweeper(pool):
    """Background loop: catches jobs whose worker went silent without the
    websocket itself dying (hung inference, worker stuck in some loop,
    process frozen but socket still technically open). Without this, such a
    job sits 'running' forever with no error surfaced anywhere -- the client
    just eventually hits the sync-hold timeout with no explanation."""
    while True:
        await asyncio.sleep(LEASE_SWEEP_INTERVAL_SECONDS)
        try:
            async with pool.acquire() as conn:
                rows = await conn.fetch(
                    """
                    UPDATE jobs SET status='queued', worker_id=NULL, started_at=NULL
                    WHERE status='running' AND lease_expires_at < now()
                    RETURNING id
                    """
                )
            if rows:
                logger.warning("lease sweeper requeued %d stale job(s): %s",
                                len(rows), [str(r["id"]) for r in rows])
                NEW_JOB.set()
        except Exception:
            logger.exception("lease sweeper failed on this pass")


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.pool = await asyncpg.create_pool(
        DATABASE_URL, min_size=2, max_size=10,
        command_timeout=10,          # don't let a stuck query hang the dispatcher forever
        max_inactive_connection_lifetime=300,
    )
    sweeper_task = asyncio.create_task(lease_sweeper(app.state.pool))
    yield
    sweeper_task.cancel()
    await app.state.pool.close()


app = FastAPI(lifespan=lifespan)


def _row_to_dict(row):
    """asyncpg returns JSONB columns as strings; decode them back to objects."""
    d = dict(row)
    for k in ("input", "params", "result"):
        if k in d and isinstance(d[k], str):
            d[k] = json.loads(d[k])
    for k, v in list(d.items()):
        if isinstance(v, uuid.UUID):
            d[k] = str(v)
    return d


@app.get(f"{API}/healthz")
async def healthz():
    return {"ok": True}


@app.post(f"{API}/jobs")
async def submit_job(req: SubmitRequest):
    job_id = uuid.uuid4()
    event = asyncio.Event()
    DONE_EVENTS[str(job_id)] = event

    async with app.state.pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO jobs
              (id, app_id, model, task, input, params, response_type, priority)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            """,
            job_id, req.app_id, req.model, req.task,
            json.dumps(req.input), json.dumps(req.params),
            req.response_type, req.priority,
        )

    NEW_JOB.set()  # wake the WS dispatcher so it pushes to a worker immediately

    try:
        await asyncio.wait_for(event.wait(), timeout=SYNC_HOLD_SECONDS)
    except asyncio.TimeoutError:
        return {"job_id": str(job_id), "status": "running", "timed_out": True}
    finally:
        DONE_EVENTS.pop(str(job_id), None)

    async with app.state.pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT id, app_id, model, task, input, params, response_type,
                   priority, status, result, error, created_at, started_at,
                   finished_at
            FROM jobs WHERE id = $1
            """,
            job_id,
        )
    return _row_to_dict(row)


@app.get(f"{API}/jobs/{{job_id}}")
async def get_job(job_id: uuid.UUID):
    async with app.state.pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT id, app_id, model, task, input, params, response_type,
                   priority, status, result, error, created_at, started_at,
                   finished_at
            FROM jobs WHERE id = $1
            """,
            job_id,
        )
    if row is None:
        raise HTTPException(404, "job not found")
    return _row_to_dict(row)


# ---- helpers used only by the WS handler ----

LEASE_DURATION_SECONDS = 300   # generous headroom above worst-case real inference
                               # time. If this is ever too short, a job that's
                               # merely slow (not stuck) gets wrongly requeued
                               # and can run twice -- see worker_id guards below
                               # for why that's now safe even if it happens.


async def _claim_one(worker_id: str):
    """Atomically claim the highest-priority queued job. None if empty."""
    async with app.state.pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            UPDATE jobs SET
                status = 'running',
                worker_id = $1,
                started_at = now(),
                lease_expires_at = now() + make_interval(secs => $2)
            WHERE id = (
                SELECT id FROM jobs
                WHERE status = 'queued'
                ORDER BY priority DESC, created_at ASC
                FOR UPDATE SKIP LOCKED
                LIMIT 1
            )
            RETURNING id, model, task, input, params
            """,
            worker_id, LEASE_DURATION_SECONDS,
        )
    return _row_to_dict(row) if row else None


async def _mark_done(job_id: str, result: dict, worker_id: str):
    """Only applies if this worker still actually holds the claim. If the
    lease expired and the job got reassigned before this completion arrived,
    worker_id won't match anymore and this becomes a harmless no-op instead
    of clobbering whatever the reassigned attempt wrote."""
    async with app.state.pool.acquire() as conn:
        tag = await conn.execute(
            """
            UPDATE jobs SET status='done', result=$1, finished_at=now()
            WHERE id=$2 AND worker_id=$3 AND status='running'
            """,
            json.dumps(result), uuid.UUID(job_id), worker_id,
        )
    if tag.endswith("0"):
        logger.warning(
            "late/stale completion for job %s from worker %s ignored "
            "(lease likely already reassigned)", job_id, worker_id,
        )
    event = DONE_EVENTS.get(job_id)
    if event is not None:
        event.set()


async def _mark_failed(job_id: str, error: str, worker_id: str):
    async with app.state.pool.acquire() as conn:
        tag = await conn.execute(
            """
            UPDATE jobs SET status='failed', error=$1, finished_at=now()
            WHERE id=$2 AND worker_id=$3 AND status='running'
            """,
            error, uuid.UUID(job_id), worker_id,
        )
    if tag.endswith("0"):
        logger.warning(
            "late/stale failure for job %s from worker %s ignored "
            "(lease likely already reassigned)", job_id, worker_id,
        )
    event = DONE_EVENTS.get(job_id)
    if event is not None:
        event.set()


async def _requeue(job_id: str, worker_id: str):
    """Worker's socket died mid-job -- put it back for another worker.
    Race-safe: only requeue if it's still 'running' under THIS worker_id --
    if the lease already expired and the sweeper reassigned it to someone
    else, this becomes a no-op instead of stealing it back."""
    async with app.state.pool.acquire() as conn:
        await conn.execute(
            """
            UPDATE jobs SET status='queued', worker_id=NULL, started_at=NULL
            WHERE id=$1 AND worker_id=$2 AND status='running'
            """,
            uuid.UUID(job_id), worker_id,
        )
    NEW_JOB.set()


# ---- worker leg: ONE persistent websocket per DGX ----

@app.websocket(f"{API}/ws")
async def worker_ws(websocket: WebSocket):
    await websocket.accept()
    worker_id = websocket.query_params.get("worker_id", "unknown-worker")

    # jobs this socket has sent but not yet gotten a result for. If the
    # socket drops, these get requeued so nothing is lost.
    in_flight: dict[str, bool] = {}

    async def dispatcher():
        """Push queued jobs down the socket as they appear."""
        while True:
            job = await _claim_one(worker_id)
            if job is None:
                NEW_JOB.clear()
                await NEW_JOB.wait()
                continue
            in_flight[job["id"]] = True
            await websocket.send_json({"type": "job", **job})

    async def receiver():
        """Read completions/failures coming back from the worker."""
        while True:
            msg = await websocket.receive_json()
            try:
                mtype = msg.get("type")
                jid = msg.get("id")
                if mtype == "complete":
                    await _mark_done(jid, msg.get("result", {}), worker_id)
                    in_flight.pop(jid, None)
                elif mtype == "fail":
                    await _mark_failed(jid, msg.get("error", "worker reported failure"), worker_id)
                    in_flight.pop(jid, None)
                elif mtype == "ping":
                    await websocket.send_json({"type": "pong"})
            except Exception:
                # A single malformed/unexpected message (bad id, bad UUID,
                # unknown type) shouldn't take down the whole connection and
                # punish every other in-flight job on it -- log and move on.
                logger.exception("bad message from worker %s, ignoring: %r",
                                  worker_id, msg)

    disp = asyncio.create_task(dispatcher())
    recv = asyncio.create_task(receiver())
    try:
        await asyncio.gather(disp, recv)
    except WebSocketDisconnect as e:
        logger.info("worker %s disconnected: code=%s reason=%r", worker_id, e.code, e.reason)
    except Exception:
        # THIS is the line that was hiding the real bug -- log it instead of
        # swallowing it, so DB errors, pool timeouts, etc. actually show up.
        logger.exception("worker_ws crashed for worker_id=%s", worker_id)
    finally:
        disp.cancel()
        recv.cancel()
        # requeue anything this worker had claimed but not finished
        for jid in list(in_flight):
            try:
                await _requeue(jid, worker_id)
            except Exception:
                logger.exception("failed to requeue job %s after disconnect", jid)
