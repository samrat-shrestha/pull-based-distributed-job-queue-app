"""
Websocket variant of main_inmemory.py.

ONLY the worker leg changes: instead of the DGX long-polling GET /jobs/next,
it holds one persistent websocket at WS /lab/infer/v1/ws and receives jobs
pushed to it, sending completions back over the same socket.

The client side (POST /jobs with sync-hold, GET /jobs/{id}) is byte-for-byte
identical to the polling version -- clients don't know or care which the DGX uses.

Run:  uvicorn main_inmemory_ws:app --host 127.0.0.1 --port 8007
"""
import asyncio
import time
import uuid

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field

SYNC_HOLD_SECONDS = 60     # how long submit_job will wait before giving up

app = FastAPI()

# in-memory "table" -- fine for a test, gone on restart
JOBS: dict[str, dict] = {}
QUEUE: list[str] = []
LOCK = asyncio.Lock()

# set when /complete or /fail arrives, so submit_job() can wake up
DONE_EVENTS: dict[str, asyncio.Event] = {}

# signaled whenever a new job is queued, so the dispatcher wakes instantly
NEW_JOB = asyncio.Event()

API = "/lab/infer/v1"


class SubmitRequest(BaseModel):
    model: str
    input: dict
    task: str | None = None
    params: dict = Field(default_factory=dict)
    response_type: str = "sync"
    priority: int = 0
    app_id: str = "unknown"


# ---- client-facing endpoints (UNCHANGED from polling version) ----

@app.get(f"{API}/healthz")
async def healthz():
    return {"ok": True}


@app.post(f"{API}/jobs")
async def submit_job(req: SubmitRequest):
    job_id = str(uuid.uuid4())
    event = asyncio.Event()

    async with LOCK:
        JOBS[job_id] = {
            "id": job_id,
            "app_id": req.app_id,
            "model": req.model,
            "task": req.task,
            "input": req.input,
            "params": req.params,
            "response_type": req.response_type,
            "priority": req.priority,
            "status": "queued",
            "result": None,
            "error": None,
            "submitted_at": time.time(),
            "claimed_at": None,
            "done_at": None,
        }
        QUEUE.append(job_id)
        DONE_EVENTS[job_id] = event

    NEW_JOB.set()  # wake the dispatcher so it pushes to a worker immediately

    try:
        await asyncio.wait_for(event.wait(), timeout=SYNC_HOLD_SECONDS)
    except asyncio.TimeoutError:
        return {"job_id": job_id, "status": "running", "timed_out": True}
    finally:
        DONE_EVENTS.pop(job_id, None)

    return dict(JOBS[job_id])


@app.get(f"{API}/jobs/{{job_id}}")
async def get_job(job_id: str):
    job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(404, "job not found")
    out = dict(job)
    if job["done_at"] and job["claimed_at"]:
        out["dispatch_latency_s"] = job["claimed_at"] - job["submitted_at"]
        out["total_latency_s"] = job["done_at"] - job["submitted_at"]
    return out


# ---- helpers shared with the WS handler ----

async def _claim_one(worker_id: str):
    """Pop the highest-priority queued job and mark it running. None if empty."""
    async with LOCK:
        if not QUEUE:
            return None
        QUEUE.sort(key=lambda jid: (-JOBS[jid]["priority"], JOBS[jid]["submitted_at"]))
        job_id = QUEUE.pop(0)
        job = JOBS[job_id]
        job["status"] = "running"
        job["worker_id"] = worker_id
        job["claimed_at"] = time.time()
        return {
            "id": job_id,
            "model": job["model"],
            "task": job["task"],
            "input": job["input"],
            "params": job["params"],
        }


def _mark_done(job_id: str, result: dict):
    job = JOBS.get(job_id)
    if job is None:
        return
    job["status"] = "done"
    job["result"] = result
    job["done_at"] = time.time()
    ev = DONE_EVENTS.get(job_id)
    if ev is not None:
        ev.set()


def _mark_failed(job_id: str, error: str, requeue: bool):
    job = JOBS.get(job_id)
    if job is None:
        return
    if requeue:
        # worker died mid-job (socket dropped) -- put it back for another worker
        job["status"] = "queued"
        job["worker_id"] = None
        job["claimed_at"] = None
        QUEUE.append(job_id)
        NEW_JOB.set()
    else:
        job["status"] = "failed"
        job["error"] = error
        job["done_at"] = time.time()
        ev = DONE_EVENTS.get(job_id)
        if ev is not None:
            ev.set()


# ---- worker leg: ONE persistent websocket per DGX ----

@app.websocket(f"{API}/ws")
async def worker_ws(websocket: WebSocket):
    await websocket.accept()
    worker_id = websocket.query_params.get("worker_id", "unknown-worker")

    # jobs this socket has sent but not yet gotten a result for. If the socket
    # drops, these get requeued so nothing is lost.
    in_flight: dict[str, bool] = {}

    async def dispatcher():
        """Push queued jobs down the socket as they appear."""
        while True:
            job = await _claim_one(worker_id)
            if job is None:
                # nothing queued -- wait until submit_job signals a new one
                NEW_JOB.clear()
                await NEW_JOB.wait()
                continue
            in_flight[job["id"]] = True
            await websocket.send_json({"type": "job", **job})

    async def receiver():
        """Read completions/failures coming back from the worker."""
        while True:
            msg = await websocket.receive_json()
            mtype = msg.get("type")
            jid = msg.get("id")
            if mtype == "complete":
                _mark_done(jid, msg.get("result", {}))
                in_flight.pop(jid, None)
            elif mtype == "fail":
                _mark_failed(jid, msg.get("error", "worker reported failure"), requeue=False)
                in_flight.pop(jid, None)
            elif mtype == "ping":
                await websocket.send_json({"type": "pong"})

    disp = asyncio.create_task(dispatcher())
    recv = asyncio.create_task(receiver())
    try:
        # run both until the socket closes or errors
        await asyncio.gather(disp, recv)
    except WebSocketDisconnect:
        pass
    except Exception:
        pass
    finally:
        disp.cancel()
        recv.cancel()
        # requeue anything this worker had claimed but not finished
        async with LOCK:
            for jid in list(in_flight):
                job = JOBS.get(jid)
                if job and job["status"] == "running":
                    job["status"] = "queued"
                    job["worker_id"] = None
                    job["claimed_at"] = None
                    QUEUE.append(jid)
        NEW_JOB.set()
