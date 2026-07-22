import asyncio
import time
import uuid

from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel

WORKER_POLL_SECONDS = 25   # keep under nginx proxy_read_timeout
SYNC_HOLD_SECONDS = 60     # how long submit_job will wait before giving up

app = FastAPI()

# in-memory "table" -- fine for a latency test, gone on restart
JOBS: dict[str, dict] = {}
QUEUE: list[str] = []
LOCK = asyncio.Lock()

# one Event per in-flight job_id -- set when /complete or /fail arrives,
# so submit_job() can wake up the moment the DGX is done, instead of polling.
DONE_EVENTS: dict[str, asyncio.Event] = {}


class SubmitRequest(BaseModel):
    model: str
    prompt: str


class CompleteRequest(BaseModel):
    result: str


class FailRequest(BaseModel):
    error: str


@app.get("/infer/healthz")
async def healthz():
    return {"ok": True}


@app.post("/infer/jobs")
async def submit_job(req: SubmitRequest):
    job_id = str(uuid.uuid4())
    event = asyncio.Event()

    async with LOCK:
        JOBS[job_id] = {
            "id": job_id,
            "model": req.model,
            "prompt": req.prompt,
            "status": "queued",
            "result": None,
            "error": None,
            "submitted_at": time.time(),   # for latency measurement
            "claimed_at": None,
            "done_at": None,
        }
        QUEUE.append(job_id)
        DONE_EVENTS[job_id] = event

    # HOLD HERE: don't respond to the client yet. Wait until /complete or
    # /fail sets this job's event, or until SYNC_HOLD_SECONDS runs out.
    try:
        await asyncio.wait_for(event.wait(), timeout=SYNC_HOLD_SECONDS)
    except asyncio.TimeoutError:
        # took too long -- give the client something to poll instead of
        # holding the connection forever
        return {"job_id": job_id, "status": "running", "timed_out": True}
    finally:
        DONE_EVENTS.pop(job_id, None)  # always clean up, even on timeout

    # event was set -- the job finished while we were waiting
    job = JOBS[job_id]
    return dict(job)


@app.get("/infer/jobs/next")
async def next_job(worker_id: str):
    deadline = asyncio.get_event_loop().time() + WORKER_POLL_SECONDS
    while asyncio.get_event_loop().time() < deadline:
        async with LOCK:
            if QUEUE:
                job_id = QUEUE.pop(0)
                job = JOBS[job_id]
                job["status"] = "running"
                job["worker_id"] = worker_id
                job["claimed_at"] = time.time()
                return {"id": job_id, "model": job["model"], "prompt": job["prompt"]}
        await asyncio.sleep(0.2)
    return Response(status_code=204)


@app.get("/infer/jobs/{job_id}")
async def get_job(job_id: str):
    job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(404, "job not found")
    out = dict(job)
    if job["done_at"]:
        out["dispatch_latency_s"] = job["claimed_at"] - job["submitted_at"]
        out["total_latency_s"] = job["done_at"] - job["submitted_at"]
    return out


@app.post("/infer/jobs/{job_id}/complete")
async def complete_job(job_id: str, req: CompleteRequest):
    job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(404, "job not found")
    job["status"] = "done"
    job["result"] = req.result
    job["done_at"] = time.time()

    event = DONE_EVENTS.get(job_id)
    if event is not None:
        event.set()   # wakes up the submit_job() request that's waiting

    return {"ok": True}


@app.post("/infer/jobs/{job_id}/fail")
async def fail_job(job_id: str, req: FailRequest):
    job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(404, "job not found")
    job["status"] = "failed"
    job["error"] = req.error
    job["done_at"] = time.time()

    event = DONE_EVENTS.get(job_id)
    if event is not None:
        event.set()   # wakes up the waiting submit_job() request too

    return {"ok": True}
