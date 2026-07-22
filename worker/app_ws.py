"""
Websocket variant of worker.py for the DGX.

Holds one persistent websocket to the broker, receives jobs pushed to it,
runs them via Ollama, and sends results back over the same socket.
Reconnects with backoff if the connection drops.

Needs: pip install websockets httpx
Run:   python worker_ws.py
"""
import asyncio
import json
import socket

import httpx
import websockets

# note: ws:// not http://  (wss:// through nginx TLS in production)
WS_URL = "wss://<base_path>/infer/v1/ws"
OLLAMA = "http://localhost:11434"
WORKER_ID = socket.gethostname()
PING_INTERVAL = 20  # seconds


def run_ollama(job: dict) -> dict:
    model = job["model"]
    prompt = job["input"].get("prompt", "")
    params = job.get("params") or {}
    payload = {"model": model, "prompt": prompt, "stream": False}
    if params:
        payload["options"] = dict(params)
    r = httpx.post(f"{OLLAMA}/api/generate", json=payload, timeout=300)
    r.raise_for_status()
    data = r.json()
    return {"response": data.get("response", ""), "model": model,
            "eval_count": data.get("eval_count")}


HANDLERS = {
    # "flood-predict": run_flood_model,
}

def run_job(job: dict) -> dict:
    handler = HANDLERS.get(job["model"], run_ollama)
    return handler(job)


async def heartbeat(ws):
    while True:
        await asyncio.sleep(PING_INTERVAL)
        await ws.send(json.dumps({"type": "ping"}))


async def handle_connection():
    url = f"{WS_URL}?worker_id={WORKER_ID}"
    async with websockets.connect(url, ping_interval=None) as ws:
        print(f"worker {WORKER_ID} connected to {WS_URL}")
        hb = asyncio.create_task(heartbeat(ws))
        try:
            async for raw in ws:
                msg = json.loads(raw)
                if msg.get("type") == "pong":
                    continue
                if msg.get("type") != "job":
                    continue

                job = msg
                job_id = job["id"]
                print(f"got job {job_id}: model={job['model']!r} task={job.get('task')!r}")
                try:
                    # run_job is blocking (httpx sync) -- run it off the event loop
                    result = await asyncio.to_thread(run_job, job)
                    await ws.send(json.dumps({"type": "complete", "id": job_id,
                                              "result": result}))
                    print(f"job {job_id} done")
                except Exception as e:
                    await ws.send(json.dumps({"type": "fail", "id": job_id,
                                              "error": str(e)}))
                    print(f"job {job_id} failed: {e}")
        finally:
            hb.cancel()


async def main():
    backoff = 1
    print(f"worker {WORKER_ID} starting, connecting to {WS_URL}")
    while True:
        try:
            await handle_connection()
            backoff = 1
        except Exception as e:
            print(f"connection lost: {e}; reconnecting in {backoff}s")
            await asyncio.sleep(min(backoff, 30))
            backoff *= 2


if __name__ == "__main__":
    asyncio.run(main())
