import socket
import time

import httpx

SERVER = "<server url>"
OLLAMA = "http://localhost:11434"
WORKER_ID = socket.gethostname()


def run_inference(model: str, prompt: str) -> str:
    r = httpx.post(
        f"{OLLAMA}/api/generate",
        json={"model": model, "prompt": prompt, "stream": False},
        timeout=120,
    )
    r.raise_for_status()
    return r.json()["response"]


def main():
    backoff = 1
    print(f"worker {WORKER_ID} starting, polling {SERVER}")
    with httpx.Client(timeout=40.0) as http:
        while True:
            try:
                r = http.get(f"{SERVER}/jobs/next", params={"worker_id": WORKER_ID})
                backoff = 1  # reset on any successful exchange

                if r.status_code == 204:
                    continue  # nothing queued, poll again immediately

                r.raise_for_status()
                job = r.json()
                job_id = job["id"]
                print(f"got job {job_id}: model={job['model']!r}")

                try:
                    result = run_inference(job["model"], job["prompt"])
                    http.post(f"{SERVER}/jobs/{job_id}/complete", json={"result": result})
                    print(f"job {job_id} done")
                except Exception as e:
                    http.post(f"{SERVER}/jobs/{job_id}/fail", json={"error": str(e)})
                    print(f"job {job_id} failed: {e}")

            except (httpx.ConnectError, httpx.ReadTimeout, httpx.HTTPStatusError) as e:
                print(f"connection issue: {e}, backing off {backoff}s")
                time.sleep(min(backoff, 30))
                backoff *= 2


if __name__ == "__main__":
    main()
