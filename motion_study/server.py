"""Motion Study web app: FastAPI server.

Run:  python server.py            (then open http://127.0.0.1:8765)
      python server.py --port 9000 --host 0.0.0.0   (to reach it from other machines)
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import threading
import time
import uuid
from pathlib import Path

import cv2
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import analysis

HERE = Path(__file__).resolve().parent
JOBS_DIR = HERE / "jobs"
STATIC = HERE / "static"
JOBS_DIR.mkdir(exist_ok=True)

app = FastAPI(title="Motion Study")


class Job:
    def __init__(self, jid: str, folder: Path):
        self.id = jid
        self.folder = folder
        self.files: list[Path] = []
        self.name = ""
        self.source: analysis.Source | None = None
        self.state = "uploaded"  # uploaded -> running -> done | error | cancelled
        self.message = ""
        self.frames: list[dict] = []
        self.total = 0
        self.done = 0
        self.fps = 0.0
        self.cancel = False
        self.params: dict = {}
        self.people: list[dict] = []
        self.error = ""
        self.thread: threading.Thread | None = None

    def meta(self) -> dict:
        m = self.source.meta() if self.source else {}
        m.update(id=self.id, name=self.name, state=self.state)
        return m


JOBS: dict[str, Job] = {}
SAFE = re.compile(r"[^A-Za-z0-9._-]+")


def get_job(jid: str) -> Job:
    job = JOBS.get(jid)
    if job is None:
        job = restore(jid)
    if job is None:
        raise HTTPException(404, "unknown job (it may have been deleted)")
    return job


def restore(jid: str) -> Job | None:
    """Reload a finished job from disk, so results survive a server restart."""
    if not re.fullmatch(r"[a-f0-9]{12}", jid):
        return None
    folder = JOBS_DIR / jid
    info = folder / "job.json"
    if not info.exists():
        return None
    data = json.loads(info.read_text(encoding="utf-8"))
    job = Job(jid, folder)
    job.name = data.get("name", "")
    job.files = [folder / "input" / f for f in data.get("files", [])]
    try:
        job.source = analysis.open_source(job.files, data.get("interval", 1.0))
    except Exception:
        return None
    res = folder / "results.json"
    if res.exists():
        r = json.loads(res.read_text(encoding="utf-8"))
        job.frames, job.params = r["frames"], r["params"]
        job.total = job.done = len(job.frames)
        job.state = r.get("state", "done")
    JOBS[jid] = job
    return job


def save_job(job: Job, interval: float = 1.0):
    (job.folder / "job.json").write_text(json.dumps({
        "name": job.name, "files": [f.name for f in job.files], "interval": interval,
    }), encoding="utf-8")


# --------------------------------------------------------------------------- #
# Upload: raw request body streamed to disk (works for multi-GB videos)
# --------------------------------------------------------------------------- #
@app.post("/api/upload")
async def upload(request: Request, name: str, job: str | None = None):
    if job:
        j = get_job(job)
        if j.state != "uploaded":
            raise HTTPException(400, "job already analysed")
    else:
        jid = uuid.uuid4().hex[:12]
        folder = JOBS_DIR / jid
        (folder / "input").mkdir(parents=True)
        j = JOBS[jid] = Job(jid, folder)
        j.name = name
    fname = SAFE.sub("_", Path(name).name) or "upload"
    dest = j.folder / "input" / f"{len(j.files):04d}_{fname}"
    with open(dest, "wb") as f:
        async for chunk in request.stream():
            f.write(chunk)
    j.files.append(dest)
    return {"id": j.id, "files": len(j.files)}


class OpenReq(BaseModel):
    interval: float = 1.0  # seconds between images, for image sequences


@app.post("/api/jobs/{jid}/open")
def open_job(jid: str, req: OpenReq):
    job = get_job(jid)
    if job.source:
        job.source.close()
    try:
        job.source = analysis.open_source(job.files, req.interval)
    except ValueError as e:
        raise HTTPException(400, str(e))
    if len(job.files) > 1:
        job.name = f"{len(job.files)} images"
    save_job(job, req.interval)
    return job.meta()


@app.get("/api/jobs/{jid}")
def job_info(jid: str):
    job = get_job(jid)
    return {**job.meta(), "params": job.params, "frames_ready": len(job.frames)}


@app.get("/api/jobs/{jid}/frame")
def frame_image(jid: str, i: int = 0, w: int = 1280):
    job = get_job(jid)
    if not job.source:
        raise HTTPException(400, "source not opened")
    frame = job.source.read(max(0, min(i, max(0, job.source.count - 1))))
    if frame is None:
        raise HTTPException(404, "frame not readable")
    if w and frame.shape[1] > w:
        frame = cv2.resize(frame, (w, int(round(frame.shape[0] * w / frame.shape[1]))),
                           interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return Response(buf.tobytes(), media_type="image/jpeg",
                    headers={"Cache-Control": "max-age=3600"})


class DetectReq(BaseModel):
    frame: int = 0
    quality: str = "fast"
    conf: float = 0.35
    max_width: int = 960


@app.post("/api/jobs/{jid}/detect")
def detect(jid: str, req: DetectReq):
    job = get_job(jid)
    if not job.source:
        raise HTTPException(400, "source not opened")
    if job.state == "running":
        raise HTTPException(409, "analysis already running")
    try:
        people = analysis.detect_people(job.source, req.frame, req.quality, req.conf, req.max_width)
    except ValueError as e:
        raise HTTPException(400, str(e))
    job.people = people
    return {"frame": req.frame, "people": people}


class AnalyseReq(BaseModel):
    person: int                # 1-based index shown on the preview
    start: int = 0
    end: int | None = None
    step: int = 1
    quality: str = "fast"
    conf: float = 0.35
    max_width: int = 960
    pose: bool = True
    hands: bool = True


@app.post("/api/jobs/{jid}/analyse")
def start_analysis(jid: str, req: AnalyseReq):
    job = get_job(jid)
    if job.state == "running":
        raise HTTPException(409, "analysis already running")
    match = [p for p in job.people if p["index"] == req.person]
    if not match:
        raise HTTPException(400, f"no person with index {req.person} on the detection frame; "
                                 f"valid: {[p['index'] for p in job.people] or 'none'}")
    src = job.source
    end = src.count - 1 if req.end is None else min(req.end, src.count - 1)
    if req.start > end:
        raise HTTPException(400, "start is after end")
    job.frames, job.done, job.cancel, job.error = [], 0, False, ""
    job.state, job.message = "running", "starting"
    job.params = {**req.model_dump(), "end": end, "box": match[0]["box"],
                  "fps": src.fps, "width": src.width, "height": src.height,
                  "kind": src.kind, "name": job.name, "engine": "server"}

    def run():
        try:
            analysis.analyse(job, src, req.start, end, max(1, req.step), match[0]["box"],
                             req.quality, req.conf, req.max_width, req.pose, req.hands)
            job.state = "cancelled" if job.cancel else "done"
        except Exception as e:  # surface model/IO errors to the UI
            job.state, job.error = "error", f"{type(e).__name__}: {e}"
        finally:
            (job.folder / "results.json").write_text(json.dumps({
                "params": job.params, "frames": job.frames, "state": job.state,
            }), encoding="utf-8")

    job.thread = threading.Thread(target=run, daemon=True)
    job.thread.start()
    return {"ok": True, "total": len(range(req.start, end + 1, max(1, req.step)))}


@app.get("/api/jobs/{jid}/status")
def status(jid: str, since: int = 0):
    job = get_job(jid)
    return {
        "state": job.state, "message": job.message, "error": job.error,
        "done": job.done, "total": job.total, "fps": round(job.fps, 2),
        "params": job.params, "since": since, "frames": job.frames[since:],
    }


@app.post("/api/jobs/{jid}/outputs")
async def save_output(jid: str, name: str, request: Request):
    """Store a rendered output file in jobs/<id>/outputs/ (the browser renders, the server keeps a copy)."""
    job = get_job(jid)
    fname = SAFE.sub("_", Path(name).name)
    if not fname or fname.startswith("."):
        raise HTTPException(400, "bad file name")
    out = job.folder / "outputs"
    out.mkdir(exist_ok=True)
    data = await request.body()
    (out / fname).write_bytes(data)
    return {"path": str(out / fname), "bytes": len(data)}


@app.post("/api/jobs/{jid}/cancel")
def cancel(jid: str):
    job = get_job(jid)
    job.cancel = True
    return {"ok": True}


@app.delete("/api/jobs/{jid}")
def delete(jid: str):
    job = get_job(jid)
    job.cancel = True
    if job.thread and job.thread.is_alive():
        job.thread.join(timeout=10)
    if job.source:
        job.source.close()
    JOBS.pop(jid, None)
    shutil.rmtree(job.folder, ignore_errors=True)
    return {"ok": True}


@app.get("/api/jobs")
def list_jobs():
    out = []
    for folder in sorted(JOBS_DIR.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
        info = folder / "job.json"
        if not info.exists():
            continue
        name = json.loads(info.read_text(encoding="utf-8")).get("name", "")
        out.append({"id": folder.name, "name": name, "analysed": (folder / "results.json").exists(),
                    "modified": time.strftime("%Y-%m-%d %H:%M", time.localtime(folder.stat().st_mtime))})
    return out[:30]


@app.get("/api/health")
def health():
    return {"ok": True, "device": analysis.device()}


class NoCache(StaticFiles):
    async def get_response(self, path, scope):
        r = await super().get_response(path, scope)
        r.headers["Cache-Control"] = "no-cache"
        return r


# The web page itself (the same files are published on GitHub Pages). Mounted last so /api wins.
app.mount("/", NoCache(directory=STATIC, html=True), name="site")


@app.exception_handler(Exception)
async def unhandled(request: Request, exc: Exception):
    return JSONResponse({"detail": f"{type(exc).__name__}: {exc}"}, status_code=500)


if __name__ == "__main__":
    import uvicorn

    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    a = ap.parse_args()
    print(f"Motion Study running at http://{a.host}:{a.port}  (device: {analysis.device()})")
    uvicorn.run(app, host=a.host, port=a.port, log_level="warning")
