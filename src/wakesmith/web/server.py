"""wakesmith web UI (`wakesmith serve`): FastAPI backend + a no-build HTML/JS frontend.

Binds to localhost by default. On any other address every request needs the access token printed
at startup (?token=..., then kept in a cookie): the review page plays private recordings.
"""

from __future__ import annotations

import asyncio
import json
import secrets
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response, StreamingResponse

from .. import assets, evaluate, hardware, jobs, phrases, review
from .. import project as projects
from ..pipeline import STAGES
from ..real import VERDICTS, RealData

HERE = Path(__file__).resolve().parent
PRESETS = {  # samples per class: quick look vs. a model worth deploying
    "quick": {
        "positive": 2000,
        "negative": 2000,
        "positive_test": 500,
        "negative_test": 500,
        "mww_steps": 5000,
        "oww_steps": 10000,
    },
    "standard": {
        "positive": 25000,
        "negative": 25000,
        "positive_test": 2500,
        "negative_test": 2500,
        "mww_steps": 20000,
        "oww_steps": 50000,
    },
}


def create_app(token: str | None = None) -> FastAPI:
    app = FastAPI(title="wakesmith", docs_url=None, redoc_url=None)
    state: dict = {"hw": None, "hw_time": 0.0}

    @app.middleware("http")
    async def auth(request: Request, call_next):
        if token:
            given = request.query_params.get("token") or request.cookies.get("wakesmith_token")
            if not given or not secrets.compare_digest(given, token):
                return JSONResponse({"detail": "access token required"}, status_code=401)
        resp = await call_next(request)
        if token and request.query_params.get("token"):
            resp.set_cookie("wakesmith_token", token, httponly=True, samesite="strict")
        return resp

    def proj(name: str) -> projects.Project:
        try:
            return projects.load(name)
        except FileNotFoundError:
            raise HTTPException(404, f"no project {name}") from None

    # -- pages -------------------------------------------------------------------------------
    @app.get("/", response_class=HTMLResponse)
    def index():
        return (HERE / "index.html").read_text()

    @app.get("/p/{name}/review/", response_class=HTMLResponse)
    def review_page(name: str):
        proj(name)
        return (HERE / "review.html").read_text()

    # -- machine -----------------------------------------------------------------------------
    @app.get("/api/hardware")
    def hw():
        if state["hw"] is None or time.time() - state["hw_time"] > 30:
            h, problem = hardware.detect_safe()
            state["hw"] = {
                **h.as_dict(),
                "problem": problem,
                "tts_devices": h.tts_devices(),
                "tts_batch": h.tts_batch(),
                "workers": h.workers(),
                "disk_free_gb": hardware.disk_free_gb(str(assets.home().parent)),
                "disk_warning": hardware.disk_warning(str(assets.home().parent)),
            }
            state["hw_time"] = time.time()
        return state["hw"]

    @app.get("/api/assets")
    def asset_list():
        return [
            {
                "name": a.name,
                "present": a.present(),
                "gb": round(a.size / 1e9, 2),
                "license": a.license,
                "profile": a.profile,
                "used_for": a.used_for,
            }
            for a in assets.ASSETS.values()
        ]

    @app.get("/api/suggest")
    def suggest(phrase: str, n: int = 50):
        return phrases.suggest(phrase, n)

    @app.get("/api/preview")
    def preview(text: str):
        """One TTS clip of `text` (random voice), as WAV, from a separate CPU process."""
        from ..tts import preview as pv

        if not text.strip() or len(text) > 80:
            raise HTTPException(400, "text must be 1-80 characters")
        try:
            data = pv.wav(text.strip())
        except RuntimeError as e:
            raise HTTPException(503, str(e)) from None
        return Response(data, media_type="audio/wav", headers={"Cache-Control": "no-store"})

    # -- projects ----------------------------------------------------------------------------
    def summary(p: projects.Project) -> dict:
        st = p.state()
        results = {}
        for kind in ("mww", "oww"):
            f = p.path("results", f"{kind}.json")
            if f.exists():
                r = json.loads(f.read_text())
                results[kind] = {
                    "table": r["table"],
                    "positives": r["positives"],
                    "negatives": r["negatives"],
                }
        return {
            "name": p.name,
            "phrase": p.phrase,
            "label": p.display,
            "negatives": p.negatives,
            "real": vars(p.real),
            "samples": vars(p.samples),
            "microwakeword": vars(p.microwakeword),
            "openwakeword": vars(p.openwakeword),
            "data_profile": p.data_profile,
            "stages": {s: st.get(s) for s in STAGES},
            "job": jobs.status(p),
            "results": results,
            "exported": sorted(
                str(f.relative_to(p.path("export")))
                for f in p.path("export").rglob("*")
                if f.is_file()
            )
            if p.path("export").exists()
            else [],
            "review_clips": len(review.load(p)["clips"]),
            "dir": str(p.dir),
        }

    @app.get("/api/projects")
    def list_projects():
        home = projects.projects_home()
        out = []
        for f in sorted(home.glob("*/project.yaml")) if home.exists() else []:
            p = projects.load(f.parent)
            st = p.state()
            out.append(
                {
                    "name": p.name,
                    "label": p.display,
                    "job": jobs.status(p),
                    "done": [s for s in STAGES if (st.get(s) or {}).get("status") == "done"],
                }
            )
        return out

    @app.post("/api/projects")
    async def new_project(request: Request):
        body = await request.json()
        phrase = str(body.get("phrase", "")).strip()
        if not phrase:
            raise HTTPException(400, "phrase required")
        negs = [str(x).strip() for x in body.get("negatives", []) if str(x).strip()]
        try:
            p = projects.create(
                phrase,
                body.get("name") or None,
                label=body.get("label", ""),
                negatives=negs or phrases.suggest(phrase, 50),
            )
        except FileExistsError as e:
            raise HTTPException(409, str(e)) from None
        _apply(p, body)
        return summary(p)

    def _apply(p: projects.Project, body: dict) -> None:
        if "negatives" in body:
            p.negatives = [str(x).strip() for x in body["negatives"] if str(x).strip()]
        for key in ("positives", "negatives_dir"):
            if key in body:
                path = str(body[key] or "").strip()
                if path and not Path(path).expanduser().is_dir():
                    raise HTTPException(400, f"not a folder: {path}")
                if key == "positives":
                    p.real.positives = path
                else:
                    p.real.negatives = path
        preset = PRESETS.get(body.get("preset", ""))
        if preset:
            for k in ("positive", "negative", "positive_test", "negative_test"):
                setattr(p.samples, k, preset[k])
            p.microwakeword.steps, p.openwakeword.steps = preset["mww_steps"], preset["oww_steps"]
        for key, obj in (("microwakeword", p.microwakeword), ("openwakeword", p.openwakeword)):
            for k, v in (body.get(key) or {}).items():
                if k in ("enabled", "cutoff", "threshold"):
                    setattr(obj, k, type(getattr(obj, k))(v))
        p.save()

    @app.get("/api/projects/{name}")
    def get_project(name: str):
        return summary(proj(name))

    @app.put("/api/projects/{name}")
    async def update_project(name: str, request: Request):
        p = proj(name)
        if jobs.status(p)["running"]:
            raise HTTPException(409, "a job is running; stop it before changing settings")
        _apply(p, await request.json())
        return summary(p)

    # -- jobs --------------------------------------------------------------------------------
    @app.post("/api/projects/{name}/train")
    async def train(name: str, request: Request):
        p = proj(name)
        body = await request.json() if await request.body() else {}
        first, last = body.get("from", "generate"), body.get("to", "export")
        if first not in STAGES or last not in STAGES:
            raise HTTPException(400, "unknown stage")
        try:
            return jobs.start(
                p, ["train", str(p.dir), "--from", first, "--to", last], f"train {first} -> {last}"
            )
        except RuntimeError as e:
            raise HTTPException(409, str(e)) from None

    @app.post("/api/projects/{name}/stop")
    def stop(name: str):
        return {"stopped": jobs.stop(proj(name))}

    @app.get("/api/projects/{name}/events")
    async def events(name: str, request: Request):
        """Server-sent events: new log lines, progress and job status, once a second."""
        p = proj(name)
        logf = p.path("logs", "train.log")

        def snapshot(pos: int) -> tuple[int, dict]:
            lines: list[str] = []
            if logf.exists() and logf.stat().st_size > pos:
                with open(logf, "rb") as f:
                    f.seek(pos)
                    chunk = f.read()
                pos += len(chunk)
                lines = chunk.decode("utf-8", "replace").splitlines()
            data = {
                "lines": lines[-200:],
                "progress": jobs.progress(p),
                "job": jobs.status(p),
                "stages": {s: p.state().get(s) for s in STAGES},
            }
            return pos, data

        async def stream():
            pos = max(0, logf.stat().st_size - 6000) if logf.exists() else 0
            while not await request.is_disconnected():
                pos, data = await asyncio.to_thread(snapshot, pos)
                yield f"data: {json.dumps(data)}\n\n"
                await asyncio.sleep(1.0)

        return StreamingResponse(
            stream(), media_type="text/event-stream", headers={"Cache-Control": "no-store"}
        )

    # -- results & export --------------------------------------------------------------------
    @app.post("/api/projects/{name}/test")
    def test(name: str):
        p = proj(name)
        try:
            return jobs.start(p, ["train", str(p.dir), "--from", "test", "--to", "test"], "test")
        except RuntimeError as e:
            raise HTTPException(409, str(e)) from None

    @app.post("/api/projects/{name}/export")
    async def do_export(name: str, request: Request):
        from .. import export

        p = proj(name)
        body = await request.json() if await request.body() else {}
        if "cutoff" in body:
            p.microwakeword.cutoff = float(body["cutoff"])
        if "threshold" in body:
            p.openwakeword.threshold = float(body["threshold"])
        p.save()
        files = export.export(p)
        return [str(f.relative_to(p.path("export"))) for f in files]

    @app.get("/api/projects/{name}/files/{rel:path}")
    def download(name: str, rel: str):
        root = proj(name).path("export").resolve()
        f = (root / rel).resolve()
        if root not in f.parents or not f.is_file():
            raise HTTPException(404, "no such file")
        return FileResponse(f, filename=f.name)

    # -- review page API (used by /p/{name}/review/) ------------------------------------------
    @app.post("/api/projects/{name}/review/build")
    async def review_build(name: str, request: Request):
        p = proj(name)
        body = await request.json() if await request.body() else {}
        kind = body.get("model", "mww")
        args = ["review", str(p.dir), "--model", kind]
        if body.get("threshold") is not None:
            args += ["--threshold", str(float(body["threshold"]))]
        try:
            return jobs.start(p, args, f"review list ({kind})")
        except RuntimeError as e:
            raise HTTPException(409, str(e)) from None

    @app.get("/p/{name}/review/clips.json")
    def review_clips(name: str):
        p = proj(name)
        data = review.load(p)
        return {**data, "phrase": p.display, "decisions": review.decisions(p)}

    @app.get("/p/{name}/review/audio/{i}")
    def review_audio(name: str, i: int):
        p = proj(name)
        clips = review.load(p)["clips"]
        if not 0 <= i < len(clips):
            raise HTTPException(404, "no such clip")
        c = clips[i]
        return FileResponse(RealData(p).file(c["src"], c["file"]), media_type="audio/wav")

    @app.post("/p/{name}/review/mark")
    async def review_mark(name: str, request: Request):
        p = proj(name)
        body = await request.json()
        clips = review.load(p)["clips"]
        try:
            clip = clips[int(body["i"])]
        except (KeyError, ValueError, IndexError):
            raise HTTPException(400, "bad clip index") from None
        verdict = body.get("verdict")
        if verdict is not None and verdict not in VERDICTS:
            raise HTTPException(400, "bad verdict")
        review.mark(p, clip, verdict, str(body.get("note", "")))
        return {"ok": True}

    @app.get("/api/projects/{name}/heldout")
    def heldout(name: str):
        p = proj(name)
        out = {}
        for kind in ("mww", "oww"):
            f = p.path("results", f"{kind}.json")
            if f.exists():
                out[kind] = evaluate.format_table(json.loads(f.read_text()))
        return out

    return app
