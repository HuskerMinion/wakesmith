"""wakesmith web UI (`wakesmith serve`): FastAPI backend + a no-build HTML/JS frontend.

Binds to localhost by default. On any other address every request needs the access token printed
at startup (?token=..., then kept in a cookie): the review page plays private recordings.
"""

from __future__ import annotations

import asyncio
import json
import os
import secrets
import sys
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response, StreamingResponse

from .. import (
    archive,
    assets,
    assistant,
    clipsync,
    commonvoice,
    ensemble,
    eta,
    evaluate,
    hardware,
    jobs,
    languages,
    models,
    mww,
    oww,
    phrases,
    recordings,
    review,
    rounds,
    runs,
    sessions,
    sysstat,
    verdict,
    versions,
)
from .. import project as projects
from ..pipeline import STAGES
from ..real import KIND, VERDICTS, RealData

HERE = Path(__file__).resolve().parent
COMBINED_KINDS = ("mww-combined", "mww-combined-cv")  # results of the combined model (ensemble.py)
TAKE_MAX_BYTES = 2_000_000  # 30 s of 16 kHz 16-bit mono is under 1 MB; room for WAV headers
STATIC = {
    "wakesmith-w.png": "image/png",
    "favicon.png": "image/png",
}  # brand mark, from the branding kit
STATIC |= {  # the brand fonts, Latin subsets (SIL OFL 1.1, see FONTS-LICENSE.txt)
    f"{fam}-latin-{w}-normal.woff2": "font/woff2"
    for fam, weights in (
        ("montserrat", (600, 700)),
        ("inter", (400, 500, 600)),
        ("jetbrains-mono", (400, 500)),
    )
    for w in weights
}
STATIC["FONTS-LICENSE.txt"] = "text/plain; charset=utf-8"
PRESETS = {  # samples per class: quick look vs. a model worth deploying
    "quick": {
        "positive": 2000,
        "negative": 2000,
        "positive_test": 500,
        "negative_test": 500,
        "mww_steps": 5000,
        "mww_candidates": 1,
        "oww_steps": 10000,
    },
    "standard": {
        "positive": 25000,
        "negative": 25000,
        "positive_test": 2500,
        "negative_test": 2500,
        "mww_steps": 20000,
        "mww_candidates": 3,
        "oww_steps": 50000,
    },
}


def guide_text() -> str:
    """README.md: from the source tree when running from one (an editable install's metadata keeps the
    README of the day it was installed), else the copy built into the installed package's metadata."""
    src = HERE.parents[2] / "README.md"
    if src.is_file() and (HERE.parents[2] / "pyproject.toml").is_file():
        return src.read_text()
    from importlib.metadata import PackageNotFoundError, metadata

    try:
        md = metadata("wakesmith")
        # Metadata 2.1+ keeps the long description in the message body, older versions in a field
        text = md.get("Description") or getattr(md, "get_payload", lambda: "")() or ""
    except PackageNotFoundError:
        text = ""
    return text or "The guide isn't available in this installation."


QUEUE_CHECK_S = 30  # the launcher starts the next queued run itself; this catches the rest
ARCHIVE_CHECK_S = 86400  # the auto-archive rule: at startup, then once a day


@asynccontextmanager
async def _lifespan(app: FastAPI):
    """Keep the queue moving when no launcher could: jobs started before the queue existed, or a
    launcher killed together with its job. Also applies the auto-archive rule (archive.py)."""

    async def check():
        last_archive = None
        while True:
            try:
                await asyncio.to_thread(jobs.kick)
            except Exception as e:  # noqa: BLE001 - keep checking; the next try may work
                print(f"wakesmith: job queue check failed: {e}", file=sys.stderr)
            if last_archive is None or time.monotonic() - last_archive >= ARCHIVE_CHECK_S:
                last_archive = time.monotonic()
                try:
                    await asyncio.to_thread(archive.archive_idle)
                except Exception as e:  # noqa: BLE001 - try again tomorrow
                    print(f"wakesmith: auto-archive check failed: {e}", file=sys.stderr)
            await asyncio.sleep(QUEUE_CHECK_S)

    task = asyncio.create_task(check())
    yield
    task.cancel()


def create_app(token: str | None = None) -> FastAPI:
    app = FastAPI(title="wakesmith", docs_url=None, redoc_url=None, lifespan=_lifespan)
    state: dict = {"hw": None, "hw_time": 0.0}

    @app.middleware("http")
    async def auth(request: Request, call_next):
        # Any web page could send requests to a wakesmith on localhost: refuse writes from another site
        # (Origin not this server) and requests for a Host name other than this machine's.
        origin, host = request.headers.get("origin"), request.headers.get("host", "")
        write = request.method not in ("GET", "HEAD", "OPTIONS")
        if write and origin and origin.split("://", 1)[-1].rstrip("/") != host:
            return JSONResponse({"detail": "cross-site request refused"}, status_code=403)
        name = host.split("]")[0] + "]" if host.startswith("[") else host.rsplit(":", 1)[0]
        if not token and name not in ("localhost", "127.0.0.1", "[::1]"):
            return JSONResponse({"detail": "unknown host"}, status_code=403)
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
        except (ValueError, TypeError) as e:  # a hand-edited project.yaml it can't use
            raise HTTPException(409, f"{name}: project.yaml can't be used ({e})") from None

    # -- pages -------------------------------------------------------------------------------
    @app.get("/", response_class=HTMLResponse)
    def index():
        # no-cache: the browser checks for a newer page on every load, so an update shows on a
        # plain reload (the fonts and images stay cached)
        return HTMLResponse(
            (HERE / "index.html").read_text(), headers={"Cache-Control": "no-cache"}
        )

    @app.get("/static/{name}")
    def static(name: str):
        if name not in STATIC:
            raise HTTPException(404, "not found")
        return FileResponse(
            HERE / name, media_type=STATIC[name], headers={"Cache-Control": "max-age=86400"}
        )

    @app.get("/guide")
    def guide():
        """The README of the version that's running (Help page), not whatever a website shows."""
        return Response(guide_text(), media_type="text/plain; charset=utf-8")

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
                "mww_device": mww.train_device(h.gpus),
                "oww_feature_device": oww.feature_device(h.gpus),
                "oww_device": oww.train_device(h.gpus),
                "workers": h.workers(),
                "disk_free_gb": hardware.disk_free_gb(str(assets.home().parent)),
                "disk_warning": hardware.disk_warning(str(assets.home().parent)),
                "ram_warning": hardware.ram_warning(h),
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
            if not a.optional or a.present()
        ]

    # -- downloading the training data from the page (what `wakesmith setup` downloads) ----------
    dl = {"running": False, "current": None, "done": [], "error": None}

    def _part_bytes(a) -> int:
        target = a.local.with_suffix(".zip") if a.unzip else a.local
        part = target.with_name(target.name + ".part")
        try:
            return part.stat().st_size if part.is_file() else 0
        except OSError:
            return 0

    def _download_all() -> None:
        try:
            for a in [a for a in assets.required() if not a.present()]:
                dl["current"] = a.name
                assets.fetch(a, progress=False)
                dl["done"].append(a.name)
        except Exception as e:  # noqa: BLE001 - shown on the page; the next press resumes
            dl["error"] = f"{dl['current']}: {e}"
        finally:
            dl["running"], dl["current"] = False, None

    @app.get("/api/assets/download")
    def download_status():
        cur = assets.ASSETS.get(dl["current"]) if dl["current"] else None
        missing = [a for a in assets.required() if not a.present()]
        return {
            **dl,
            "missing": [a.name for a in missing],
            "missing_gb": round(sum(a.size for a in missing) / 1e9, 1),
            "current_bytes": _part_bytes(cur) if cur else 0,
            "current_size": cur.size if cur else 0,
        }

    dl_lock = threading.Lock()

    @app.post("/api/assets/download")
    def download_start():
        with dl_lock:  # two presses at once start one download
            if dl["running"]:
                raise HTTPException(409, "already downloading")
            dl.update(running=True, done=[], error=None)
        threading.Thread(target=_download_all, daemon=True).start()
        return download_status()

    def offered(language: str) -> None:
        """400 unless new wake words may be in this language (experimental ones only when turned on)."""
        if language not in [x.code for x in languages.available()]:
            raise HTTPException(
                400,
                f"language {language!r} isn't offered (experimental languages: Data & licenses page)",
            )

    @app.get("/api/suggest")
    def suggest(phrase: str, n: int = 50, language: str = "en"):
        offered(language)
        return phrases.suggest(phrase, n, language=language)

    @app.get("/api/languages")
    def language_list():
        """The languages a new wake word can be in; experimental ones only when turned on."""
        return {
            "languages": [
                {"code": x.code, "name": x.name, "experimental": x.experimental}
                for x in languages.available()
            ],
            "experimental": languages.experimental_on(),
        }

    @app.post("/api/languages")
    async def language_settings(request: Request):
        try:
            body = await request.json()
        except ValueError:
            body = None
        if not isinstance(body, dict) or not isinstance(body.get("experimental"), bool):
            raise HTTPException(400, 'send {"experimental": true} or {"experimental": false}')
        languages.set_experimental(body["experimental"])
        return language_list()

    @app.get("/api/preview")
    def preview(text: str, language: str = "en"):
        """One TTS clip of `text` (random voice), as WAV, from a separate CPU process."""
        from ..tts import preview as pv

        if not text.strip() or len(text) > 80:
            raise HTTPException(400, "text must be 1-80 characters")
        offered(language)
        try:
            data = pv.wav(text.strip(), language)
        except RuntimeError as e:
            raise HTTPException(503, str(e)) from None
        return Response(data, media_type="audio/wav", headers={"Cache-Control": "no-store"})

    # -- projects ----------------------------------------------------------------------------
    def summary(p: projects.Project) -> dict:
        results = {}
        for kind in ("mww", "oww", "mww-cv", "oww-cv", *COMBINED_KINDS):
            f = p.path("results", f"{kind}.json")
            if f.exists():
                results[kind] = evaluate.brief(json.loads(f.read_text()))
        return {
            "name": p.name,
            "phrase": p.phrase,
            "label": p.title,
            "word": p.display,
            "version": p.version,
            "note": p.note,
            "negatives": p.negatives,
            "real": vars(p.real),
            "samples": vars(p.samples),
            "microwakeword": vars(p.microwakeword),
            "openwakeword": vars(p.openwakeword),
            "commonvoice": vars(p.commonvoice),
            "uses_commonvoice": p.uses_commonvoice,  # off for other languages, whatever the setting
            "conversational": vars(p.conversational),
            "data_profile": p.data_profile,
            "language": p.language,
            "stages": jobs.stages(p, STAGES),
            "estimates": eta.estimates(p),
            "job": jobs.status(p),
            "results": results,
            "exported": sorted(
                str(f.relative_to(p.path("export")))
                for f in p.path("export").rglob("*")
                if f.is_file() and f.parent.name in (*models.KINDS, "export")
            )
            if p.path("export").exists()
            else [],
            "review_clips": len(review.load(p)["clips"]),
            "live": models.live(p),
            "archive": archive.info(p),
            "combined": _combined(p),
            "verdict": verdict.load(p),
            "live_version": {
                k: {"title": lv[0].title, "sha": lv[1]["sha"]}
                for k in ("mww", "oww")
                if (lv := verdict.live_version(p, k))
            },
            "dir": str(p.dir),
        }

    def _combined(p: projects.Project) -> dict | None:
        """The combined model (ensemble.py), if built: candidates, size, arena, note."""
        info = ensemble.info(p)
        return {**info, "note": ensemble.note(info)} if info else None

    @app.get("/api/projects")
    def list_projects():
        out = []
        found = projects.all_projects()
        projects.assign_versions(found)
        for p in found:
            st = jobs.stages(p, STAGES)
            job = jobs.status(p)
            if not job["running"] and jobs.outside_run(p):
                job = {**job, "running": True, "outside": True}
            out.append(
                {
                    "name": p.name,
                    "label": p.display,
                    "title": p.title,
                    "phrase": p.phrase,
                    "family": p.family,
                    "version": p.version,
                    "note": p.note,
                    "created": projects.created(p),
                    "last": _last_run(p, job),
                    "job": job,
                    "done": [s for s in STAGES if (st.get(s) or {}).get("status") == "done"],
                    "failed": [s for s in STAGES if (st.get(s) or {}).get("status") == "failed"],
                    "models": {"mww": p.microwakeword.enabled, "oww": p.openwakeword.enabled},
                    "brief": _brief(p),
                    "live": models.live(p),
                    "archived": bool((archive.info(p) or {}).get("archived")),
                    "to_check": rounds.to_check(clips := rounds.load(p)),
                    "new_since": rounds.newer_than(clips, projects.created(p)),
                    "verdict": {
                        k: {
                            "better": v.get("better"),
                            "sentence": v.get("sentence") or v.get("error"),
                        }
                        for k, v in verdict.load(p).items()
                    },
                }
            )
        return out

    def _last_run(p: projects.Project, job: dict) -> float | None:
        """When the project last trained: its job's start or end, else the newest stage's start."""
        times = [job.get("finished"), job.get("started")]
        times += [v.get("started") for v in p.state().values() if isinstance(v, dict)]
        times = [t for t in times if isinstance(t, (int, float))]
        if p.path(
            "state.json"
        ).exists():  # a finished stage keeps no start time; the file was written then
            times.append(p.path("state.json").stat().st_mtime)
        return max(times) if times else None

    def _brief(p: projects.Project) -> dict:
        """Test result at each model's chosen cutoff, for the wake words table. Without takes of the
        phrase of your own, the caught side comes from the Common Voice speakers' test."""

        def at(kind: str, tag: str, cut: float):
            f = p.path("results", f"{kind}{tag}.json")
            if not f.exists():
                return None
            r = json.loads(f.read_text())
            row = min(r["table"], key=lambda row: abs(row["threshold"] - cut), default=None)
            return (r, row) if row else None

        out = {}
        for kind, cut in (("mww", p.microwakeword.cutoff), ("oww", p.openwakeword.threshold)):
            own, cv = at(kind, "", cut), at(kind, "-cv", cut)
            pos = own if own and own[0]["positives"] else cv  # who says the phrase
            neg = own if own and own[0]["negatives"] else cv  # what isn't it
            if not pos and not neg:
                continue
            row = (pos or neg)[1]
            out[kind] = {
                "cutoff": row["threshold"],
                "caught": pos[1]["caught"] if pos else 0,
                "positives": pos[0]["positives"] if pos else 0,
                "false": neg[1]["false"] if neg else 0,
                "negatives": neg[0]["negatives"] if neg else 0,
            }
        return out

    @app.post("/api/projects")
    async def new_project(request: Request):
        body = await request.json()
        phrase = str(body.get("phrase", "")).strip()
        if not phrase:
            raise HTTPException(400, "phrase required")
        negs = [str(x).strip() for x in body.get("negatives", []) if str(x).strip()]
        language = str(body.get("language") or "en")
        offered(language)
        try:
            p = projects.create(
                phrase,
                body.get("name") or None,
                label=body.get("label", ""),
                negatives=negs or phrases.suggest(phrase, 50, language=language),
                language=language,
            )
        except FileExistsError as e:
            raise HTTPException(409, str(e)) from None
        # negatives are settled above (the given ones, or suggestions when none were given): an
        # empty list from the form must not wipe the suggestions again
        _apply(p, {k: v for k, v in body.items() if k != "negatives"})
        return summary(p)

    def _apply(p: projects.Project, body: dict) -> None:
        for key in ("label", "note"):  # the wake word's name; what this version tried
            if key in body:
                setattr(p, key, " ".join(str(body[key] or "").split())[:120])
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
            p.microwakeword.candidates = preset["mww_candidates"]
        cv = body.get("commonvoice") or {}
        if cv.get("enabled") and p.language != "en":
            raise HTTPException(400, "Common Voice here is the English set: not for this wake word")
        for k in ("enabled", "train"):
            if k in cv:
                setattr(p.commonvoice, k, bool(cv[k]))
        if "dataset" in cv:
            p.commonvoice.dataset = str(cv["dataset"] or "").strip()
        conv = body.get("conversational") or {}
        if "clips" in conv:
            try:
                n = int(conv["clips"] or 0)
            except (TypeError, ValueError):
                raise HTTPException(400, "everyday sentences must be a whole number") from None
            if not 0 <= n <= 100000:
                raise HTTPException(400, "everyday sentences: 0 to 100,000 clips")
            p.conversational.clips = n
            if not n and p.state().get("conversational"):
                p.set_state(conversational=None)  # turned off: an earlier failure no longer counts
        for k in ("llm_url", "model"):
            if k in conv:
                setattr(p.conversational, k, str(conv[k] or "").strip())
        if "weight" in conv:
            try:
                p.conversational.weight = float(conv["weight"])
            except (TypeError, ValueError):
                raise HTTPException(
                    400, "the everyday sentences' weight must be a number"
                ) from None
        for key, obj in (("microwakeword", p.microwakeword), ("openwakeword", p.openwakeword)):
            for k, v in (body.get(key) or {}).items():
                if k in ("enabled", "cutoff", "threshold", "combine", "house_rirs"):
                    setattr(obj, k, type(getattr(obj, k))(v))
                elif k == "frontend" and key == "microwakeword":
                    if v not in ("tflm", "zserge"):
                        raise HTTPException(400, "frontend must be tflm or zserge")
                    obj.frontend = v
        p.save()

    @app.get("/api/projects/{name}")
    def get_project(name: str):
        return summary(proj(name))

    @app.get("/api/projects/{name}/size")
    def project_size(name: str):
        p = proj(name)
        rec = {k: len(recordings.listing(p, k)["takes"]) for k in recordings.KINDS}
        return {"bytes": projects.size_bytes(p), "recorded_takes": rec, "dir": str(p.dir)}

    @app.delete("/api/projects/{name}")
    def delete_project(name: str):
        p = proj(name)
        if jobs.status(p)["running"]:
            raise HTTPException(409, "a job is running; stop it before deleting the wake word")
        try:
            freed = projects.delete(p)
        except ValueError as e:
            raise HTTPException(400, str(e)) from None
        jobs.drop_project(p.name)
        return {"deleted": p.name, "freed_bytes": freed}

    # -- storage: sizes and archiving --------------------------------------------------------
    @app.get("/api/storage")
    def storage():
        rows = archive.storage()
        s = archive.settings()
        rule = (
            s["archive_after_days"] or archive.DEFAULT_DAYS
        )  # "Archive all idle" works when it's off
        for r in rows:
            r["auto_skip"] = archive.eligible(r, rule)
        return {"projects": rows, "settings": s, "rule_days": rule}

    @app.put("/api/storage")
    async def storage_settings(request: Request):
        try:
            return archive.save_settings(await request.json())
        except ValueError as e:
            raise HTTPException(400, str(e)) from None

    @app.post("/api/projects/{name}/archive")
    async def archive_project(name: str, request: Request):
        p = proj(name)
        body = await request.json() if await request.body() else {}
        try:
            return await asyncio.to_thread(archive.archive, p, bool(body.get("clips")))
        except archive.Busy as e:
            raise HTTPException(409, str(e)) from None

    @app.post("/api/storage/archive-idle")
    async def archive_idle():
        """The auto rule's pick, now (with its default days when the rule is off)."""
        days = archive.settings()["archive_after_days"] or archive.DEFAULT_DAYS
        return {"archived": await asyncio.to_thread(archive.archive_idle, days, "you")}

    @app.get("/api/projects/{name}/new-version")
    def new_version_preview(name: str):
        return versions.preview(proj(name))

    @app.post("/api/projects/{name}/new-version")
    async def new_version(name: str, request: Request):
        """Make the next version from this one and queue its run: device clips synced and labeled,
        the newest day held out, trained from the recordings on, compared with the live version."""
        p = proj(name)
        body = await request.json() if await request.body() else {}
        q = versions.create(p, str(body.get("note") or "").strip()[:120])
        item = jobs.enqueue(
            q, ["prepare-version", str(q.dir)], "new version: device clips, then train"
        )
        return {"name": q.name, "title": q.title, "queued": item}

    @app.get("/api/projects/{name}/everyday")
    def everyday(name: str, kind: str = "mww"):
        out = verdict.everyday(proj(name), kind)
        if not out:
            raise HTTPException(404, "no held-out results yet")
        return out

    @app.post("/api/projects/{name}/verdict")
    def make_verdict(name: str):
        """Score the live version's model on this version's held-out clips (a minute or two)."""
        p = proj(name)
        if jobs.status(p)["running"]:
            raise HTTPException(409, "a job is running; compare after it ends")
        out = {}
        for kind, on in (("mww", p.microwakeword.enabled), ("oww", p.openwakeword.enabled)):
            if on:
                v = verdict.compare(p, kind, log=lambda *_: None)
                if v:
                    out[kind] = v
        if not out:
            raise HTTPException(
                400,
                "nothing to compare: test this version first, and mark a model live on the version your devices run",
            )
        return out

    @app.put("/api/projects/{name}")
    async def update_project(name: str, request: Request):
        p = proj(name)
        body = await request.json()
        if set(body) <= {"label", "note"}:  # names can change while it trains
            for key in ("label", "note"):
                if key in body:
                    setattr(p, key, " ".join(str(body[key] or "").split())[:120])
            p.save()
            if "label" in body:  # the name belongs to the wake word: every version takes it
                for kin in projects.all_projects():
                    if kin.family == p.family and kin.name != p.name and kin.label != p.label:
                        kin.label = p.label
                        kin.save()
            return summary(p)
        if jobs.status(p)["running"]:
            raise HTTPException(409, "a job is running; stop it before changing settings")
        _apply(p, body)
        return summary(p)

    # -- takes recorded in the browser --------------------------------------------------------
    def _idle(p: projects.Project) -> None:
        if jobs.status(p)["running"]:
            raise HTTPException(409, "a job is running; recordings can change after it ends")

    @app.get("/api/projects/{name}/recordings")
    def rec_list(name: str):
        p = proj(name)
        return {k: recordings.listing(p, k) for k in recordings.KINDS}

    @app.post("/api/projects/{name}/recordings/{kind}")
    async def rec_add(
        name: str,
        kind: str,
        request: Request,
        condition: str = "",
        heldout: bool = False,
        speaker: str = "",
    ):
        p = proj(name)
        _idle(p)
        body = bytearray()
        async for chunk in request.stream():  # counted as it arrives: no Content-Length is trusted
            body += chunk
            if len(body) > TAKE_MAX_BYTES:
                raise HTTPException(413, "a take is at most 30 s of 16 kHz audio")
        try:
            return recordings.add(p, kind, bytes(body), condition, heldout, speaker)
        except PermissionError as e:
            raise HTTPException(409, str(e)) from None
        except (ValueError, RuntimeError) as e:  # soundfile raises RuntimeError on non-WAV bytes
            raise HTTPException(400, str(e)) from None

    @app.post("/api/projects/{name}/recordings/{kind}/use")
    def rec_use(name: str, kind: str):
        p = proj(name)
        _idle(p)
        try:
            recordings.use(p, kind)
        except ValueError as e:
            raise HTTPException(400, str(e)) from None
        return recordings.listing(p, kind)

    @app.get("/api/projects/{name}/recordings/{kind}/{file}")
    def rec_audio(name: str, kind: str, file: str):
        try:
            return FileResponse(recordings.path(proj(name), kind, file), media_type="audio/wav")
        except (ValueError, FileNotFoundError):
            raise HTTPException(404, "no such take") from None

    @app.delete("/api/projects/{name}/recordings/{kind}/{file}")
    def rec_delete(name: str, kind: str, file: str):
        p = proj(name)
        _idle(p)
        try:
            recordings.delete(p, kind, file)
        except (ValueError, FileNotFoundError):
            raise HTTPException(404, "no such take") from None
        return {"deleted": file}

    @app.post("/api/projects/{name}/recordings/{kind}/{file}/speaker")
    async def rec_speaker(name: str, kind: str, file: str, request: Request):
        p = proj(name)
        _idle(p)
        try:
            recordings.set_speaker(p, kind, file, str((await request.json()).get("speaker", "")))
        except (ValueError, FileNotFoundError):
            raise HTTPException(404, "no such take") from None
        return {"file": file}

    @app.post("/api/projects/{name}/recordings/{kind}/{file}/heldout")
    async def rec_heldout(name: str, kind: str, file: str, request: Request):
        p = proj(name)
        _idle(p)
        try:
            recordings.set_heldout(p, kind, file, bool((await request.json()).get("held")))
        except (ValueError, FileNotFoundError):
            raise HTTPException(404, "no such take") from None
        return {"file": file}

    @app.get("/api/projects/{name}/candidates")
    def candidates(name: str):
        """Best of N: mww/candidates/pick.json (empty when the project trains one candidate), plus
        "previous": the previous round kept in mww/candidates.previous, if any."""
        p = proj(name)
        f = p.path("mww", "candidates", "pick.json")
        out = json.loads(f.read_text()) if f.exists() else {}
        if prev := ensemble.previous(p):
            out["previous"] = prev
        return out

    @app.post("/api/projects/{name}/combine")
    async def combine(name: str, request: Request):
        """Build the combined model from the finished candidates, then score it (a job, CPU only).
        Body (optional): {"previous": true} for the previous round's candidates."""
        p = proj(name)
        body = await request.json() if await request.body() else {}
        prev = bool(body.get("previous"))
        if len(ensemble.available(p, previous=prev)) < 2:
            raise HTTPException(409, "combining needs at least 2 trained candidates")
        args = ["combine", str(p.dir), *(["--previous"] if prev else [])]
        try:
            return jobs.start(
                p, args, "combine previous candidates" if prev else "combine candidates"
            )
        except RuntimeError as e:
            raise HTTPException(409, str(e)) from None

    @app.get("/api/projects/{name}/sets")
    def result_sets(name: str):
        """Held-out results per test set (source folder), at every cutoff of the results table
        and the chosen one. Common Voice takes are added to their model's phrase sets."""
        p = proj(name)
        chosen = {"mww": p.microwakeword.cutoff, "oww": p.openwakeword.threshold}
        chosen["mww-combined"] = chosen["mww"]
        out = {}
        for kind, cut in chosen.items():
            f = p.path("results", f"{kind}.json")
            if not f.exists():
                continue
            r = json.loads(f.read_text())
            if "pos" not in r:
                continue
            cutoffs = sorted({row["threshold"] for row in r["table"]} | {cut})
            sets = evaluate.by_set(r, cutoffs)
            cv = p.path("results", f"{kind}-cv.json")
            if cv.exists():
                rc = json.loads(cv.read_text())
                takes = rc.get("pos") or {}
                sets += evaluate.by_set(
                    {"pos": takes, "set_of": dict.fromkeys(takes, "cv")}, cutoffs
                )
            # "report": the half of the held-out clips not used to pick the kept candidate
            out[kind] = {"cutoffs": cutoffs, "sets": sets, "half": r.get("half")}
        notes = _set_notes(p)
        for k in out.values():
            for s in k["sets"]:
                s["note"] = notes.get(s["id"], "")
        return out

    # -- what a version learns from: every source and group, counts, and its clips to play ----------
    def _groups(p: projects.Project) -> list[dict]:
        real = RealData(p)
        out = []
        for src, folder in real.sources.items():
            try:
                files, held, rows = real.listing(src)
            except (OSError, ValueError) as e:
                out.append(
                    {
                        "src": src,
                        "kind": KIND[src],
                        "folder": str(folder),
                        "error": str(e),
                        "groups": [],
                    }
                )
                continue
            by: dict[str, dict] = {}
            for f in files:
                g = by.setdefault(
                    evaluate.set_id(src, f),
                    {"id": evaluate.set_id(src, f), "train": 0, "held": 0, "notes": set()},
                )
                g["held" if f in held else "train"] += 1
                extra = [v for k, v in rows.get(f, {}).items() if k != "file" and v]
                if extra and len(g["notes"]) < 3:
                    g["notes"].add(extra[0].strip()[:80])
            groups = [
                {**g, "notes": sorted(g["notes"])}
                for g in sorted(by.values(), key=lambda g: g["id"])
            ]
            out.append({"src": src, "kind": KIND[src], "folder": str(folder), "groups": groups})
        return out

    @app.get("/api/projects/{name}/learns")
    def learns(name: str):
        """Everything this version learns from and is tested on, by source and group."""
        p = proj(name)
        synth = {}
        for kind in ("positive", "negative"):  # clips/<kind>_train and <kind>_test
            dirs = [p.path("clips", f"{kind}_{s}") for s in ("train", "test")]
            synth[kind] = sum(
                sum(1 for e in os.scandir(d) if e.name.endswith(".wav")) for d in dirs if d.is_dir()
            )
        return {
            "sources": _groups(p),
            "synthetic": synth,
            "sound_alikes": len(p.negatives),
            "corrections": len(review.decisions(p)),
        }

    @app.get("/api/projects/{name}/learns/clips")
    def learns_clips(name: str, group: str, offset: int = 0, limit: int = 40):
        p = proj(name)
        real = RealData(p)
        src = group.split("/")[0]
        if src not in real.sources:
            raise HTTPException(404, "no such source")
        files, held, rows = real.listing(src)
        mine = [f for f in files if evaluate.set_id(src, f) == group]
        page = mine[offset : offset + max(1, min(limit, 200))]
        return {
            "total": len(mine),
            "clips": [
                {
                    "f": f,
                    "held": f in held,
                    "said": next(
                        (v for k, v in rows.get(f, {}).items() if k in ("said", "text") and v), ""
                    ),
                }
                for f in page
            ],
        }

    @app.get("/api/projects/{name}/learns/audio")
    def learns_audio(name: str, src: str, f: str):
        p = proj(name)
        real = RealData(p)
        if src not in real.sources or f not in set(real.listing(src)[0]):
            raise HTTPException(404, "no such clip")
        return FileResponse(real.file(src, f), media_type="audio/wav")

    def _set_notes(p: projects.Project) -> dict[str, str]:
        """A test set's description from labels.csv, when all its held-out clips share one
        (e.g. "neg2_test (synthetic look-alike)")."""
        notes: dict[str, set[str]] = {}
        real = RealData(p)
        try:
            for src in real.sources:
                _, held, rows = real.listing(src)
                for f in held:
                    extra = [v for k, v in rows.get(f, {}).items() if k != "file" and v]
                    notes.setdefault(evaluate.set_id(src, f), set()).add(
                        extra[0].strip() if extra else ""
                    )
        except (OSError, ValueError):
            return {}
        return {k: next(iter(v)) for k, v in notes.items() if len(v) == 1 and "" not in v}

    # -- device clips (rounds.py) -------------------------------------------------------------
    @app.get("/api/projects/{name}/devices")
    def devices(name: str, holdout_from: str = ""):
        p = proj(name)
        meta = p.path("devices", "round.json")
        return {
            "summary": rounds.summary(p, holdout_from or None),
            "clips": rounds.load(p),
            "applied": json.loads(meta.read_text()) if meta.exists() else None,
            "sync": _sync_info(p),
        }

    def _sync_info(p: projects.Project) -> dict:
        last = clipsync.last_result(p)
        return {
            **vars(p.devices),
            "last": {**last, "text": clipsync.describe(last)} if last else None,
        }

    @app.put("/api/projects/{name}/devices/sync")
    async def devices_sync_settings(name: str, request: Request):
        """Where `clips sync` looks and which clips are this word's (project.yaml devices:)."""
        p, body = proj(name), await request.json()
        source = str(body.get("source", p.devices.source)).strip()
        if source and not Path(source).expanduser().is_dir():
            raise HTTPException(400, f"not a folder: {source}")
        p.devices.source = source
        p.devices.model_prefix = str(body.get("model_prefix", p.devices.model_prefix)).strip()
        p.devices.since = str(body.get("since", p.devices.since)).strip()
        p.save()
        return _sync_info(p)

    @app.post("/api/projects/{name}/devices/sync")
    def devices_sync(name: str):
        """Sync now: `clips sync` as a background job (import the new clips, transcribe them)."""
        p = proj(name)
        if not p.devices.source:
            raise HTTPException(400, "choose the folder your device clips arrive in first")
        try:
            return jobs.start(p, ["clips", "sync", str(p.dir)], "sync device clips")
        except RuntimeError as e:
            raise HTTPException(409, str(e)) from None

    @app.get("/api/clips-schedule")
    async def clips_schedule():
        return await asyncio.to_thread(clipsync.status)

    @app.put("/api/clips-schedule")
    async def clips_schedule_set(request: Request):
        """The daily sync of every wake word with a source folder: {on, at: "HH:MM"}."""
        body = await request.json()
        try:
            if body.get("on"):
                return await asyncio.to_thread(clipsync.install, str(body.get("at") or "03:30"))
            return await asyncio.to_thread(clipsync.remove)
        except ValueError as e:
            raise HTTPException(400, str(e)) from None
        except (RuntimeError, OSError) as e:
            raise HTTPException(500, f"could not change the daily sync: {e}") from None

    @app.post("/api/projects/{name}/devices/import")
    async def devices_import(name: str, request: Request):
        body = await request.json()
        try:
            return rounds.import_folder(proj(name), str(body.get("folder", "")).strip())
        except ValueError as e:
            raise HTTPException(400, str(e)) from None

    @app.post("/api/projects/{name}/devices/transcribe")
    def devices_transcribe(name: str):
        p = proj(name)
        try:
            return jobs.start(p, ["clips", "transcribe", str(p.dir)], "transcribe device clips")
        except RuntimeError as e:
            raise HTTPException(409, str(e)) from None

    @app.post("/api/projects/{name}/devices/labels")
    async def devices_labels(name: str, request: Request):
        body = await request.json()
        try:
            return {"changed": rounds.label(proj(name), body.get("labels", {}))}
        except ValueError as e:
            raise HTTPException(400, str(e)) from None

    @app.post("/api/projects/{name}/devices/accept")
    def devices_accept(name: str):
        return {"changed": rounds.accept_suggestions(proj(name))}

    @app.post("/api/projects/{name}/devices/apply")
    async def devices_apply(name: str, request: Request):
        body = await request.json()
        frm = str(body.get("holdout_from", "")).strip()
        if not frm:
            raise HTTPException(400, "choose where the held-out days start")
        return rounds.apply(proj(name), frm)

    @app.get("/api/projects/{name}/devices/audio")
    def devices_audio(name: str, id: str):
        try:
            f = rounds.file(proj(name), id)
        except ValueError:
            raise HTTPException(404, "no such clip") from None
        if not f.is_file():
            raise HTTPException(404, "no such clip")
        return FileResponse(f, media_type="audio/wav")

    # -- jobs --------------------------------------------------------------------------------
    @app.post("/api/projects/{name}/train")
    async def train(name: str, request: Request):
        p = proj(name)
        body = await request.json() if await request.body() else {}
        first, last = body.get("from", "generate"), body.get("to", "export")
        if first not in STAGES or last not in STAGES:
            raise HTTPException(400, "unknown stage")
        args, title = (
            ["train", str(p.dir), "--from", first, "--to", last],
            f"train {first} -> {last}",
        )
        if body.get("queue"):  # after the runs already waiting; starts now if nothing is running
            return {"queued": jobs.enqueue(p, args, title)}
        try:
            return jobs.start(p, args, title)
        except RuntimeError as e:
            raise HTTPException(409, str(e)) from None

    @app.post("/api/projects/{name}/stop")
    def stop(name: str):
        return {"stopped": jobs.stop(proj(name))}

    @app.get("/api/queue")
    def queue():
        """Runs waiting, jobs running now (any project) and the last finished ones."""
        running = []
        for p in jobs.running_all():
            j = jobs.status(p)
            if j["running"]:
                running.append(
                    {
                        "project": p.name,
                        "label": p.title,
                        "title": j.get("title", ""),
                        "started": j.get("started"),
                    }
                )
                continue
            # started from the command line or a script
            st = [
                v
                for v in p.state().values()
                if isinstance(v, dict) and v.get("status") == "running"
            ]
            running.append(
                {
                    "project": p.name,
                    "label": p.title,
                    "title": "started outside this page",
                    "started": min((v.get("started", 0) for v in st), default=None),
                    "outside": True,
                }
            )
        return {
            "queue": jobs.queue(),
            "running": running,
            "history": jobs.history(),
            "max_jobs": jobs.max_jobs(),
            "now": time.time(),
        }

    @app.delete("/api/queue/{item}")
    def queue_remove(item: str):
        if not jobs.dequeue(item):
            raise HTTPException(404, "that run is no longer waiting")
        return {"removed": item}

    @app.post("/api/queue/{item}/move")
    async def queue_move(item: str, request: Request):
        try:
            to = int((await request.json())["to"])
        except (KeyError, TypeError, ValueError):
            raise HTTPException(400, "to: the new position (0 = next)") from None
        if not jobs.move(item, to):
            raise HTTPException(404, "that run is no longer waiting")
        return {"queue": jobs.queue()}

    @app.get("/api/projects/{name}/events")
    async def events(name: str, request: Request):
        """Server-sent events: new log lines, progress and job status, once a second."""
        p = proj(name)
        logf = p.path("logs", "train.log")
        est = {"at": 0.0, "value": {}}

        def snapshot(pos: int) -> tuple[int, dict]:
            if time.time() - est["at"] > 30:  # listing held-out clips isn't free: not every second
                est["at"], est["value"] = time.time(), eta.estimates(p)
            lines: list[str] = []
            if logf.exists() and logf.stat().st_size > pos:
                with open(logf, "rb") as f:
                    f.seek(pos)
                    chunk = f.read()
                pos += len(chunk)
                lines = chunk.decode("utf-8", "replace").splitlines()
            job = jobs.status(p)
            data = {
                "lines": lines[-200:],
                "progress": jobs.progress(p),
                "job": job,
                "sys": sysstat.snapshot(job),
                "stages": jobs.stages(p, STAGES),
                "estimates": est["value"],
                "now": time.time(),
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
        combined = body.get("combined")  # the combined model or the kept candidate; absent: setting
        try:
            export.use_combined(p, None if combined is None else bool(combined))
        except ValueError as e:
            raise HTTPException(409, str(e)) from None
        p.save()
        files = export.export(p, combined=None if combined is None else bool(combined))
        return [str(f.relative_to(p.path("export"))) for f in files]

    @app.get("/api/projects/{name}/files/{rel:path}")
    def download(name: str, rel: str):
        root = proj(name).path("export").resolve()
        f = (root / rel).resolve()
        if root not in f.parents or not f.is_file():
            raise HTTPException(404, "no such file")
        return FileResponse(f, filename=f.name)

    def _zip(p: projects.Project, export_dir: Path, kind: str, zname: str) -> Response:
        """One model's export folder (plus the license note) as a zip: both models are named
        <project>.tflite, so single downloads of the two collide in the browser."""
        import io
        import zipfile

        d = export_dir / kind
        if kind not in models.KINDS or not d.is_dir():
            raise HTTPException(404, "nothing exported for that model")
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
            for f in sorted(d.iterdir()):
                z.write(f, f.name)
            note = export_dir / "LICENSE-NOTE.txt"
            if note.exists():
                z.write(note, note.name)
        return Response(
            buf.getvalue(),
            media_type="application/zip",
            headers={"Content-Disposition": f'attachment; filename="{zname}"'},
        )

    @app.get("/api/projects/{name}/zip/{kind}")
    def download_zip(name: str, kind: str):
        p = proj(name)
        return _zip(p, p.path("export"), kind, f"{p.name}-{kind}.zip")

    # -- Mozilla Common Voice ---------------------------------------------------------------------
    @app.get("/api/projects/{name}/commonvoice")
    def cv_status(name: str):
        """Is the dataset here, and how many takes and speakers have the phrase (a few seconds the
        first time for a phrase: it reads the transcripts)."""
        p = proj(name)
        ds = commonvoice.dataset(p)
        out = {
            "dataset": str(ds),
            "present": commonvoice.present(ds),
            "settings": vars(p.commonvoice),
            "language": p.language,
        }
        if p.language != "en":
            out["present"] = False  # the dataset here is the English one
        elif out["present"]:
            out.update(commonvoice.count(ds, p.phrase))
        return out

    # -- past runs ------------------------------------------------------------------------------
    @app.get("/api/projects/{name}/runs")
    def run_list(name: str):
        return runs.listing(proj(name))

    def _run(p: projects.Project, rid: str) -> Path:
        try:
            return runs.path(p, rid)
        except (ValueError, FileNotFoundError):
            raise HTTPException(404, "no such run") from None

    @app.get("/api/projects/{name}/runs/{rid}/zip/{kind}")
    def run_zip(name: str, rid: str, kind: str):
        p = proj(name)
        return _zip(p, _run(p, rid) / "export", kind, f"{p.name}-{rid}-{kind}.zip")

    @app.post("/api/projects/{name}/runs/{rid}/label")
    async def run_label(name: str, rid: str, request: Request):
        p = proj(name)
        _run(p, rid)
        runs.set_label(p, rid, str((await request.json()).get("label", "")))
        return {"id": rid}

    @app.delete("/api/projects/{name}/runs/{rid}")
    def run_delete(name: str, rid: str):
        p = proj(name)
        _run(p, rid)
        runs.delete(p, rid)
        return {"deleted": rid}

    # -- model history (models.py) ----------------------------------------------------------------
    @app.get("/api/projects/{name}/models")
    def model_list(name: str):
        """Every model this project exported, newest first."""
        return sorted(models.shown(proj(name)), key=lambda e: e["created"], reverse=True)

    @app.post("/api/projects/{name}/models/backfill")
    def model_backfill(name: str):
        return {"added": len(models.backfill(proj(name)))}

    @app.post("/api/projects/{name}/models/{sha}/status")
    async def model_status(name: str, sha: str, request: Request):
        p = proj(name)
        try:
            return models.set_status(p, sha, str((await request.json()).get("status", "")))
        except KeyError:
            raise HTTPException(404, "no such model") from None
        except ValueError as e:
            raise HTTPException(400, str(e)) from None

    @app.post("/api/projects/{name}/models/{sha}/notes")
    async def model_notes(name: str, sha: str, request: Request):
        p = proj(name)
        try:
            return models.set_notes(p, sha, str((await request.json()).get("notes", "")))
        except KeyError:
            raise HTTPException(404, "no such model") from None

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

    # -- optional AI assistant (assistant.py): off unless turned on, advice only ------------------
    def _ask(messages: list[dict]) -> dict:
        try:
            return assistant.chat(messages)
        except assistant.AssistantError as e:
            raise HTTPException(502, str(e)) from None

    def _on() -> None:
        if not assistant.load_settings()["enabled"]:
            raise HTTPException(409, "the AI assistant is off: turn it on under Hardware")

    @app.get("/api/assistant")
    def assistant_get():
        return assistant.public(assistant.load_settings())

    @app.put("/api/assistant")
    async def assistant_put(request: Request):
        try:
            return assistant.public(assistant.save_settings(await request.json()))
        except ValueError as e:
            raise HTTPException(400, str(e)) from None

    @app.post("/api/assistant/test")
    def assistant_test():
        _on()
        t0 = time.time()
        r = _ask(assistant.TEST_MESSAGES)
        return {**r, "seconds": round(time.time() - t0, 1)}

    def _failure(p: projects.Project) -> dict:
        """The failed stage (the first one, in pipeline order: later failures are older), its
        log excerpt and the quick checks. Empty when nothing failed or a job is running."""
        if jobs.status(p)["running"]:
            return {}
        st = jobs.stages(p, STAGES)
        stage = next((s for s in STAGES if (st.get(s) or {}).get("status") == "failed"), None)
        if stage is None:
            return {}
        error = st[stage].get("error", "")
        log = assistant.failure_logs(p.path("logs"), error)
        job = jobs.status(p)
        if job.get("result") == "stopped" or job.get("stop_requested"):
            quick = [
                {
                    "title": "You stopped this run",
                    "advice": "Press Train to go on from here.",
                    "evidence": "",
                }
            ]
        else:  # wakesmith's own "stopped before it finished" note names every cause: don't match on it
            quick = assistant.quick_causes(error.replace(jobs.STOPPED, "") + "\n" + "\n".join(log))
        ended = ""
        if job.get("finished"):
            ended = {
                "stopped": "stopped by the person (Stop button)",
                "ok": "finished without an error",
            }.get(job.get("result"), "failed: " + jobs._why(job.get("exit")))
        return {"stage": stage, "error": error, "log": log, "quick": quick, "ended": ended}

    @app.get("/api/projects/{name}/failure")
    def failure(name: str):
        """Why the last job failed, by wakesmith's own checks: no model involved."""
        f = _failure(proj(name))
        return {k: v for k, v in f.items() if k != "log"}

    @app.post("/api/projects/{name}/explain")
    def explain(name: str):
        import yaml

        p = proj(name)
        _on()
        f = _failure(p)
        if not f:
            raise HTTPException(409, "no failed stage to explain")
        data = yaml.safe_load(p.path("project.yaml").read_text()) or {}
        messages = assistant.failure_messages(
            assistant.project_summary(data),
            f["stage"],
            f["error"],
            f["log"],
            assistant.hardware_summary(hw()),
            f["quick"],
            ended=f["ended"],
        )
        if not assistant.shows_a_cause(f["error"], f["log"], f["quick"], f["ended"]):
            # nothing for a model to read: asked anyway, small models guess "out of memory"
            return {
                "answer": assistant.NO_CAUSE,
                "model": "",
                "seconds": 0,
                "stage": f["stage"],
                "sent": "(not sent: the log shows no error to explain)",
            }
        return {**_ask(messages), "stage": f["stage"], "sent": messages[1]["content"]}

    def _advice_numbers(p: projects.Project, chosen: dict) -> dict:
        res = {}
        for kind in ("mww", "mww-cv", "oww", "oww-cv", *COMBINED_KINDS):
            f = p.path("results", f"{kind}.json")
            if f.exists():
                r = json.loads(f.read_text())
                res[kind] = {k: r[k] for k in ("table", "positives", "negatives")}
        pick = p.path("mww", "candidates", "pick.json")
        return assistant.advice_numbers(
            p.phrase,
            res,
            chosen,
            json.loads(pick.read_text()) if pick.exists() else None,
            rounds.summary(p),
            runs.listing(p),
        )

    @app.post("/api/projects/{name}/advice")
    async def advice(name: str, request: Request):
        """Advice on the latest round. Body (optional): the cutoffs chosen on the page but not
        exported yet, {"cutoff": mww, "threshold": oww}."""
        p = proj(name)
        _on()
        body = await request.json() if await request.body() else {}
        chosen = {
            "mww": float(body.get("cutoff", p.microwakeword.cutoff)),
            "oww": float(body.get("threshold", p.openwakeword.threshold)),
        }
        numbers = await asyncio.to_thread(_advice_numbers, p, chosen)
        if not numbers["models"]:
            raise HTTPException(409, "no held-out results yet: the advisor needs test numbers")
        answer = await asyncio.to_thread(_ask, assistant.advice_messages(numbers))
        return {**answer, "numbers": numbers}

    @app.get("/api/projects/{name}/heldout")
    def heldout(name: str):
        p = proj(name)
        out = {}
        for kind in ("mww", "oww"):
            f = p.path("results", f"{kind}.json")
            if f.exists():
                out[kind] = evaluate.format_table(json.loads(f.read_text()))
        return out

    # -- record a room (sessions.py): one long recording, split into takes for every wake word ----
    def sess(sid: str) -> str:
        try:
            sessions.folder(sid)
        except ValueError as e:
            raise HTTPException(400, str(e)) from None
        except FileNotFoundError as e:
            raise HTTPException(404, str(e)) from None
        return sid

    def sess_view(sid: str) -> dict:
        segs = sessions.segments(sid)
        marks = sessions.sent_marks(sid)
        return {
            "session": sessions.load(sid),
            "segments": [{**x, "sent": marks.get(x["i"], [])} for x in segs],
            "split": sessions.split_status(sid),
            "audio": (sessions.folder(sid) / "audio.wav").exists(),
        }

    @app.get("/api/sessions")
    def sess_list():
        return {
            "sessions": sessions.listing(),
            "rooms": sessions.rooms(),
            "projects": sessions.choices(),
            "spots": [n for n, _ in sessions.SPOTS],
            "house_rirs": len(sessions.house_rirs()),
        }

    @app.post("/api/sessions")
    async def sess_create(request: Request):
        b = await request.json()
        try:
            return sessions.create(
                str(b.get("room", "")),
                [str(x) for x in b.get("speakers") or []],
                [str(x) for x in b.get("projects") or []],
                [str(x) for x in b["spots"]] if b.get("spots") else None,
                str(b.get("device", "")),
            )
        except FileNotFoundError as e:
            raise HTTPException(404, f"no such wake word: {e}") from None
        except ValueError as e:
            raise HTTPException(400, str(e)) from None

    @app.get("/api/sessions/{sid}")
    def sess_get(sid: str):
        return sess_view(sess(sid))

    @app.delete("/api/sessions/{sid}")
    def sess_delete(sid: str):
        sessions.delete(sess(sid))
        return {"deleted": sid}

    @app.post("/api/sessions/{sid}/chunk")
    async def sess_chunk(sid: str, request: Request, seq: int = 0):
        sess(sid)
        try:
            return sessions.add_chunk(sid, seq, await request.body())
        except PermissionError as e:
            raise HTTPException(409, str(e)) from None
        except (ValueError, RuntimeError) as e:  # soundfile raises RuntimeError on bad WAV bytes
            raise HTTPException(400, str(e)) from None

    @app.post("/api/sessions/{sid}/stop")
    def sess_stop(sid: str, split: bool = True):
        sess(sid)
        s = sessions.finish(sid)
        if split and s["seconds"] > 0:
            try:
                sessions.start_split(sid)
            except RuntimeError:
                pass  # already being split
        return sess_view(sid)

    @app.post("/api/sessions/{sid}/split")
    def sess_split(sid: str):
        sess(sid)
        try:
            sessions.start_split(sid)
        except RuntimeError as e:
            raise HTTPException(409, str(e)) from None
        return sess_view(sid)

    @app.get("/api/sessions/{sid}/log")
    def sess_log(sid: str):
        f = sessions.folder(sess(sid)) / "split.log"
        text = f.read_text(errors="replace")[-4000:] if f.exists() else ""
        return Response(text, media_type="text/plain; charset=utf-8")

    @app.put("/api/sessions/{sid}/segments")
    async def sess_labels(sid: str, request: Request):
        sess(sid)
        try:
            sessions.set_labels(sid, (await request.json()).get("changes") or {})
        except ValueError as e:
            raise HTTPException(400, str(e)) from None
        return sess_view(sid)

    @app.get("/api/sessions/{sid}/audio")
    def sess_audio(sid: str, start: float = 0.0, end: float = 0.0):
        """The whole recording, or the stretch start..end seconds."""
        f = sessions.folder(sess(sid)) / "audio.wav"
        if not f.exists():
            raise HTTPException(404, "not finished yet")
        if not end:
            return FileResponse(f, media_type="audio/wav")
        import io

        import soundfile as sf

        with sf.SoundFile(f) as w:
            w.seek(max(0, int(start * w.samplerate)))
            a = w.read(max(0, int((end - start) * w.samplerate)), dtype="int16")
        b = io.BytesIO()
        sf.write(b, a, sessions.SR, format="WAV", subtype="PCM_16")
        return Response(b.getvalue(), media_type="audio/wav")

    @app.post("/api/sessions/{sid}/send")
    def sess_send(sid: str):
        sess(sid)
        if sessions.split_status(sid).get("status") == "running":
            raise HTTPException(409, "still splitting: send when it has finished")
        if not sessions.segments(sid):
            raise HTTPException(409, "nothing found yet: split the recording first")
        return {**sessions.send(sid), **sess_view(sid)}

    return app
