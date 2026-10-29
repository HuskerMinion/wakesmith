"""wakesmith command line. Everything the web UI does is available here too."""

from __future__ import annotations

import json
import os
from pathlib import Path

import typer

app = typer.Typer(
    no_args_is_help=True,
    add_completion=False,
    help="Forge your own wake word: microWakeWord + openWakeWord models.",
)


CPU_HELP = "use only the CPU, even if a GPU is found"


def _cpu_only(cpu: bool) -> None:
    if cpu:
        os.environ["WAKESMITH_CPU"] = "1"  # inherited by worker processes


@app.command()
def doctor(cpu: bool = typer.Option(False, "--cpu", help=CPU_HELP)) -> None:
    """Show detected hardware, the settings that follow from it, and downloaded data."""
    from . import assets, hardware, mww

    _cpu_only(cpu)
    hw, problem = hardware.detect_safe()
    if problem:
        typer.echo(f"warning: {problem}")
    typer.echo(f"accelerator: {hw.accel}")
    for g in hw.gpus:
        typer.echo(f"  {g.device}: {g.name}, {g.memory_gb} GB")
    if hw.accel == "xpu" and (bad := hardware.xpu_selftest()):
        typer.echo(f"warning: {bad}")
    typer.echo(
        f"CPU cores: {hw.cpus}   RAM: {hw.ram_free_gb:.0f} of {hw.ram_gb:.0f} GB free   "
        f"disk free: {hardware.disk_free_gb(str(assets.home().parent))} GB"
    )
    typer.echo(
        f"settings: TTS on {', '.join(hw.tts_devices())} (batch {hw.tts_batch()}), "
        f"microWakeWord training on {mww.train_device(hw.gpus)}, {hw.workers()} CPU workers"
    )
    if low := hardware.disk_warning(str(assets.home().parent)):
        typer.echo(f"warning: {low}")
    if low := hardware.ram_warning(hw):
        typer.echo(f"warning: {low}")
    typer.echo(f"data in {assets.home()}:")
    for a in assets.ASSETS.values():
        if a.optional and not a.present():
            continue  # an experimental language's voices: listed once a project fetched them
        mark = "ok " if a.present() else "-- "
        typer.echo(f"  {mark}{a.name:28s} {a.size / 1e9:6.2f} GB  {a.license}")


def _offered(language: str) -> None:
    """Refuse a language new wake words can't be in (experimental ones only when turned on)."""
    from . import languages

    if language not in [lang.code for lang in languages.available()]:
        raise typer.BadParameter(
            f"{language!r} is not on: languages other than English are experimental; turn them on with "
            "WAKESMITH_EXPERIMENTAL_LANGUAGES=1 or on the web page's Data page",
            param_hint="--language",
        )


@app.command()
def suggest(
    phrase: str, n: int = 40, language: str = typer.Option("en", help="en, or de (experimental)")
) -> None:
    """Suggest sound-alike phrases to reject for PHRASE."""
    from . import phrases

    _offered(language)
    for s in phrases.suggest(phrase, n, language=language):
        typer.echo(s)


@app.command()
def new(
    phrase: str,
    name: str = typer.Option(None, help="project name (default: from the phrase)"),
    positives: Path = typer.Option(None, help="folder of real recordings of the phrase"),
    negatives: Path = typer.Option(None, help="folder of real not-the-phrase recordings"),
    sound_alikes: int = typer.Option(50, help="sound-alike phrases to suggest"),
    language: str = typer.Option(
        "en", help="en, or de (experimental: turn on with WAKESMITH_EXPERIMENTAL_LANGUAGES=1)"
    ),
) -> None:
    """Create a project for PHRASE (edit its project.yaml afterwards if you like)."""
    from . import phrases, project

    _offered(language)
    p = project.create(
        phrase,
        name,
        negatives=phrases.suggest(phrase, sound_alikes, language=language),
        language=language,
    )
    if positives:
        p.real.positives = str(positives.expanduser().resolve())
    if negatives:
        p.real.negatives = str(negatives.expanduser().resolve())
    p.save()
    typer.echo(f"created {p.dir}/project.yaml with {len(p.negatives)} sound-alike phrases")


@app.command()
def train(
    project_: str = typer.Argument(..., metavar="PROJECT"),
    first: str = typer.Option("generate", "--from", help="first stage"),
    last: str = typer.Option("export", "--to", help="last stage"),
    cpu: bool = typer.Option(False, "--cpu", help=CPU_HELP),
) -> None:
    """Run the pipeline (stages: generate real mww-features mww-train oww-features oww-train
    test export). Finished work is reused: generation resumes, `--from real` retrains after your
    recordings or labels change."""
    from . import pipeline, project

    _cpu_only(cpu)
    pipeline.run(project.load(project_), first, last)


@app.command()
def version(
    project_: str = typer.Argument(..., metavar="PROJECT"),
    note: str = typer.Option("", help="what this version tries (shown on the page)"),
    train_it: bool = typer.Option(True, "--train/--no-train", help="prepare and train it now"),
    cpu: bool = typer.Option(False, "--cpu", help=CPU_HELP),
) -> None:
    """Make the next version of PROJECT's wake word: same recipe and recordings, its device clips plus
    the ones that came since (clear Whisper suggestions taken as labels), the newest day held out for
    the test, then trained from the recordings on and compared with the version on your devices."""
    from . import project, versions

    q = versions.create(project.load(project_), note)
    typer.echo(f"{q.title}: {q.dir}")
    if train_it:
        prepare_version(str(q.dir), cpu)


@app.command("prepare-version", hidden=True)
def prepare_version(
    project_: str = typer.Argument(..., metavar="PROJECT"), cpu: bool = False
) -> None:
    """Sync and label device clips, hold out the newest day, train from the recordings (the page runs
    this as a job after making a version)."""
    from . import pipeline, project, versions

    _cpu_only(cpu)
    q = project.load(project_)
    versions.prepare(q, log=typer.echo)  # a failed sync still trains, on the clips it already has
    pipeline.run(q, "real", "export")


@app.command()
def test(
    project_: str = typer.Argument(..., metavar="PROJECT"),
    kind: str = typer.Option("all", help="mww, oww or all"),
    model: Path = typer.Option(None, help="score this model file instead"),
    frontend: str = typer.Option(
        None, help="microWakeWord features: tflm (ESPHome) or zserge (default: the project's)"
    ),
) -> None:
    """Score held-out real recordings."""
    from . import evaluate, project

    if frontend not in (None, "tflm", "zserge"):
        raise typer.BadParameter("tflm or zserge", param_hint="--frontend")
    p = project.load(project_)
    for k in ("mww", "oww") if kind == "all" else (kind,):
        r = evaluate.score(p, k, model, frontend, report_half=model is None and frontend is None)
        typer.echo(evaluate.format_table(r))


@app.command()
def export(
    project_: str = typer.Argument(..., metavar="PROJECT"),
    cutoff: float = typer.Option(None, help="microWakeWord probability cutoff"),
    threshold: float = typer.Option(None, help="openWakeWord threshold"),
    combined: bool = typer.Option(
        None,
        "--combined/--single",
        help="microWakeWord: the combined model (`wakesmith combine`) or the kept candidate "
        "(default: the project's microwakeword.combine setting)",
    ),
) -> None:
    """Write device-ready files to <project>/export."""
    from . import export as ex
    from . import project

    p = project.load(project_)
    if cutoff is not None:
        p.microwakeword.cutoff = cutoff
    if threshold is not None:
        p.openwakeword.threshold = threshold
    try:
        both = ex.use_combined(p, combined)
    except ValueError as e:
        raise typer.BadParameter(str(e), param_hint="--combined") from None
    p.save()
    for f in ex.export(p, combined=bool(both)):
        typer.echo(f)


@app.command()
def combine(
    project_: str = typer.Argument(..., metavar="PROJECT"),
    candidates: str = typer.Option(
        "", help="candidate numbers, e.g. 1,2,3 (default: every one that finished)"
    ),
    test: bool = typer.Option(True, help="score it on the held-out recordings afterwards"),
    previous: bool = typer.Option(
        False, "--previous", help="from the previous round's candidates (mww/candidates.previous)"
    ),
) -> None:
    """Average the best-of-N candidates into one microWakeWord model (mww/combined): fewer false
    triggers, N times the size. Works on any project trained with microwakeword.candidates > 1."""
    from . import ensemble, evaluate, project

    p = project.load(project_)
    try:
        nums = [int(x) for x in candidates.replace("s", "").split(",") if x.strip()]
    except ValueError:
        raise typer.BadParameter("numbers like 1,2,3", param_hint="--candidates") from None
    try:
        typer.echo(ensemble.combine(p, nums or None, log=typer.echo, previous=previous))
    except (ValueError, RuntimeError) as e:
        raise typer.Exit(f"error: {e}") from None
    if test:
        evaluate.score_combined(p, log=typer.echo)


@app.command()
def download(names: list[str] = typer.Argument(None, help="asset names (default: all)")) -> None:
    """Download training data and models (see `wakesmith doctor` for names and licenses)."""
    from . import assets

    for n in names or list(assets.ASSETS):
        typer.echo(f"{n}: {assets.fetch(assets.ASSETS[n])}")


@app.command()
def setup(
    cpu: bool = typer.Option(False, "--cpu", help="CPU only, even if a GPU is found"),
    yes: bool = typer.Option(False, "--yes", "-y", help="download data without asking"),
    no_data: bool = typer.Option(False, "--no-data", help="skip the data downloads"),
) -> None:
    """Install the right torch build for this machine, microWakeWord, and the training data."""
    from . import assets
    from . import setup_env as se

    missing = se.missing_tools()
    if missing:
        typer.echo(f"missing: {', '.join(missing)} (e.g. sudo apt install {' '.join(missing)})")
        raise typer.Exit(1)
    gpus = se.find_gpus()
    accel = "cpu" if cpu else se.choose_accel(gpus)
    typer.echo(f"GPUs: {', '.join(f'{v} {a}' for v, a in gpus) or 'none'} -> torch build: {accel}")
    try:
        import torch

        current = torch.__version__
    except ImportError:
        current = ""
    tag = {"xpu": "+xpu", "cuda": "+cu", "rocm": "+rocm", "cpu": "+cpu"}[accel]
    if accel == "rocm":
        typer.echo("AMD GPU support (ROCm) is experimental and untested: reports welcome")
    if tag in current and se.torch_sees(accel):
        typer.echo(f"torch {current} already fits")
    else:
        se.install_torch(accel)
        if not se.torch_sees(accel):
            typer.echo(f"torch {accel} build can't run on the GPU; falling back to CPU")
            accel = "cpu"
            se.install_torch("cpu")
    typer.echo(f"microWakeWord: {se.install_microwakeword()}")
    if no_data:
        return
    todo = [a for a in assets.required() if not a.present()]
    gb = sum(a.size for a in todo) / 1e9
    if todo and (yes or typer.confirm(f"download {len(todo)} items, {gb:.1f} GB?")):
        for a in todo:
            typer.echo(f"{a.name}: {assets.fetch(a)}")
    typer.echo("done: run `wakesmith doctor` to see what wakesmith will use")


@app.command()
def review(
    project_: str = typer.Argument(..., metavar="PROJECT"),
    model: str = typer.Option("mww", help="score with the mww or oww model"),
    threshold: float = typer.Option(None, help="disagreement threshold (default: export cutoff)"),
) -> None:
    """List real recordings the trained model disagrees with, for the review page."""
    from . import project
    from . import review as rv

    out = rv.build(project.load(project_), model, threshold)
    pos = sum(c["label"] == "wake" for c in out["clips"])
    typer.echo(
        f"{len(out['clips'])} disputed clips at {out['threshold']}: {pos} labeled as the "
        f"phrase scored low, {len(out['clips']) - pos} others scored high"
    )


clips_app = typer.Typer(
    help="Device clips: import, transcribe, label and hold out (the next round)."
)
app.add_typer(clips_app, name="clips")


@clips_app.command("import")
def clips_import(
    project_: str = typer.Argument(..., metavar="PROJECT"), folder: str = typer.Argument(...)
) -> None:
    """Copy the clips in FOLDER (sidecar .json with at/model/score/device read when present)."""
    from . import project, rounds

    r = rounds.import_folder(project.load(project_), folder)
    typer.echo(f"{r['added']} clips added ({r['known']} already there)")


@clips_app.command("transcribe")
def clips_transcribe(
    project_: str = typer.Argument(..., metavar="PROJECT"),
    device: str = typer.Option(
        "", help="torch device for Whisper (default: the first GPU, else cpu)"
    ),
) -> None:
    """Whisper on every clip not transcribed yet, with a label suggestion."""
    from . import hardware, mww, project, rounds

    dev = device or mww.train_device(hardware.detect_safe()[0].gpus)
    r = rounds.transcribe(project.load(project_), dev, progress=lambda i, n: typer.echo(f"{i}/{n}"))
    typer.echo(f"{r['transcribed']} transcribed on {dev}")


@clips_app.command("accept")
def clips_accept(project_: str = typer.Argument(..., metavar="PROJECT")) -> None:
    """Label the clear false alarms (Whisper heard none of the phrase's words) "not the phrase"; clips where
    it heard the phrase wait for a person, since the TV saying it isn't a call."""
    from . import project, rounds

    typer.echo(f"{rounds.accept_suggestions(project.load(project_))} labels set")


@clips_app.command("status")
def clips_status(
    project_: str = typer.Argument(..., metavar="PROJECT"),
    holdout_from: str = typer.Option("", help="ISO time: count what would be held out"),
) -> None:
    """Counts: clips, transcribed, labels, unclear."""
    from . import project, rounds

    typer.echo(json.dumps(rounds.summary(project.load(project_), holdout_from or None), indent=1))


@clips_app.command("apply")
def clips_apply(
    project_: str = typer.Argument(..., metavar="PROJECT"),
    holdout_from: str = typer.Option(..., help="ISO time; labeled clips from then on are held out"),
) -> None:
    """Write the labeled clips into the project's training data (then train --from real)."""
    from . import project, rounds

    typer.echo(json.dumps(rounds.apply(project.load(project_), holdout_from), indent=1))


DEVICE_HELP = (
    "Whisper's device: cpu, xpu:N, cuda:N, or auto (the GPU when no other wakesmith job runs, "
    "else cpu)"
)


@clips_app.command("sync")
def clips_sync(
    project_: str = typer.Argument(..., metavar="PROJECT"),
    source: str = typer.Option("", "--from", help="puller folder (default: devices.source)"),
    model_prefix: str = typer.Option(
        None, help="only clips whose sidecar model starts with this (default: devices.model_prefix)"
    ),
    transcribe_device: str = typer.Option("auto", help=DEVICE_HELP),
    since: str = typer.Option(
        None, help="ISO time: only clips from then on (default: devices.since)"
    ),
    limit: int = typer.Option(0, help="at most this many new clips per run, oldest first (0: all)"),
    save: bool = typer.Option(
        False, "--save", help="keep --from/--model-prefix/--since as the project's settings"
    ),
) -> None:
    """Bring in the word's new device clips and transcribe them: afterwards only the unclear ones
    need a person."""
    from . import clipsync, project

    p = project.load(project_)
    if save:
        p.devices.source = source or p.devices.source
        p.devices.model_prefix = p.devices.model_prefix if model_prefix is None else model_prefix
        p.devices.since = p.devices.since if since is None else since
        p.save()
    try:
        r = clipsync.sync(
            p,
            source,
            model_prefix,
            transcribe_device,
            since,
            limit,
            progress=lambda i, n: typer.echo(f"transcribed {i}/{n}"),
        )
    except clipsync.Busy as e:
        clipsync.log(f"{p.name}: skipped: {e}")
        raise typer.Exit(f"skipped: {e}") from None
    except ValueError as e:
        raise typer.Exit(f"error: {e}") from None
    except Exception as e:
        clipsync.failed(p, f"{type(e).__name__}: {e}")
        raise
    typer.echo(f"{p.name}: {clipsync.describe(r)}")


@clips_app.command("sync-all")
def clips_sync_all(
    transcribe_device: str = typer.Option("auto", help=DEVICE_HELP),
    limit: int = typer.Option(0, help="at most this many new clips per wake word (0: all)"),
) -> None:
    """`clips sync` for every wake word with a source folder (devices.source)."""
    from . import clipsync

    rs = clipsync.sync_all(transcribe_device, limit)
    for r in rs:
        typer.echo(f"{r['project']}: {clipsync.describe(r)}")
    if not rs:
        typer.echo("no wake word has a source folder for device clips (devices.source)")
    if any(r.get("result") == "failed" for r in rs):
        raise typer.Exit(1)


@clips_app.command("schedule")
def clips_schedule(
    at: str = typer.Option("03:30", help="time of day (24 h, local)"),
    off: bool = typer.Option(False, "--off", help="turn the daily sync off and remove it"),
    show: bool = typer.Option(False, "--status", help="show the next run and the last result"),
) -> None:
    """Run `clips sync-all` every day (a systemd user timer; log in <projects>/clips-sync.log)."""
    import subprocess

    from . import clipsync

    try:
        if show:
            s = clipsync.status()
        elif off:
            s = clipsync.remove()
            typer.echo("daily device-clip sync: off")
        else:
            s = clipsync.install(at)
            typer.echo(f"daily device-clip sync: on, at {s['at']}")
    except (ValueError, RuntimeError, OSError, subprocess.SubprocessError) as e:
        raise typer.Exit(f"error: {e}") from None
    if not s["installed"]:
        if show:
            typer.echo("daily device-clip sync: off")
        return
    typer.echo(f"enabled: {'yes' if s['enabled'] else 'no'}   next run: {s['next'] or '-'}")
    typer.echo(f"last run: {s.get('last_run') or '-'}   result: {s.get('last_result') or '-'}")
    for line in s["log"][-5:] if show else []:
        typer.echo(f"  {line}")


def _gb(n: float) -> str:
    return f"{n / 1e9:.1f} GB" if n >= 1e9 else f"{n / 1e6:.0f} MB"


@app.command("archive")
def archive_cmd(
    project_: str = typer.Argument(..., metavar="PROJECT"),
    dry_run: bool = typer.Option(False, "--dry-run", help="only show what would be removed"),
    clips: bool = typer.Option(
        False, "--clips", help="also remove the project's own synthetic clips (unless shared)"
    ),
) -> None:
    """Free disk space: remove what training rebuilds by itself (features, prepared recordings,
    spare candidate checkpoints). Settings, recordings, results, exports and the model stay; the next
    Train rebuilds the features first."""
    from . import archive, project

    p = project.load(project_)
    pl = archive.plan(p, clips)
    for r in pl["remove"]:
        typer.echo(f"{_gb(r['bytes']):>9}  {r['path']}  ({r['why']})")
    for k in pl["kept"]:
        typer.echo(f"     kept  {k}")
    typer.echo(
        f"{_gb(pl['bytes']):>9}  in total; stages to rerun: {', '.join(pl['stages']) or 'none'}"
    )
    if dry_run:
        return
    try:
        r = archive.archive(p, clips)
    except archive.Busy as e:
        raise typer.Exit(str(e)) from None
    typer.echo(f"archived {p.name}: freed {_gb(r['freed_bytes'])}")


@app.command("storage")
def storage_cmd() -> None:
    """Disk use per project: size, what archiving would free, last activity, models in use."""
    from datetime import datetime

    from . import archive

    days = archive.settings()["archive_after_days"]
    rows = archive.storage()
    head = f"{'project':24} {'size':>8} {'freeable':>9} {'last activity':16} {'idle':>5} {'in use':14} archived"
    typer.echo(head)
    for r in rows:
        use = ", ".join(
            x
            for x in (
                f"{r['live']} live" if r["live"] else "",
                f"{r['offered']} offered" if r["offered"] else "",
            )
            if x
        )
        when = (
            datetime.fromtimestamp(r["last_activity"]).strftime("%Y-%m-%d %H:%M")
            if r["last_activity"]
            else "-"
        )
        arch = (
            datetime.fromtimestamp(r["archived_at"]).strftime("%Y-%m-%d") if r["archived"] else ""
        )
        typer.echo(
            f"{r['name'][:24]:24} {_gb(r['bytes']):>8} {_gb(r['freeable']):>9} {when:16} "
            f"{r['idle_days']:>4.0f}d {use or '-':14} {arch}"
        )
    total = sum(r["bytes"] for r in rows)
    free = sum(r["freeable"] for r in rows)
    typer.echo(f"{'total':24} {_gb(total):>8} {_gb(free):>9}")
    typer.echo(
        f"auto archive: after {days} idle days (projects with an offered or live model are left alone)"
        if days
        else "auto archive: off"
    )


@app.command("commonvoice-count")
def commonvoice_count(phrase: str) -> None:
    """How many Common Voice takes and speakers say PHRASE (needs the dataset)."""
    from . import commonvoice

    ds = commonvoice.dataset()
    if not commonvoice.present(ds):
        raise typer.Exit(f"no Common Voice at {ds}: run `wakesmith commonvoice-download` first")
    c = commonvoice.count(ds, phrase)
    typer.echo(
        f"{phrase!r}: {c['takes']} takes from {c['speakers']} speakers in Common Voice 17.0 English"
    )


@app.command("commonvoice-download")
def commonvoice_download() -> None:
    """Download Mozilla Common Voice 17.0 English (CC0, about 47 GB) for real-voice tests."""
    from . import commonvoice

    typer.echo(f"done: {commonvoice.download(typer.echo)}")


def _ui_token(secrets) -> str:
    """The LAN access token, kept between restarts so the printed link stays the same
    (delete the file to get a new one)."""
    from . import assets

    f = assets.home() / "ui-token"
    if f.exists() and len(t := f.read_text().strip()) >= 16:
        return t
    f.parent.mkdir(parents=True, exist_ok=True)
    t = secrets.token_urlsafe(16)
    f.write_text(t + "\n")
    f.chmod(0o600)
    return t


@app.command()
def serve(
    host: str = typer.Option("127.0.0.1", help="0.0.0.0 to use it from other devices on your LAN"),
    port: int = typer.Option(8765),
    cpu: bool = typer.Option(False, "--cpu", help=CPU_HELP),
    https: bool = typer.Option(
        False,
        "--https",
        help="serve over HTTPS (self-signed): browsers only allow the microphone "
        "on localhost or HTTPS, so use this to record takes from another device",
    ),
) -> None:
    """Start the web UI."""
    import secrets

    _cpu_only(cpu)
    import uvicorn

    from .web.server import create_app

    local = host in ("127.0.0.1", "localhost", "::1")
    token = None if local else _ui_token(secrets)
    shown = "localhost" if local else host
    tls = {}
    if https:
        from . import tls as tlsmod

        cert, key = tlsmod.certificate()
        tls = {"ssl_certfile": str(cert), "ssl_keyfile": str(key)}
    url = f"{'https' if https else 'http'}://{shown}:{port}/" + ("" if local else f"?token={token}")
    typer.echo(f"wakesmith UI: {url}")
    if token:
        typer.echo("anyone with this link (token included) can use the UI: keep it on your LAN")
    if https:
        typer.echo("self-signed certificate: the browser warns once; accept it to continue")
    elif not local:
        typer.echo("recording takes in the browser needs HTTPS from other devices: add --https")
    uvicorn.run(create_app(token), host=host, port=port, log_level="warning", **tls)


room_app = typer.Typer(
    help="Whole-room recordings: split them into takes, send them to the wake words."
)
app.add_typer(room_app, name="room")


@room_app.command("list")
def room_list() -> None:
    """Recorded room sessions, newest first."""
    from . import sessions

    for s in sessions.listing():
        typer.echo(
            f"{s['id']}  {s['room']}  {s['seconds']:.0f} s  {s['segments']} segments  {s['split'] or ''}"
        )


@room_app.command("split")
def room_split(
    session: str = typer.Argument(..., help="session id (wakesmith room list)"),
    device: str = typer.Option(
        "auto", help="torch device for Whisper; auto: the GPU unless a training job runs, else cpu"
    ),
    model: str = typer.Option("turbo", help="Whisper model"),
) -> None:
    """Find the takes, claps and room sound in a session and label them (replaces earlier labels)."""
    from . import sessions

    typer.echo(json.dumps(sessions.run_split(session, device, model), indent=1))


@room_app.command("send")
def room_send(session: str = typer.Argument(..., help="session id")) -> None:
    """Send the labeled segments to the wake words' recordings (nothing is sent twice)."""
    from . import sessions

    typer.echo(json.dumps(sessions.send(session), indent=1))
