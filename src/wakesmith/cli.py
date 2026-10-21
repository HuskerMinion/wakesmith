"""wakesmith command line. Everything the web UI does is available here too."""

from __future__ import annotations

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
    from . import assets, hardware

    _cpu_only(cpu)
    hw, problem = hardware.detect_safe()
    if problem:
        typer.echo(f"warning: {problem}")
    typer.echo(f"accelerator: {hw.accel}")
    for g in hw.gpus:
        typer.echo(f"  {g.device}: {g.name}, {g.memory_gb} GB")
    typer.echo(
        f"CPU cores: {hw.cpus}   RAM: {hw.ram_free_gb:.0f} of {hw.ram_gb:.0f} GB free   "
        f"disk free: {hardware.disk_free_gb(str(assets.home().parent))} GB"
    )
    typer.echo(
        f"settings: TTS on {', '.join(hw.tts_devices())} (batch {hw.tts_batch()}), "
        f"{hw.workers()} CPU workers"
    )
    if low := hardware.disk_warning(str(assets.home().parent)):
        typer.echo(f"warning: {low}")
    if low := hardware.ram_warning(hw):
        typer.echo(f"warning: {low}")
    typer.echo(f"data in {assets.home()}:")
    for a in assets.ASSETS.values():
        mark = "ok " if a.present() else "-- "
        typer.echo(f"  {mark}{a.name:28s} {a.size / 1e9:6.2f} GB  {a.license}")


@app.command()
def suggest(phrase: str, n: int = 40) -> None:
    """Suggest sound-alike phrases to reject for PHRASE."""
    from . import phrases

    for s in phrases.suggest(phrase, n):
        typer.echo(s)


@app.command()
def new(
    phrase: str,
    name: str = typer.Option(None, help="project name (default: from the phrase)"),
    positives: Path = typer.Option(None, help="folder of real recordings of the phrase"),
    negatives: Path = typer.Option(None, help="folder of real not-the-phrase recordings"),
    sound_alikes: int = typer.Option(50, help="sound-alike phrases to suggest"),
) -> None:
    """Create a project for PHRASE (edit its project.yaml afterwards if you like)."""
    from . import phrases, project

    p = project.create(phrase, name, negatives=phrases.suggest(phrase, sound_alikes))
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
        typer.echo(evaluate.format_table(evaluate.score(p, k, model, frontend)))


@app.command()
def export(
    project_: str = typer.Argument(..., metavar="PROJECT"),
    cutoff: float = typer.Option(None, help="microWakeWord probability cutoff"),
    threshold: float = typer.Option(None, help="openWakeWord threshold"),
) -> None:
    """Write device-ready files to <project>/export."""
    from . import export as ex
    from . import project

    p = project.load(project_)
    if cutoff is not None:
        p.microwakeword.cutoff = cutoff
    if threshold is not None:
        p.openwakeword.threshold = threshold
    p.save()
    for f in ex.export(p):
        typer.echo(f)


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
    tag = {"xpu": "+xpu", "cuda": "+cu", "cpu": "+cpu"}[accel]
    if tag in current and se.torch_sees(accel):
        typer.echo(f"torch {current} already fits")
    else:
        se.install_torch(accel)
        if not se.torch_sees(accel):
            typer.echo(f"torch {accel} build doesn't see the GPU; falling back to CPU")
            accel = "cpu"
            se.install_torch("cpu")
    typer.echo(f"microWakeWord: {se.install_microwakeword()}")
    if no_data:
        return
    todo = [a for a in assets.ASSETS.values() if not a.present()]
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
        f"{len(out['clips'])} disputed clips at {out['threshold']}: {pos} labelled as the "
        f"phrase scored low, {len(out['clips']) - pos} others scored high"
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
