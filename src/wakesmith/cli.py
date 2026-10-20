"""wakesmith command line. Everything the web UI does is available here too."""

from __future__ import annotations

from pathlib import Path

import typer

app = typer.Typer(
    no_args_is_help=True,
    add_completion=False,
    help="Forge your own wake word: microWakeWord + openWakeWord models.",
)


@app.command()
def doctor() -> None:
    """Show detected hardware, the settings that follow from it, and downloaded data."""
    from . import assets, hardware

    hw = hardware.detect()
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
) -> None:
    """Run the pipeline (stages: generate real mww-features mww-train oww-features oww-train
    test export). Finished work is reused: generation resumes, `--from real` retrains after your
    recordings or labels change."""
    from . import pipeline, project

    pipeline.run(project.load(project_), first, last)


@app.command()
def test(
    project_: str = typer.Argument(..., metavar="PROJECT"),
    kind: str = typer.Option("all", help="mww, oww or all"),
    model: Path = typer.Option(None, help="score this model file instead"),
) -> None:
    """Score held-out real recordings."""
    from . import evaluate, project

    p = project.load(project_)
    for k in ("mww", "oww") if kind == "all" else (kind,):
        typer.echo(evaluate.format_table(evaluate.score(p, k, model)))


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
