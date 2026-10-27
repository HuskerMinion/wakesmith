"""End-to-end smoke test: a tiny project through every stage, on the CPU and on the GPU.

Slow (a few minutes each) and needs the downloaded data (`wakesmith setup`), so it only runs when asked:
    WAKESMITH_SMOKE=1 pytest tests/test_smoke.py
Catches what the unit tests can't: a stage that crashes on real data (an open-file limit, a GPU kernel
error, a broken export) before a real training run finds it.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

from wakesmith import hardware
from wakesmith import project as projects
from wakesmith.project import Samples

pytestmark = pytest.mark.skipif(
    os.environ.get("WAKESMITH_SMOKE") != "1", reason="slow end-to-end run: set WAKESMITH_SMOKE=1"
)


def _gpu() -> bool:
    h, _problem = hardware.detect_safe()  # detection problems: the CPU run still tells us something
    return bool(h.gpus)


@pytest.mark.parametrize("where", ["cpu", "gpu"])
def test_tiny_project_through_every_stage(tmp_path, where):
    if where == "gpu" and not _gpu():
        pytest.skip("no GPU")
    p = projects.create(
        "okay smoke",
        negatives=["okay spoke", "smoke", "oak smoke", "okay"],
        where=tmp_path / "okay_smoke",
        samples=Samples(positive=160, negative=160, positive_test=40, negative_test=40),
    )
    p.microwakeword.steps = 600  # past one validation (every 500 steps)
    p.openwakeword.steps = 300
    p.save()
    cmd = [sys.executable, "-m", "wakesmith", "train", str(p.dir)]
    if where == "cpu":
        cmd.append("--cpu")
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=3600, check=False)
    assert r.returncode == 0, (r.stdout + r.stderr)[-4000:]
    state = json.loads((p.dir / "state.json").read_text())
    for stage in (
        "generate",
        "mww-features",
        "mww-train",
        "oww-features",
        "oww-train",
        "test",
        "export",
    ):
        assert state.get(stage, {}).get("status") == "done", (stage, state.get(stage))
    exported = sorted(
        f.relative_to(p.dir / "export").as_posix() for f in (p.dir / "export").rglob("*")
    )
    assert (
        "microwakeword/okay_smoke.tflite" in exported
        and "microwakeword/okay_smoke.json" in exported
    )
    assert "openwakeword/okay_smoke.tflite" in exported and "LICENSE-NOTE.txt" in exported
    manifest = json.loads((p.dir / "export/microwakeword/okay_smoke.json").read_text())
    assert manifest["model"] == "okay_smoke.tflite" and manifest["micro"]["feature_step_size"] == 10
