"""`wakesmith setup`: install what pip can't choose for you, then fetch the data.

1. Find GPUs from PCI ids (/sys/bus/pci), before torch is installed: Intel -> XPU wheels,
   NVIDIA -> CUDA wheels, otherwise CPU wheels.
2. Install that torch build into this Python environment (uv when available, else pip).
3. Install microWakeWord from a pinned clone, editable (its wheel omits microwakeword.audio).
4. Download the training data (asks first; ~24 GB).
5. Check torch really sees the GPU; fall back to CPU wheels if not.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

from . import assets

MWW_REPO = "https://github.com/kahrendt/microWakeWord.git"
MWW_COMMIT = "4665173cd35f1cff9a61e06fc427f124766c488e"
TORCH_INDEX = {
    "xpu": "https://download.pytorch.org/whl/xpu",
    "cuda": "https://download.pytorch.org/whl/cu128",
    "cpu": "https://download.pytorch.org/whl/cpu",
}
VENDORS = {"0x8086": "intel", "0x10de": "nvidia", "0x1002": "amd"}


def pci_gpus() -> list[tuple[str, str]]:
    """(vendor, pci address) of display controllers, from sysfs (no lspci needed)."""
    out = []
    for dev in sorted(Path("/sys/bus/pci/devices").glob("*")):
        try:
            if not (dev / "class").read_text().startswith("0x03"):
                continue
            vendor = VENDORS.get((dev / "vendor").read_text().strip(), "other")
        except OSError:
            continue
        out.append((vendor, dev.name))
    return out


def choose_accel(gpus: list[tuple[str, str]]) -> str:
    vendors = {v for v, _ in gpus}
    if "nvidia" in vendors:
        return "cuda"
    if "intel" in vendors:
        return "xpu"  # verified after install: integrated GPUs may not be supported
    return "cpu"


def _pip(*args: str) -> None:
    uv = shutil.which("uv")
    cmd = (
        [uv, "pip", "install", "--python", sys.executable, *args]
        if uv
        else [sys.executable, "-m", "pip", "install", *args]
    )
    subprocess.run(cmd, check=True)


def install_torch(accel: str) -> None:
    _pip(
        "--reinstall-package",
        "torch",
        "--reinstall-package",
        "torchaudio",
        "torch",
        "torchaudio",
        "--index-url",
        TORCH_INDEX[accel],
    )


def torch_sees(accel: str) -> bool:
    code = {
        "xpu": "import torch; print(torch.xpu.is_available())",
        "cuda": "import torch; print(torch.cuda.is_available())",
        "cpu": "import torch; print(True)",
    }[accel]
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)
    return r.stdout.strip().endswith("True")


def install_microwakeword() -> Path:
    src = assets.home() / "src" / "microWakeWord"
    if not (src / ".git").exists():
        src.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "-q", MWW_REPO, str(src)], check=True)
    subprocess.run(["git", "-C", str(src), "checkout", "-q", MWW_COMMIT], check=True)
    _pip("-e", str(src))
    return src


def missing_tools() -> list[str]:
    return [t for t in ("espeak-ng", "git") if shutil.which(t) is None]
