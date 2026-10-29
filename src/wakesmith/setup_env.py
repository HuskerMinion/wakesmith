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
    # experimental: Radeon cards ROCm supports, on Linux; others fail the check after install
    "rocm": "https://download.pytorch.org/whl/rocm7.2",
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


def nvidia_smi_gpus() -> list[tuple[str, str]]:
    """NVIDIA GPUs that nvidia-smi sees but sysfs doesn't: under WSL2 the card reaches Linux
    through /dev/dxg, not the PCI bus. Windows ships nvidia-smi into /usr/lib/wsl/lib."""
    exe = shutil.which("nvidia-smi") or next(
        (str(p) for p in (Path("/usr/lib/wsl/lib/nvidia-smi"),) if p.exists()), None
    )
    if not exe:
        return []
    try:
        out = subprocess.run(
            [exe, "--query-gpu=pci.bus_id", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        ).stdout
    except (OSError, subprocess.TimeoutExpired):
        return []
    return [("nvidia", line.strip()) for line in out.splitlines() if line.strip()]


def find_gpus() -> list[tuple[str, str]]:
    gpus = pci_gpus()
    if not any(v == "nvidia" for v, _ in gpus):
        gpus += nvidia_smi_gpus()
    return gpus


def choose_accel(gpus: list[tuple[str, str]]) -> str:
    vendors = {v for v, _ in gpus}
    if "nvidia" in vendors:
        return "cuda"
    if "intel" in vendors:
        return "xpu"  # verified after install: integrated GPUs may not be supported
    if "amd" in vendors:
        return "rocm"  # experimental, verified after install (integrated Radeons usually fail)
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
    # Reinstall even when a torch is there: it may be a build for another device. pip has no
    # --reinstall-package (uv only), and its --force-reinstall would reinstall every dependency too.
    if shutil.which("uv"):
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
        return
    subprocess.run(
        [sys.executable, "-m", "pip", "uninstall", "-y", "torch", "torchaudio"], check=True
    )
    try:  # pip can't swap builds in one step: until this succeeds there is no PyTorch at all
        _pip("torch", "torchaudio", "--index-url", TORCH_INDEX[accel])
    except subprocess.CalledProcessError:
        raise RuntimeError(
            "installing PyTorch failed after the old build was removed; run `wakesmith setup` again "
            "(a network problem, most often) to put it back"
        ) from None


# A matrix product on the device, not just "is a device there": an installed build can list a
# card it has no kernels for (a new GPU generation, an integrated Radeon under ROCm).
_RUNS = "x = torch.ones(64, 64, device={dev!r}); print(float((x @ x)[0, 0]) == 64.0)"
_SEES = {
    "xpu": "import torch; assert torch.xpu.is_available(); " + _RUNS.format(dev="xpu"),
    "cuda": "import torch; assert torch.cuda.is_available(); " + _RUNS.format(dev="cuda"),
    "rocm": "import torch; assert torch.cuda.is_available() and torch.version.hip; "
    + _RUNS.format(dev="cuda"),
    "cpu": "import torch; print(True)",
}


def torch_sees(accel: str) -> bool:
    try:
        r = subprocess.run(
            [sys.executable, "-c", _SEES[accel]],
            capture_output=True,
            text=True,
            check=False,
            timeout=300,
        )
    except subprocess.TimeoutExpired:  # a stuck driver: treat as not working
        return False
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
