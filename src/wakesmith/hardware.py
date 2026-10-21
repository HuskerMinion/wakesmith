"""What the machine has, and the settings that follow from it."""

from __future__ import annotations

import os
import shutil
from dataclasses import asdict, dataclass, field

LOW_DISK_GB = 40  # a full run (clips, augmented features, models) needs about this much


def forced_cpu() -> bool:
    """WAKESMITH_CPU=1 (set by --cpu): behave as if there were no GPU."""
    return os.environ.get("WAKESMITH_CPU", "") not in ("", "0")


@dataclass
class Gpu:
    device: str  # torch device string, e.g. "xpu:0"
    name: str
    memory_gb: float


@dataclass
class Hardware:
    accel: str  # xpu | cuda | mps | cpu
    gpus: list[Gpu] = field(default_factory=list)
    cpus: int = 1
    ram_gb: float = 0.0
    ram_free_gb: float = 0.0  # available now (other programs, e.g. a local LLM, hold the rest)
    openvino_gpu: bool = False

    def as_dict(self) -> dict:
        return asdict(self)

    # -- derived settings --------------------------------------------------------------------
    def tts_batch(self) -> int:
        if not self.gpus:
            return 16
        mem = min(g.memory_gb for g in self.gpus)
        return 100 if mem >= 14 else 64 if mem >= 7 else 32

    def tts_devices(self) -> list[str]:
        """One generation process per GPU, limited by free RAM (~4 GB per process)."""
        if not self.gpus:
            return ["cpu"]
        fit = max(1, int(self.ram_free_gb // 5))
        return [g.device for g in self.gpus][:fit]

    def workers(self) -> int:
        """CPU processes for augmentation/features (~1 GB each), limited by free RAM."""
        return max(1, min(self.cpus, int(self.ram_free_gb // 1.5) or 1))


def detect(gpu: bool = True) -> Hardware:
    """gpu=False: CPU/RAM only, without opening the GPU driver (stages that don't use it)."""
    cpus = os.cpu_count() or 1
    ram, free = _ram_gb("MemTotal"), _ram_gb("MemAvailable")
    if not gpu or forced_cpu():
        return Hardware("cpu", [], cpus, ram, free)
    import torch

    if hasattr(torch, "xpu") and torch.xpu.is_available():
        gpus = [
            Gpu(
                f"xpu:{i}",
                torch.xpu.get_device_properties(i).name,
                round(torch.xpu.get_device_properties(i).total_memory / 2**30, 1),
            )
            for i in range(torch.xpu.device_count())
        ]
        return Hardware("xpu", gpus, cpus, ram, free, _openvino_gpu())
    if torch.cuda.is_available():
        gpus = [
            Gpu(
                f"cuda:{i}",
                torch.cuda.get_device_name(i),
                round(torch.cuda.get_device_properties(i).total_memory / 2**30, 1),
            )
            for i in range(torch.cuda.device_count())
        ]
        return Hardware("cuda", gpus, cpus, ram, free)
    if torch.backends.mps.is_available():
        return Hardware("mps", [Gpu("mps", "Apple GPU", ram)], cpus, ram, free)
    return Hardware("cpu", [], cpus, ram, free)


def _ram_gb(key: str) -> float:
    try:
        with open("/proc/meminfo") as f:
            kb = int(next(line for line in f if line.startswith(key)).split()[1])
        return round(kb / 2**20, 1)
    except (OSError, StopIteration):
        return 8.0


def _openvino_gpu() -> bool:
    try:
        import onnxruntime as ort

        return "OpenVINOExecutionProvider" in ort.get_available_providers()
    except ImportError:
        return False


def disk_free_gb(path: str) -> float:
    return round(shutil.disk_usage(path).free / 2**30, 1)


def disk_warning(path: str) -> str:
    free = disk_free_gb(path)
    if free >= LOW_DISK_GB:
        return ""
    return f"only {free:.0f} GB free on the data disk; a full run needs about {LOW_DISK_GB} GB"


def detect_safe(timeout: float = 30.0) -> tuple[Hardware, str]:
    """detect() in a child process, so a hung GPU driver can't hang the caller.

    Returns (hardware, problem). On timeout or error the hardware is CPU-only and `problem` says
    why. A process stuck in the driver can't be killed; it is left behind, the caller goes on."""
    import json
    import subprocess
    import sys

    if forced_cpu():
        return detect(gpu=False), ""
    code = (
        "import json; from dataclasses import asdict; from wakesmith import hardware; "
        "print(json.dumps(asdict(hardware.detect())))"
    )
    proc = subprocess.Popen(
        [sys.executable, "-c", code], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True
    )
    try:
        out, _ = proc.communicate(timeout=timeout)
        if proc.returncode != 0:
            raise ValueError(f"exit {proc.returncode}")
        d = json.loads(out.strip().splitlines()[-1])
        d["gpus"] = [Gpu(**g) for g in d["gpus"]]
        return Hardware(**d), ""
    except subprocess.TimeoutExpired:
        proc.kill()  # don't wait for it: a process stuck in the driver never exits
        why = (
            "the GPU driver is not responding (a crashed GPU job can wedge it; a reboot "
            "usually fixes it)"
        )
    except (ValueError, IndexError) as e:
        why = f"GPU detection failed: {e}"
    return detect(gpu=False), why
