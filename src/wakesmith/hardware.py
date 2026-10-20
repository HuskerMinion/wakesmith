"""What the machine has, and the settings that follow from it."""

from __future__ import annotations

import os
import shutil
from dataclasses import asdict, dataclass, field


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


def detect() -> Hardware:
    import torch

    cpus = os.cpu_count() or 1
    ram, free = _ram_gb("MemTotal"), _ram_gb("MemAvailable")
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
