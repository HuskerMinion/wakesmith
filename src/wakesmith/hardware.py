"""What the machine has, and the settings that follow from it."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass, field

LOW_DISK_GB = 40  # a full run (clips, augmented features, models) needs about this much
LOW_RAM_GB = (
    14  # microWakeWord training holds its negative sets in memory: killed on a 10 GB VM, fine on 16
)


def forced_cpu() -> bool:
    """WAKESMITH_CPU=1 (set by --cpu): behave as if there were no GPU."""
    return os.environ.get("WAKESMITH_CPU", "") not in ("", "0")


def worker_start() -> str:
    """How worker processes start: "fork" on Linux (they share the parent's memory-mapped data), "spawn" on
    macOS, where a child forked after torch, MPS or Accelerate has started can hang. WAKESMITH_WORKER_START
    ("fork" or "spawn") overrides it."""
    forced = os.environ.get("WAKESMITH_WORKER_START", "")
    if forced in ("fork", "spawn"):
        return forced
    return "spawn" if sys.platform == "darwin" else "fork"


def tensorflow_on_cpu():
    """TensorFlow for in-process work (openWakeWord), kept off every GPU unless WAKESMITH_TF_GPU is set.

    Hiding cards with CUDA_VISIBLE_DEVICES stops working once PyTorch has started the CUDA driver in this
    process (speech generation does): TensorFlow then saw an RTX 5060 Ti and failed to compile for it."""
    import tensorflow as tf

    if not os.environ.get("WAKESMITH_TF_GPU"):
        try:
            tf.config.set_visible_devices([], "GPU")
        except RuntimeError:  # devices already initialized in this process: leave them
            pass
    return tf


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
        """WAKESMITH_TTS_BATCH=<n> overrides: smaller batches draw less GPU power (a PC that resets under load)."""
        if os.environ.get("WAKESMITH_TTS_BATCH", "").isdigit():
            return max(1, int(os.environ["WAKESMITH_TTS_BATCH"]))
        if not self.gpus:
            return 16
        # Batch 50 held 12.4 of 16 GB on an RTX 5060 Ti; at 100 it ran out, and under WSL2 the driver spills to
        # system memory instead of failing, until Windows resets the "hung" GPU (TDR) and can freeze.
        mem = min(g.memory_gb for g in self.gpus)
        batch = (
            100 if mem >= 24 else 50 if mem >= 14 else 32 if mem >= 10 else 20 if mem >= 7 else 12
        )
        # 100 is also the ceiling on 32 GB cards: batch 200 of a long phrase at the slowest speed reserved 30.5 GB
        # on an Arc Pro B70 (100: 13.3 GB), and an Intel GPU that runs out spills into system RAM until the
        # kernel's OOM killer steps in. This is only the starting size: tts.generate's memory guard measures
        # each batch and shrinks later ones to fit the card. Intel XPU once needed a lower cap because cut-off clips came from a broken
        # compute runtime (26.05: wrong torch.nonzero/unique, see xpu_selftest); on 26.35, batch 100-400 cut no
        # more clips than the CPU does.
        return batch

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
        # ROCm builds of torch drive AMD cards through the same cuda:N devices
        accel = "rocm" if getattr(torch.version, "hip", None) else "cuda"
        return Hardware(accel, gpus, cpus, ram, free)
    if torch.backends.mps.is_available():
        return Hardware("mps", [Gpu("mps", "Apple GPU", ram)], cpus, ram, free)
    return Hardware("cpu", [], cpus, ram, free)


def _ram_gb(key: str) -> float:
    try:
        with open("/proc/meminfo") as f:
            kb = int(next(line for line in f if line.startswith(key)).split()[1])
        return round(kb / 2**20, 1)
    except (OSError, StopIteration):
        pass
    if sys.platform == "darwin":
        try:
            return round(_mac_ram(key == "MemAvailable") / 2**30, 1)
        except (OSError, ValueError, AttributeError, subprocess.SubprocessError):
            pass
    return 8.0


def _mac_ram(available: bool) -> int:
    """Bytes of RAM on macOS: the total, or what is free for a new process (free, inactive and speculative
    pages, as vm_stat counts them; purgeable pages are already among the active and inactive ones)."""
    total = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    if not available:
        return total
    out = subprocess.run(
        ["/usr/bin/vm_stat"], capture_output=True, text=True, timeout=5, check=True
    ).stdout
    page = int(re.search(r"page size of (\d+) bytes", out).group(1))
    pages = dict(re.findall(r"^Pages ([^:]+):\s+(\d+)\.?$", out, re.MULTILINE))
    free = sum(int(pages.get(k, 0)) for k in ("free", "inactive", "speculative"))
    return min(total, free * page)


def _openvino_gpu() -> bool:
    try:
        import onnxruntime as ort

        return "OpenVINOExecutionProvider" in ort.get_available_providers()
    except ImportError:
        return False


def disk_free_gb(path: str) -> float:
    return round(shutil.disk_usage(path).free / 2**30, 1)


def ram_warning(hw: Hardware) -> str:
    if not hw.ram_gb or hw.ram_gb >= LOW_RAM_GB:
        return ""
    return (
        f"only {hw.ram_gb:.0f} GB RAM; microWakeWord training needs about 16 GB "
        "(train openWakeWord only, or add memory)"
    )


def disk_warning(path: str) -> str:
    free = disk_free_gb(path)
    if free >= LOW_DISK_GB:
        return ""
    return f"only {free:.0f} GB free on the data disk; a full run needs about {LOW_DISK_GB} GB"


_XPU_SELFTEST = """
import json, torch
torch.manual_seed(0)
out = []
for i in range(torch.xpu.device_count()):
    dev = f"xpu:{i}"
    m = torch.rand(3, 5000) < 0.5
    m[1, 100:300] = True  # a run of True: the broken runtimes drop entries from runs
    x = torch.randint(0, 100, (1200,))
    # results are only read back; using bad indices on the GPU would fault it
    nz = torch.equal(torch.nonzero(m.to(dev)).cpu(), torch.nonzero(m))
    got = [t.cpu() for t in torch.unique(x.to(dev), return_inverse=True, return_counts=True)]
    want = torch.unique(x, return_inverse=True, return_counts=True)
    un = all(g.shape == w.shape and torch.equal(g, w) for g, w in zip(got, want))
    out.append({"device": dev, "driver": torch.xpu.get_device_properties(i).driver_version, "ok": nz and un})
print(json.dumps(out))
"""


def xpu_selftest_message(results: list[dict]) -> str:
    bad = [r for r in results if not r["ok"]]
    if not bad:
        return ""
    devices = ", ".join(r["device"] for r in bad)
    return (
        f"the Intel GPU runtime (driver {bad[0]['driver']}) computes wrong results on {devices} "
        "(torch.nonzero/unique, boolean-mask indexing): GPU speech generation and training can crash, "
        "hang the GPU or make bad clips. Update Intel's compute runtime to 26.35 or newer with its IGC "
        "(https://github.com/intel/compute-runtime/releases), or run with --cpu"
    )


def xpu_selftest(timeout: float = 120.0) -> str:
    """Check, in a child process, that the Intel GPU runtime gets torch.nonzero and unique right.

    Intel's compute runtime 26.05 runs PyTorch's xpu kernels for these wrong on Battlemage (and Lunar
    Lake, by public reports): runs of adjacent True values lose entries, so boolean-mask indexing,
    masked_select, isin and unique are wrong too, and their bad indices fault the GPU. 26.35 is right.
    Returns a warning, or "" when the runtime is fine or there's no Intel GPU."""
    import json
    import subprocess
    import sys

    proc = subprocess.Popen(
        [sys.executable, "-c", _XPU_SELFTEST],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    try:
        out, _ = proc.communicate(timeout=timeout)
        if proc.returncode != 0:
            raise ValueError(f"exit {proc.returncode}")
        return xpu_selftest_message(json.loads(out.strip().splitlines()[-1]))
    except subprocess.TimeoutExpired:
        proc.kill()
        return "the Intel GPU self-test did not finish (the GPU driver is not responding)"
    except (ValueError, IndexError) as e:
        return f"the Intel GPU self-test failed to run: {e}"


def detect_safe(timeout: float = 30.0) -> tuple[Hardware, str]:
    """detect() in a child process, so a hung GPU driver can't hang the caller.

    Returns (hardware, problem). On timeout or error the hardware is CPU-only and `problem` says
    why. A process stuck in the driver can't be killed; it is left behind, the caller goes on."""
    import json
    import subprocess
    import sys

    if forced_cpu():
        return detect(gpu=False), ""
    import importlib.util

    if (
        importlib.util.find_spec("torch") is None
    ):  # e.g. after `uv tool install --force`: a fresh env
        return detect(gpu=False), "torch is not installed: run `wakesmith setup`"
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
