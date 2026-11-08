"""Live memory of a running job, for the Training page: system RAM and each GPU's memory the job holds.
Linux only (it reads /proc); elsewhere snapshot() is None and the page shows no memory bars.

GPU memory comes from the kernel, never by opening the GPU in the server: the DRM fdinfo of the job's
processes (Intel xe/i915, AMD amdgpu: "drm-resident-vram0"; "drm-resident-gtt" is system RAM mapped for the
GPU, which includes ordinary pinned buffers, so it is reported but not read as an overflow), or nvidia-smi for
NVIDIA cards.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from collections import defaultdict
from pathlib import Path

_KIB = re.compile(r"^(drm-[a-z-]+[0-9]*):\s+(\d+)\s*([KMG]i?B)?", re.MULTILINE)
_UNIT = {"KiB": 1024, "MiB": 1024**2, "GiB": 1024**3, None: 1}


def ram() -> dict:
    info = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        k, _, v = line.partition(":")
        info[k] = int(v.split()[0]) * 1024
    total, avail = info.get("MemTotal", 0), info.get("MemAvailable", 0)
    return {"total_gb": round(total / 2**30, 1), "used_gb": round((total - avail) / 2**30, 1)}


def start_ticks(pid: int) -> int | None:
    """When a process started, in clock ticks since boot (field 22 of stat; the comm field, 2, may hold
    spaces): told apart from a later process with the same pid, whatever the wall clock did since.
    None without /proc."""
    try:
        return int(Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[19])
    except (OSError, ValueError, IndexError):
        return None


def booted_at() -> int | None:
    """When the machine started (wall clock), or None without /proc."""
    try:
        for line in Path("/proc/stat").read_text().splitlines():
            if line.startswith("btime "):
                return int(line.split()[1])
    except (OSError, ValueError, IndexError):
        pass
    return None


def _with_children(root: int) -> list[int]:
    """root and every process below it."""
    kids: dict[int, list[int]] = defaultdict(list)
    for d in Path("/proc").iterdir():
        if d.name.isdigit():
            try:  # field 4 of stat is the parent; the comm field (2) may hold spaces
                kids[int((d / "stat").read_text().rsplit(")", 1)[1].split()[1])].append(int(d.name))
            except (OSError, IndexError, ValueError):
                continue
    out, todo = [], [root]
    while todo:
        pid = todo.pop()
        out.append(pid)
        todo.extend(kids.get(pid, ()))
    return out


def job_pids(job: dict) -> list[int]:
    """The job's processes: its systemd unit's cgroup, else the processes in its session; for a run
    typed in a terminal, whose session is the shell's, the run and what it started."""
    unit = job.get("unit")
    if unit:
        uid = os.getuid()
        f = Path(
            f"/sys/fs/cgroup/user.slice/user-{uid}.slice/user@{uid}.service/app.slice/{unit}.service/cgroup.procs"
        )
        try:
            return [int(x) for x in f.read_text().split()]
        except OSError:
            pass
    run = job.get("run_pid")
    if run and not job.get("own_session"):
        return _with_children(run)
    sid = job.get("pid")
    if not sid:
        return []
    out = []
    for d in Path("/proc").iterdir():
        if d.name.isdigit():
            try:  # field 6 of stat is the session id; the comm field (2) may hold spaces
                if int((d / "stat").read_text().rsplit(")", 1)[1].split()[3]) == sid:
                    out.append(int(d.name))
            except (OSError, IndexError, ValueError):
                continue
    return out


def gpu(pids: list[int]) -> list[dict]:
    """[{"device": pci address or name, "vram_gb", "system_gb"}] held by these processes."""
    per: dict[str, dict[str, float]] = defaultdict(lambda: {"vram": 0.0, "gtt": 0.0})
    seen = set()
    for pid in pids:
        try:
            fds = list(Path(f"/proc/{pid}/fdinfo").iterdir())
        except OSError:
            continue
        for fd in fds:
            try:
                t = fd.read_text()
            except OSError:
                continue
            if "drm-driver" not in t:
                continue
            client = re.search(r"drm-client-id:\s+(\d+)", t)
            dev = re.search(r"drm-pdev:\s+(\S+)", t)
            key = (dev.group(1) if dev else "?", client.group(1) if client else fd.name)
            if key in seen:  # several fds can share one DRM client
                continue
            seen.add(key)
            for name, n, unit in _KIB.findall(t):
                b = int(n) * _UNIT.get(unit or None, 1)
                if name.startswith("drm-resident-vram"):
                    per[key[0]]["vram"] += b
                elif name == "drm-resident-gtt":
                    per[key[0]]["gtt"] += b
    out = [
        {
            "device": d,
            "vram_gb": round(v["vram"] / 2**30, 2),
            "system_gb": round(v["gtt"] / 2**30, 2),
        }
        for d, v in sorted(per.items())
    ]
    if shutil.which("nvidia-smi") and pids:
        try:
            r = subprocess.run(
                ["nvidia-smi", "--query-compute-apps=pid,used_memory,gpu_bus_id", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5, check=False,
            )  # fmt: skip
            nv: dict[str, float] = defaultdict(float)
            for line in r.stdout.splitlines():
                parts = [x.strip() for x in line.split(",")]
                if len(parts) == 3 and parts[0].isdigit() and int(parts[0]) in pids:
                    nv[parts[2]] += float(parts[1]) / 1024
            out += [
                {"device": d, "vram_gb": round(g, 2), "system_gb": 0.0}
                for d, g in sorted(nv.items())
            ]
        except (OSError, subprocess.TimeoutExpired, ValueError):
            pass
    return out


def snapshot(job: dict) -> dict | None:
    if not Path("/proc/meminfo").exists():  # macOS
        return None
    pids = job_pids(job) if job.get("running") else []
    return {"ram": ram(), "gpus": gpu(pids)}
