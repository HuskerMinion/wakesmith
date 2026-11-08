"""Updating wakesmith from the web page.

The newest GitHub Release's wheel is downloaded, its sha256 checked against the one GitHub lists, and
it is installed into the environment wakesmith runs from. `uv tool install --force` would start a
fresh environment, taking PyTorch and microWakeWord with it (and the files of the running server);
installing in place keeps them, so `wakesmith setup` isn't needed. If the new version doesn't import,
the old one goes back. `wakesmith serve` then replaces itself (same process id, terminal and service).

Only for an installed wheel: a source checkout is updated with git, and never contacts GitHub here.
"""

from __future__ import annotations

import hashlib
import http.client
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from collections.abc import Callable
from pathlib import Path

from . import __version__, assets, jobs, setup_env

REPO = "HuskerMinion/wakesmith"
CHECK_EVERY_S = 12 * 3600
RETRY_S = 3600  # after a failed check (offline, GitHub's limit for anonymous requests)
NET_TIMEOUT_S = 15
INSTALL_TIMEOUT_S = 1800  # a new dependency can be large
MAX_WHEEL_BYTES = 100_000_000
MAX_METADATA_BYTES = 1_000_000
WRITES_WAIT_S = 120  # changes already underway when an update starts get this long to finish
ACTIVE = ("starting", "downloading", "installing", "checking", "restarting")
_SMOKE = (
    "import wakesmith, wakesmith.cli, wakesmith.web.server as s; s.create_app(); "
    "print(wakesmith.__version__)"
)
_REFRESH_MWW = "from wakesmith import setup_env; setup_env.refresh_microwakeword()"


def parse(version: str) -> tuple[int, ...] | None:
    """(0, 4, 8) for a plain release number; None for anything else (0.3.0.dev0, a stray tag)."""
    if not re.fullmatch(r"[0-9]{1,6}(\.[0-9]{1,6}){1,3}", version):
        return None
    return tuple(map(int, version.split(".")))


def source_install() -> bool:
    """Running from a source tree, or a development version: git's job, not this module's."""
    root = Path(__file__).resolve().parents[2]
    return (root / "pyproject.toml").is_file() or parse(__version__) is None


def _settings_file() -> Path:
    return assets.home() / "update.json"


def auto_check() -> bool:
    try:
        return json.loads(_settings_file().read_text()).get("auto", True) is not False
    except (OSError, ValueError, AttributeError):
        return True


def set_auto_check(on: bool) -> None:
    f = _settings_file()
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps({"auto": bool(on)}) + "\n")


class _OnlyGitHub(urllib.request.HTTPRedirectHandler):
    """A wheel's address redirects to GitHub's storage; nowhere else, and never to plain http."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        u = urllib.parse.urlsplit(newurl)
        try:
            host, port = u.hostname or "", u.port
        except ValueError:
            host, port = "", None
        ok = host == "github.com" or host.endswith((".github.com", ".githubusercontent.com"))
        # the address urllib connects to is the raw one: anything its parts could disagree on is out
        plain = re.fullmatch(r"[a-z0-9.-]+(:443)?", u.netloc.lower()) and port in (None, 443)
        if u.scheme != "https" or not ok or not plain:
            raise urllib.error.URLError(f"refused a redirect to {u.scheme}://{host or u.netloc}")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


_OPENER = urllib.request.build_opener(_OnlyGitHub)


def _get(url: str, timeout: float = NET_TIMEOUT_S, accept: str = "application/vnd.github+json"):
    req = urllib.request.Request(
        url, headers={"Accept": accept, "User-Agent": f"wakesmith/{__version__}"}
    )
    return _OPENER.open(req, timeout=timeout)


def release(which: str = "latest") -> dict:
    """A release ("latest", or "tags/v0.4.7") and its wheel. Addresses are built here, not taken from
    the answer, so a download only ever comes from this project's releases."""
    with _get(f"https://api.github.com/repos/{REPO}/releases/{which}") as r:
        data = json.load(r)
    if not isinstance(data, dict) or not isinstance(data.get("assets", []), list):
        raise ValueError("GitHub's answer isn't a release")  # noqa: TRY004 - reported as a failed check
    tag = str(data.get("tag_name", ""))
    version = tag.removeprefix("v")
    if not tag.startswith("v") or parse(version) is None:
        raise ValueError(f"the release is tagged {tag!r}, not a version")
    name = f"wakesmith-{version}-py3-none-any.whl"
    assets_ = [a for a in data.get("assets", []) if isinstance(a, dict)]
    asset = next((a for a in assets_ if a.get("name") == name), None)
    if asset is None:
        raise ValueError(f"release {tag} has no {name}")
    sha = re.fullmatch(r"sha256:([0-9a-f]{64})", str(asset.get("digest", "")))
    if not sha:
        raise ValueError(f"GitHub lists no sha256 for {name}, so it can't be checked")
    return {
        "version": version,
        "name": name,
        "sha256": sha[1],
        "url": f"https://github.com/{REPO}/releases/download/{tag}/{name}",
        "page": f"https://github.com/{REPO}/releases/tag/{tag}",
    }


def download(rel: dict, folder: Path) -> Path:
    """The release's wheel in `folder`, checked: its sha256, and that it says it's that version."""
    out = folder / rel["name"]
    sha = hashlib.sha256()
    size = 0
    with _get(rel["url"], timeout=60, accept="application/octet-stream") as r, out.open("wb") as f:
        while chunk := r.read(1 << 20):
            size += len(chunk)
            if size > MAX_WHEEL_BYTES:
                raise ValueError(f"{rel['name']} is larger than a wheel of wakesmith can be")
            sha.update(chunk)
            f.write(chunk)
    if sha.hexdigest() != rel["sha256"]:
        raise ValueError(f"{rel['name']} doesn't match the sha256 GitHub lists: not installed")
    try:
        with zipfile.ZipFile(out) as z:
            entry = z.getinfo(f"wakesmith-{rel['version']}.dist-info/METADATA")
            if entry.file_size > MAX_METADATA_BYTES:
                raise ValueError(f"{rel['name']} isn't a wheel of wakesmith: not installed")
            meta = z.read(entry).decode()
    except (zipfile.BadZipFile, KeyError) as e:
        raise ValueError(f"{rel['name']} isn't a wheel of that version: not installed") from e
    if f"\nVersion: {rel['version']}\n" not in meta:
        raise ValueError(f"{rel['name']} says it's another version: not installed")
    return out


def _run(cmd: list[str]) -> str:
    """Run from the data folder (a folder named wakesmith in the current one could be imported instead)."""
    home = assets.home()
    home.mkdir(parents=True, exist_ok=True)
    # Its own session and files, no pipes: Ctrl-C in the server's terminal must not cut an install in
    # half, neither by the signal nor by a pipe that closes when the server goes.
    with tempfile.TemporaryFile() as out, tempfile.TemporaryFile() as err:
        r = subprocess.run(
            cmd,
            cwd=home,
            stdout=out,
            stderr=err,
            timeout=INSTALL_TIMEOUT_S,
            start_new_session=True,
            check=False,
        )
        out.seek(0)
        err.seek(0)
        stdout = out.read().decode(errors="replace")
        stderr = err.read().decode(errors="replace")
    if r.returncode:
        raise RuntimeError((stderr or stdout).strip()[-800:] or f"exited with {r.returncode}")
    return stdout


def _install(rel: dict, downloaded: Callable[[], None] = lambda: None) -> None:
    assets.home().mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=assets.home(), prefix="update-") as d:
        wheel = download(rel, Path(d))
        downloaded()
        _run(setup_env.pip_command(str(wheel)))


def _works(version: str) -> None:
    """The installed wakesmith imports, makes its app and is `version`; RuntimeError says otherwise."""
    out = _run([sys.executable, "-c", _SMOKE]).strip()
    got = out.splitlines()[-1] if out else ""
    if got != version:
        raise RuntimeError(f"it reports version {got or 'nothing'}, not {version}")


def clean_leftovers() -> None:
    """Download folders of an update that was cut off (the server stopped in the middle); not while
    another server's update is still going."""
    if assets.update_underway():
        return
    for d in assets.home().glob("update-*"):
        if d.is_dir():
            shutil.rmtree(d, ignore_errors=True)


def restart_command() -> list[str]:
    """The argv that starts this server again: the console script of this environment (a `python -m`
    could import a folder named wakesmith from the current directory instead)."""
    script = Path(sys.executable).parent / "wakesmith"
    if script.is_file():
        return [str(script), *sys.argv[1:]]
    return [sys.executable, "-m", "wakesmith", *sys.argv[1:]]


def restart() -> None:
    """Replace this process with the new version. Call after the server has closed its port."""
    cmd = restart_command()
    sys.stdout.flush()
    sys.stderr.flush()
    os.execv(cmd[0], cmd)


def busy() -> str | None:
    """Why updating has to wait: a run imports wakesmith's files while it works and starts new
    processes from them, and restarting would cut off the queue's launcher."""
    if running := jobs.running_all():
        names = ", ".join(p.name for p in running)
        return f"Training is running ({names}): update when it has finished."
    if queued := jobs.queue():
        return f"{len(queued)} run{'s are' if len(queued) > 1 else ' is'} queued: update when they have finished."
    return None


class Updater:
    """What the page knows about updates, and the one update that may be running."""

    def __init__(self) -> None:
        self.latest: dict | None = None
        self.check_error: str | None = None
        self.checked = 0.0
        self.tried = 0.0
        self.job: dict = {"stage": None, "error": None, "target": None}
        self.restart: Callable[[], None] | None = None  # set by `wakesmith serve`
        self.restarting = False
        self.waiting: Callable[[], str | None] = lambda: None  # what only the server knows about
        self.writes = 0  # changes being made through the server now (its middleware counts them)
        self._lock = threading.Lock()
        self._check_lock = threading.Lock()

    def active(self) -> bool:
        return self.job["stage"] in ACTIVE

    def frozen(self) -> bool:
        """Files are being replaced, or are new while this server still runs the old code: nothing
        may start until wakesmith has been started again."""
        return self.active() or self.job["stage"] == "installed"

    def check(self, force: bool = False) -> None:
        """Ask GitHub for the newest release: on request, or when due and automatic checks are on."""
        if source_install():
            return
        with self._check_lock:
            wait = RETRY_S if self.check_error else CHECK_EVERY_S
            if not force and (not auto_check() or time.time() - self.tried < wait):
                return
            self.tried = time.time()
            try:
                self.latest, self.check_error = release(), None
                self.checked = self.tried
            except (OSError, ValueError, http.client.HTTPException) as e:  # URLError is an OSError
                self.check_error = f"Couldn't check for updates: {e}"

    def available(self) -> bool:
        if source_install() or not self.latest:
            return False
        return parse(self.latest["version"]) > parse(__version__)

    def _why_wait(self) -> str | None:
        return self.waiting() or busy()

    def status(self) -> dict:
        latest = self.latest
        return {
            "version": __version__,
            "source": source_install(),
            "auto": auto_check(),
            "available": self.available(),
            "latest": {"version": latest["version"], "page": latest["page"]} if latest else None,
            "checked": self.checked or None,
            "check_error": self.check_error,
            "busy": self._why_wait(),
            "job": dict(self.job),
        }

    def start(self, version: str) -> dict:
        """Start updating to `version`, the one the person agreed to. ValueError says why not. The job
        is claimed first: from then on the server refuses changes, so nothing starts during the check
        with GitHub, and whatever was underway is waited for in _work."""
        if source_install():
            raise ValueError("This wakesmith runs from a source folder: update it with git.")
        with self._lock:
            if self.active():
                raise ValueError("An update is already running.")
            self.job = {"stage": "starting", "error": None, "target": version}
        try:
            if why := self._why_wait():
                raise ValueError(why)
            self.check(force=True)
            if self.check_error:
                raise ValueError(self.check_error)
            if not self.available():
                raise ValueError("No newer version to update to.")
            if self.latest["version"] != version:
                raise ValueError(
                    f"The newest version is now {self.latest['version']}: look at it first, then update."
                )
        except BaseException:
            self.job = {"stage": None, "error": None, "target": None}
            raise
        threading.Thread(target=self._work, args=(self.latest,), daemon=True).start()
        return dict(self.job)

    def _settled(self) -> str | None:
        """Wait for changes already underway to finish (a run being started, an upload, the update's
        own request); why the update can't go on, or None. Nothing new can begin by now."""
        until = time.monotonic() + WRITES_WAIT_S
        while self.writes > 0:
            if time.monotonic() > until:
                return "Something is still being changed: try again when it has finished."
            time.sleep(0.05)
        return self._why_wait()

    def _roll_back(self, why: Exception) -> str:
        old = __version__
        try:
            _install(release(f"tags/v{old}"))
            _works(old)
        except Exception as e:  # noqa: BLE001 - the message says what's wrong and how to fix it
            return (
                f"The new version didn't work ({why}), and going back to {old} failed too ({e}). "
                f"Get wakesmith-{old}-py3-none-any.whl from the Releases page and run: uv tool install "
                f"--python 3.12 --force ./wakesmith-{old}-py3-none-any.whl, then wakesmith setup"
            )
        return f"The new version didn't work, so {old} was put back: {why}"

    def _work(self, rel: dict) -> None:
        """The whole update under the lock that tells other processes (a run from the command line, the
        timed device sync) it's going. Held on after new files are installed: it ends with this process."""
        try:
            lock = assets.hold_update_lock()
        except OSError as e:
            self.job.update(stage=None, error=f"The update failed: {e}")
            return
        if lock is None:
            self.job.update(stage=None, error="Another wakesmith is updating right now.")
            return
        try:
            self._steps(rel)
        finally:
            if self.job["stage"] is None:
                os.close(lock)

    def _steps(self, rel: dict) -> None:
        try:
            if why := self._settled():
                self.job.update(stage=None, error=why)
                return
            self.job["stage"] = "downloading"

            def downloaded() -> None:  # the download took a while: a look before anything changes
                if why := self._why_wait():
                    raise ValueError(why)
                self.job["stage"] = "installing"

            _install(rel, downloaded)
        except Exception as e:  # noqa: BLE001 - shown on the page
            self.job["error"] = f"The update failed: {e}"
            if (
                self.job["stage"] == "installing"
            ):  # an installer cut off can leave the old one broken
                try:
                    _works(__version__)
                except Exception:  # noqa: BLE001 - whatever is wrong, the old version has to go back
                    self.job["error"] = self._roll_back(e)
            self.job["stage"] = None
            return
        self.job["stage"] = "checking"
        try:
            _works(rel["version"])
        except Exception as e:  # noqa: BLE001 - whatever is wrong, the old version goes back
            self.job.update(stage=None, error=self._roll_back(e))
            return
        try:
            _run([sys.executable, "-c", _REFRESH_MWW])
        except Exception as e:  # noqa: BLE001 - the new wakesmith is fine: no reason to take it back
            self.job.update(
                stage="installed",
                error=f"microWakeWord couldn't follow ({e}): stop wakesmith, run wakesmith setup, then start wakesmith again.",
            )
            return
        if self.restart is None:
            self.job["stage"] = "installed"  # not under `wakesmith serve`: the page says to restart
            return
        self.job["stage"] = "restarting"
        self.restarting = True
        self.restart()
