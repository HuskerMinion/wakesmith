"""Updating from the web page: what's offered, what's refused, and that nothing bad gets installed."""

import hashlib
import io
import json
import logging
import os
import subprocess
import sys
import threading
import time
import zipfile

import pytest
from fastapi.testclient import TestClient

from wakesmith import assets, jobs, setup_env, update
from wakesmith.web.server import create_app

OLD, NEW = "1.0.0", "1.1.0"


def wheel(version=NEW, says=None):
    b = io.BytesIO()
    with zipfile.ZipFile(b, "w") as z:
        z.writestr(
            f"wakesmith-{version}.dist-info/METADATA",
            f"Name: wakesmith\nVersion: {says or version}\n\n",
        )
    return b.getvalue()


def answer(version=NEW, data=None, digest=None):
    digest = digest or "sha256:" + hashlib.sha256(data or wheel(version)).hexdigest()
    name = f"wakesmith-{version}-py3-none-any.whl"
    return {"tag_name": f"v{version}", "assets": [{"name": name, "digest": digest}]}


class Net:
    """GitHub: release answers by the address's end, wheels by file name, and a count of requests."""

    def __init__(self, monkeypatch):
        self.releases = {"latest": answer(), f"tags/v{OLD}": answer(OLD)}
        self.wheels = {NEW: wheel(), OLD: wheel(OLD)}
        self.urls = []
        self.fail = None
        monkeypatch.setattr(update, "_get", self.get)

    def get(self, url, timeout=0, accept=""):
        self.urls.append(url)
        if self.fail:
            raise OSError(self.fail)
        if "/releases/download/" in url:
            return io.BytesIO(self.wheels[url.split("/")[-2].removeprefix("v")])
        return io.BytesIO(json.dumps(self.releases[url.split("/releases/")[1]]).encode())


@pytest.fixture
def net(monkeypatch, tmp_path):
    monkeypatch.setenv("WAKESMITH_HOME", str(tmp_path / "home"))
    monkeypatch.setattr(update, "__version__", OLD)
    monkeypatch.setattr(update, "source_install", lambda: False)
    monkeypatch.setattr(update, "busy", lambda: None)
    monkeypatch.setattr(setup_env, "pip_command", lambda *a: ["pip", *a])
    return Net(monkeypatch)


@pytest.fixture
def client():
    app = create_app()
    c = TestClient(app, base_url="http://localhost")
    c.updater = app.state.updater
    return c


def commands(monkeypatch, smoke=(NEW,)):
    """Stand in for the installer and the checks; returns the commands run. smoke: what the installed
    version reports, one answer per check (the last one repeats)."""
    ran, answers = [], list(smoke)

    def run(cmd):
        ran.append(cmd)
        if update._SMOKE in cmd:
            return (answers.pop(0) if len(answers) > 1 else answers[0]) + "\n"
        return ""

    monkeypatch.setattr(update, "_run", run)
    return ran


def wait(updater):
    for _ in range(500):
        if not updater.active() or updater.restarting:
            return updater.job
        time.sleep(0.01)
    raise AssertionError(updater.job)


def test_versions_compare_as_numbers_and_only_plain_releases_count():
    assert update.parse("0.10.0") > update.parse("0.9.9")
    for odd in ("0.3.0.dev0", "", "1", "\u0660.\u0664.\u0669", "9" * 5000 + ".1", "1.2.3.4.5"):
        assert update.parse(odd) is None


def test_this_checkout_counts_as_a_source_install_and_never_contacts_github(monkeypatch, tmp_path):
    monkeypatch.setenv("WAKESMITH_HOME", str(tmp_path))
    assert update.source_install()
    monkeypatch.setattr(update, "_get", lambda *a, **k: pytest.fail("contacted GitHub"))
    c = TestClient(create_app(), base_url="http://localhost")
    assert c.post("/api/update/check").json()["source"] is True
    r = c.post("/api/update", json={"version": NEW})
    assert r.status_code == 409 and "git" in r.json()["detail"]


def test_a_newer_release_is_offered_and_checked_only_when_due(net, client):
    u = client.get("/api/update").json()
    assert u["available"] and u["latest"]["version"] == NEW and u["version"] == OLD
    client.get("/api/update")
    assert len(net.urls) == 1
    client.post("/api/update/check")
    assert len(net.urls) == 2


def test_the_same_or_older_release_is_not_offered(net, client):
    net.releases["latest"] = answer(OLD)
    assert client.get("/api/update").json()["available"] is False


def test_with_automatic_checks_off_github_is_asked_only_on_request(net, client):
    assert client.put("/api/update/settings", json={"auto": False}).json()["auto"] is False
    assert client.get("/api/update").json()["latest"] is None and not net.urls
    assert client.post("/api/update/check").json()["available"] is True
    # it stays off for the next server
    assert (
        TestClient(create_app(), base_url="http://localhost").get("/api/update").json()["auto"]
        is False
    )


def test_a_failed_check_is_said_and_offers_nothing(net, client):
    net.fail = "no network"
    u = client.get("/api/update").json()
    assert not u["available"] and "no network" in u["check_error"]


@pytest.mark.parametrize(
    "release",
    [
        {"tag_name": "nightly", "assets": []},  # not a version
        {"tag_name": f"v{NEW}", "assets": []},  # no wheel
        {
            "tag_name": f"v{NEW}",
            "assets": [{"name": f"wakesmith-{NEW}-py3-none-any.whl"}],
        },  # no sha256
    ],
)
def test_a_release_it_cant_use_is_not_offered(net, client, release):
    net.releases["latest"] = release
    u = client.get("/api/update").json()
    assert not u["available"] and u["check_error"]


def test_downloads_only_come_from_this_projects_releases(net):
    rel = update.release()
    assert (
        rel["url"]
        == f"https://github.com/{update.REPO}/releases/download/v{NEW}/wakesmith-{NEW}-py3-none-any.whl"
    )


def test_update_installs_the_checked_wheel_then_restarts(net, client, monkeypatch):
    ran = commands(monkeypatch)
    restarted = []
    client.updater.restart = lambda: restarted.append(True)
    assert client.post("/api/update", json={"version": NEW}).status_code == 200
    job = wait(client.updater)
    assert job["stage"] == "restarting" and job["error"] is None
    assert restarted and client.updater.restarting
    assert ran[0][0] == "pip" and ran[0][1].endswith(f"wakesmith-{NEW}-py3-none-any.whl")
    assert update._SMOKE in ran[1] and update._REFRESH_MWW in ran[2]


def test_without_a_server_to_restart_the_page_says_to_start_it_again(net, client, monkeypatch):
    commands(monkeypatch)
    client.post("/api/update", json={"version": NEW})
    assert wait(client.updater)["stage"] == "installed"


@pytest.mark.parametrize(
    "data, why",
    [
        (b"something else", "sha256"),
        (wheel(says="9.9.9"), "another version"),
    ],
)
def test_a_wheel_that_isnt_what_was_promised_is_not_installed(net, client, monkeypatch, data, why):
    ran = commands(monkeypatch)
    net.wheels[NEW] = data
    if why != "sha256":  # listed correctly, but the wrong thing inside
        net.releases["latest"] = answer(NEW, data)
    client.post("/api/update", json={"version": NEW})
    job = wait(client.updater)
    assert why in job["error"] and not ran and job["stage"] is None


def test_a_zip_that_isnt_a_wheel_is_not_installed(net, client, monkeypatch):
    ran = commands(monkeypatch)
    net.wheels[NEW] = b"not a zip"
    net.releases["latest"] = answer(NEW, b"not a zip")
    client.post("/api/update", json={"version": NEW})
    assert "isn't a wheel" in wait(client.updater)["error"] and not ran


def test_a_new_version_that_doesnt_start_is_replaced_by_the_old_one(net, client, monkeypatch):
    ran = commands(monkeypatch, smoke=("something else", OLD))
    client.updater.restart = lambda: pytest.fail("restarted a broken version")
    client.post("/api/update", json={"version": NEW})
    job = wait(client.updater)
    assert job["stage"] is None and f"{OLD} was put back" in job["error"]
    assert any(c[0] == "pip" and c[1].endswith(f"wakesmith-{OLD}-py3-none-any.whl") for c in ran)
    assert update._SMOKE in ran[-1]  # and the old one was checked too


def test_when_going_back_fails_too_it_says_how_to_fix_it(net, client, monkeypatch):
    commands(monkeypatch, smoke=("something else",))
    del net.wheels[OLD]
    client.post("/api/update", json={"version": NEW})
    error = wait(client.updater)["error"]
    assert "uv tool install" in error and "wakesmith setup" in error


def test_a_failed_install_changes_nothing_and_says_so(net, client, monkeypatch):
    def run(cmd):
        raise RuntimeError("no space left")

    monkeypatch.setattr(update, "_run", run)
    client.post("/api/update", json={"version": NEW})
    job = wait(client.updater)
    assert "no space left" in job["error"] and job["stage"] is None


def test_it_waits_while_anything_runs(net, client, monkeypatch):
    ran = commands(monkeypatch)
    monkeypatch.setattr(update, "busy", lambda: "Training is running (x)")
    r = client.post("/api/update", json={"version": NEW})
    assert r.status_code == 409 and "Training" in r.json()["detail"] and not ran
    assert client.get("/api/update").json()["busy"] == "Training is running (x)"


def test_busy_names_runs_and_queues(monkeypatch):
    running = type("P", (), {"name": "hey_jarvis"})()
    monkeypatch.setattr(jobs, "running_all", lambda: [running])
    assert "hey_jarvis" in update.busy()
    monkeypatch.setattr(jobs, "running_all", list)
    monkeypatch.setattr(jobs, "queue", lambda: [{}, {}])
    assert "2 runs are queued" in update.busy()
    monkeypatch.setattr(jobs, "queue", list)
    assert update.busy() is None


def test_it_refuses_a_version_the_person_didnt_agree_to(net, client, monkeypatch):
    ran = commands(monkeypatch)
    r = client.post("/api/update", json={"version": "1.0.5"})
    assert r.status_code == 409 and NEW in r.json()["detail"] and not ran


def test_nothing_new_starts_during_an_update_but_the_page_can_watch(net, client, monkeypatch):
    go = threading.Event()
    monkeypatch.setattr(update, "_install", lambda *a: go.wait(5))
    commands(monkeypatch)
    client.post("/api/update", json={"version": NEW})
    assert client.post("/api/update", json={"version": NEW}).status_code == 503
    assert client.post("/api/projects", json={}).status_code == 503
    assert client.get("/api/update").json()["job"]["stage"] in update.ACTIVE
    go.set()
    assert wait(client.updater)["stage"] == "installed"
    # new files, old code still running: nothing may start until it's been started again
    assert client.post("/api/projects", json={}).status_code == 503


def test_another_site_cant_start_an_update(net, client):
    r = client.post(
        "/api/update", json={"version": NEW}, headers={"origin": "https://evil.example"}
    )
    assert r.status_code == 403


def test_the_pin_is_followed_only_when_it_changed(monkeypatch, tmp_path):
    monkeypatch.setenv("WAKESMITH_HOME", str(tmp_path))
    setup_env.refresh_microwakeword()  # no clone: nothing to do, nothing run
    (tmp_path / "src/microWakeWord/.git").mkdir(parents=True)
    calls, installs = [], []
    head = setup_env.MWW_COMMIT

    def run(cmd, **k):
        calls.append(cmd)
        return type("R", (), {"stdout": head + "\n"})()

    monkeypatch.setattr(setup_env.subprocess, "run", run)
    monkeypatch.setattr(setup_env, "install_microwakeword", lambda: installs.append(True))
    setup_env.refresh_microwakeword()
    assert len(calls) == 1 and not installs
    head = "0" * 40
    setup_env.refresh_microwakeword()
    assert "fetch" in calls[-1] and installs


def test_restart_runs_this_environments_script_with_the_same_arguments(monkeypatch, tmp_path):
    (tmp_path / "wakesmith").write_text("")
    monkeypatch.setattr(update.sys, "executable", str(tmp_path / "python"))
    monkeypatch.setattr(update.sys, "argv", ["x", "serve", "--port", "1"])
    assert update.restart_command() == [str(tmp_path / "wakesmith"), "serve", "--port", "1"]
    (tmp_path / "wakesmith").unlink()
    assert update.restart_command()[1:] == ["-m", "wakesmith", "serve", "--port", "1"]


def test_nothing_can_start_while_github_is_being_asked(net, client, monkeypatch):
    """The job is claimed before the check, so a run can't slip in during it."""
    commands(monkeypatch)
    asked, go, seen = threading.Event(), threading.Event(), {}
    real = update.release

    def slow(which="latest"):
        asked.set()
        go.wait(5)
        return real(which)

    monkeypatch.setattr(update, "release", slow)
    t = threading.Thread(target=lambda: client.post("/api/update", json={"version": NEW}))
    t.start()
    assert asked.wait(5)
    seen["write"] = client.post("/api/projects", json={}).status_code
    go.set()
    t.join(5)
    assert seen["write"] == 503
    wait(client.updater)


def test_a_refused_update_leaves_nothing_claimed(net, client, monkeypatch):
    commands(monkeypatch)
    assert client.post("/api/update", json={"version": "1.0.5"}).status_code == 409
    assert not client.updater.active() and client.post("/api/projects", json={}).status_code != 503
    net.fail = "no network"
    assert client.post("/api/update", json={"version": NEW}).status_code == 409
    assert not client.updater.active()


def test_a_run_that_began_before_the_claim_stops_the_update(net, client, monkeypatch):
    """A start that was underway when the job was claimed shows up once it has finished."""
    ran = commands(monkeypatch)
    u = client.updater
    answers = iter([None])
    monkeypatch.setattr(update, "busy", lambda: next(answers, "Training is running (x)"))
    u.restart = lambda: pytest.fail("restarted")
    assert client.post("/api/update", json={"version": NEW}).status_code == 200
    job = wait(u)
    assert "Training" in job["error"] and job["stage"] is None and not ran


def test_changes_underway_are_waited_for(net, client, monkeypatch):
    ran = commands(monkeypatch)
    u = client.updater
    u.writes = 1  # a change another request is still making
    assert client.post("/api/update", json={"version": NEW}).status_code == 200
    time.sleep(0.2)
    assert u.job["stage"] == "starting" and not ran
    u.writes -= 1
    assert wait(u)["stage"] == "installed" and ran


def test_changes_that_never_finish_stop_the_update(net, client, monkeypatch):
    ran = commands(monkeypatch)
    monkeypatch.setattr(update, "WRITES_WAIT_S", 0.05)
    client.updater.writes = 1
    client.post("/api/update", json={"version": NEW})
    assert "still being changed" in wait(client.updater)["error"] and not ran


def test_the_middleware_counts_writes_only_while_they_run(net, client):
    client.post("/api/update/check")
    client.get("/api/update")
    assert client.updater.writes == 0


def test_a_broken_microwakeword_step_doesnt_take_back_a_good_version(net, client, monkeypatch):
    ran = []

    def run(cmd):
        ran.append(cmd)
        if update._REFRESH_MWW in cmd:
            raise RuntimeError("no route to github")
        return NEW + "\n" if update._SMOKE in cmd else ""

    monkeypatch.setattr(update, "_run", run)
    client.updater.restart = lambda: pytest.fail("restarted without microWakeWord")
    client.post("/api/update", json={"version": NEW})
    job = wait(client.updater)
    assert job["stage"] == "installed" and "wakesmith setup" in job["error"]
    assert sum(c[0] == "pip" for c in ran) == 1  # nothing rolled back


def test_an_install_cut_off_with_the_old_version_broken_rolls_back(net, client, monkeypatch):
    ran, installs = [], []

    def run(cmd):
        ran.append(cmd)
        if cmd[0] == "pip":
            installs.append(cmd)
            if len(installs) == 1:
                raise RuntimeError("no space left")
            return ""
        return (OLD if len(installs) > 1 else "") + "\n"

    monkeypatch.setattr(update, "_run", run)
    client.post("/api/update", json={"version": NEW})
    job = wait(client.updater)
    assert f"{OLD} was put back" in job["error"] and "no space left" in job["error"]
    assert installs[-1][1].endswith(f"wakesmith-{OLD}-py3-none-any.whl")


def test_an_install_refused_with_the_old_version_fine_just_says_so(net, client, monkeypatch):
    installs = []

    def run(cmd):
        if cmd[0] == "pip":
            installs.append(cmd)
            raise RuntimeError("no matching dependency")
        return OLD + "\n"

    monkeypatch.setattr(update, "_run", run)
    client.post("/api/update", json={"version": NEW})
    job = wait(client.updater)
    assert job["error"] == "The update failed: no matching dependency" and len(installs) == 1


def test_the_check_makes_the_app_and_takes_the_last_line_of_output(monkeypatch):
    assert "create_app()" in update._SMOKE
    monkeypatch.setattr(update, "_run", lambda cmd: f"a note from a library\n{NEW}\n")
    update._works(NEW)
    with pytest.raises(RuntimeError):
        update._works(OLD)


def test_a_cut_off_answer_from_github_is_a_failed_check(net, client, monkeypatch):
    import http.client

    def cut(*a, **k):
        raise http.client.IncompleteRead(b"")

    monkeypatch.setattr(update, "_get", cut)
    u = client.get("/api/update").json()
    assert not u["available"] and u["check_error"]


@pytest.mark.parametrize(
    "answer", [[], {"tag_name": f"v{NEW}", "assets": "x"}, {"tag_name": f"v{NEW}", "assets": ["x"]}]
)
def test_a_nonsense_answer_from_github_is_a_failed_check(net, client, answer):
    net.releases["latest"] = answer
    u = client.get("/api/update").json()
    assert not u["available"] and u["check_error"]


@pytest.mark.parametrize(
    "to, ok",
    [
        ("https://release-assets.githubusercontent.com/x.whl", True),
        ("https://github.com/x.whl", True),
        ("http://github.com/x.whl", False),
        ("https://evil.example/x.whl", False),
        ("https://github.com.evil.example/x.whl", False),
        ("https://notgithub.com/x.whl", False),
        ("https://evil.com@github.com/x.whl", False),
        ("https://evil.com\\@github.com/x.whl", False),
        ("https://evil.com%23.github.com/x.whl", False),
        ("https://evil.com:80@x.github.com/x.whl", False),
        ("https://github.com:8443/x.whl", False),
        ("https://github.com:notaport/x.whl", False),
        ("https://github.com:443/x.whl", True),
    ],
)
def test_redirects_only_go_to_github_over_https(to, ok):
    import urllib.request

    req = urllib.request.Request("https://github.com/a.whl")
    h = update._OnlyGitHub()
    if ok:
        assert h.redirect_request(req, None, 302, "", {}, to).full_url == to
    else:
        with pytest.raises(urllib.error.URLError):
            h.redirect_request(req, None, 302, "", {}, to)


def test_an_oversized_metadata_entry_is_not_read(net, client, monkeypatch):
    ran = commands(monkeypatch)
    monkeypatch.setattr(update, "MAX_METADATA_BYTES", 5)
    client.post("/api/update", json={"version": NEW})
    assert "isn't a wheel" in wait(client.updater)["error"] and not ran


def test_leftover_download_folders_are_cleared(monkeypatch, tmp_path):
    monkeypatch.setenv("WAKESMITH_HOME", str(tmp_path))
    (tmp_path / "update-abc").mkdir()
    (tmp_path / "update-abc" / "x.whl").write_bytes(b"x")
    (tmp_path / "update.json").write_text("{}")
    lock = assets.hold_update_lock()
    update.clean_leftovers()
    assert (tmp_path / "update-abc").exists()  # another server's update is still going
    os.close(lock)
    update.clean_leftovers()
    assert not (tmp_path / "update-abc").exists() and (tmp_path / "update.json").exists()


def test_the_lock_tells_when_an_update_is_going_and_ends_with_the_process(monkeypatch, tmp_path):
    monkeypatch.setenv("WAKESMITH_HOME", str(tmp_path))
    assert not assets.update_underway()  # no file yet
    lock = assets.hold_update_lock()
    assert assets.update_underway() and assets.hold_update_lock() is None
    os.close(lock)
    assert not assets.update_underway()
    # a server that dies in the middle, however: nothing stale stays behind
    code = "from wakesmith import assets; assets.hold_update_lock(); print(1, flush=True); input()"
    p = subprocess.Popen(
        [sys.executable, "-c", code], stdin=subprocess.PIPE, stdout=subprocess.PIPE
    )
    try:
        assert p.stdout.readline().strip() == b"1" and assets.update_underway()
    finally:
        p.kill()
        p.wait()
    assert not assets.update_underway()


def test_the_lock_covers_the_whole_update(net, client, monkeypatch):
    seen = []

    def run(cmd):
        seen.append(assets.update_underway())
        return NEW + "\n" if update._SMOKE in cmd else ""

    monkeypatch.setattr(update, "_run", run)
    client.updater.restart = lambda: seen.append(assets.update_underway())
    client.post("/api/update", json={"version": NEW})
    wait(client.updater)
    assert len(seen) == 4 and all(seen)  # install, check, microWakeWord, up to the restart


def test_a_failed_update_lets_go_of_the_lock(net, client, monkeypatch):
    def run(cmd):
        if cmd[0] == "pip":
            raise RuntimeError("no matching dependency")
        return OLD + "\n"

    monkeypatch.setattr(update, "_run", run)
    client.post("/api/update", json={"version": NEW})
    wait(client.updater)
    for _ in range(200):
        if not assets.update_underway():
            break
        time.sleep(0.01)
    assert not assets.update_underway()


def test_another_servers_update_stops_this_one(net, client, monkeypatch):
    ran = commands(monkeypatch)
    lock = assets.hold_update_lock()
    try:
        client.post("/api/update", json={"version": NEW})
        assert "Another wakesmith" in wait(client.updater)["error"] and not ran
    finally:
        os.close(lock)


def test_installs_run_outside_the_servers_process_group(monkeypatch, tmp_path):
    monkeypatch.setenv("WAKESMITH_HOME", str(tmp_path))
    assert (
        update._run([sys.executable, "-c", "import os; print(os.getsid(0) == os.getpid())"])
        == "True\n"
    )
    with pytest.raises(RuntimeError, match="what went wrong"):
        update._run([sys.executable, "-c", "import sys; sys.exit('what went wrong')"])
    got = {}
    real = subprocess.run

    def run(cmd, **k):
        got.update(k)
        return real(cmd, **k)

    monkeypatch.setattr(update.subprocess, "run", run)
    update._run([sys.executable, "-c", "pass"])
    # no pipes: one that closes with the server would stop an installer in the middle of its work
    assert got["start_new_session"] is True and got["cwd"] == tmp_path
    assert got["stdout"] not in (None, subprocess.PIPE)
    assert got["stderr"] not in (None, subprocess.PIPE)


def _serve(monkeypatch, run):
    """wakesmith serve with uvicorn replaced by `run`; returns what happened."""
    import uvicorn
    from typer.testing import CliRunner

    from wakesmith.cli import app

    class Server:
        started = True
        should_exit = False

        def __init__(self, config):
            self.app = config.app

        def run(self):
            run(self)

    monkeypatch.setattr(uvicorn, "Server", Server)
    # the real Config points uvicorn's logging at the runner's stream, closed when it returns
    monkeypatch.setattr(uvicorn, "Config", lambda app, **k: type("C", (), {"app": app}))
    restarted = []
    monkeypatch.setattr(update, "restart", lambda: restarted.append(True))
    return CliRunner().invoke(app, ["serve", "--port", "1"]), restarted


def test_serve_restarts_after_an_update_and_quietly(monkeypatch, tmp_path):
    monkeypatch.setenv("WAKESMITH_HOME", str(tmp_path))
    log = logging.getLogger("uvicorn.error")
    level = log.level

    def run(server):
        u = server.app.state.updater
        u.restarting = True
        u.restart()
        assert server.should_exit and log.level == logging.CRITICAL

    try:
        r, restarted = _serve(monkeypatch, run)
    finally:
        log.setLevel(level)
    assert r.exit_code == 0 and restarted, r.output


def test_ctrl_c_during_a_restart_stops_the_server(monkeypatch, tmp_path):
    monkeypatch.setenv("WAKESMITH_HOME", str(tmp_path))

    def run(server):
        server.app.state.updater.restarting = True
        raise KeyboardInterrupt

    r, restarted = _serve(monkeypatch, run)
    assert r.exit_code == 0 and not restarted


def test_the_queue_doesnt_start_runs_during_an_update(net, monkeypatch):
    from wakesmith.web import server as srv

    kicks, slept = [], threading.Event()
    monkeypatch.setattr(jobs, "kick", lambda: kicks.append(True))
    monkeypatch.setattr(srv, "QUEUE_CHECK_S", 0.01)
    app = create_app()
    app.state.updater.job["stage"] = "installing"
    with TestClient(app, base_url="http://localhost"):
        slept.wait(0.15)
    assert not kicks


def test_a_runs_session_is_used_only_when_the_run_leads_it(monkeypatch):
    from wakesmith import sysstat

    me = __import__("os").getpid()
    pids = sysstat.job_pids({"run_pid": me, "own_session": False, "pid": 1})
    assert me in pids and 1 not in pids
    assert sys.executable  # the helper reads /proc on Linux only


def test_a_failed_update_leaves_the_server_working(net, client, monkeypatch):
    def run(cmd):
        if cmd[0] == "pip":
            raise RuntimeError("no matching dependency")
        return OLD + "\n"

    monkeypatch.setattr(update, "_run", run)
    client.post("/api/update", json={"version": NEW})
    assert wait(client.updater)["stage"] is None
    assert client.post("/api/update/check").status_code == 200


def test_a_run_begun_during_the_download_stops_the_install(net, client, monkeypatch):
    ran = commands(monkeypatch)
    answers = iter([None, None])
    monkeypatch.setattr(update, "busy", lambda: next(answers, "Training is running (x)"))
    client.post("/api/update", json={"version": NEW})
    job = wait(client.updater)
    assert "Training" in job["error"] and job["stage"] is None and not ran


def test_a_room_split_makes_it_wait(net, client, monkeypatch):
    from wakesmith import sessions

    ran = commands(monkeypatch)
    monkeypatch.setattr(sessions, "split_running", lambda: True)
    r = client.post("/api/update", json={"version": NEW})
    assert r.status_code == 409 and "split" in r.json()["detail"] and not ran


def test_split_running_goes_by_the_splits_own_process(monkeypatch, tmp_path):
    from wakesmith import sessions, sysstat

    monkeypatch.setattr(sessions, "home", lambda: tmp_path)
    monkeypatch.setattr(sessions, "folder", lambda sid: tmp_path / sid)
    assert not sessions.split_running()
    (tmp_path / "a").mkdir()
    me = os.getpid()
    f = tmp_path / "a" / "split.json"
    f.write_text(json.dumps({"status": "running", "pid": me, "ticks": sysstat.start_ticks(me)}))
    assert sessions.split_running()
    f.write_text(json.dumps({"status": "running", "pid": me, "ticks": 1}))  # a reused pid
    assert not sessions.split_running()
    f.write_text("[]")  # a folder it can't make sense of
    assert not sessions.split_running()


def _busy_lock(monkeypatch, tmp_path):
    monkeypatch.setenv("WAKESMITH_HOME", str(tmp_path))
    monkeypatch.setenv("WAKESMITH_PROJECTS", str(tmp_path / "projects"))
    return assets.hold_update_lock()


def test_a_run_refuses_to_start_during_an_update(tmp_path, monkeypatch):
    from typer.testing import CliRunner

    from wakesmith import pipeline, project
    from wakesmith.cli import app

    lock = _busy_lock(monkeypatch, tmp_path)
    p = project.create("hey jarvis")
    try:
        with pytest.raises(RuntimeError, match="being updated"):
            pipeline.run(p)
        r = CliRunner().invoke(app, ["train", p.name])  # the command line says so in a line
        assert r.exit_code != 0 and not isinstance(r.exception, RuntimeError)
    finally:
        os.close(lock)


def test_a_timed_device_sync_waits_for_an_update(tmp_path, monkeypatch):
    from typer.testing import CliRunner

    from wakesmith import clipsync
    from wakesmith.cli import app

    lock = _busy_lock(monkeypatch, tmp_path)
    monkeypatch.setattr(clipsync, "log", lambda m: None)
    try:
        assert clipsync.sync_all() == []
        r = CliRunner().invoke(app, ["clips", "sync-all"])
        assert r.exit_code == 0 and "being updated" in r.output and "no wake word" not in r.output
    finally:
        os.close(lock)


def test_queued_runs_wait_for_an_update(tmp_path, monkeypatch):
    lock = _busy_lock(monkeypatch, tmp_path)
    monkeypatch.setattr(jobs, "queue", lambda: pytest.fail("looked at the queue"))
    try:
        assert jobs.kick() == []
    finally:
        os.close(lock)


def test_where_locks_dont_work_nothing_looks_like_an_update(monkeypatch, tmp_path, net, client):
    import errno
    import fcntl

    ran = commands(monkeypatch)
    (assets.home()).mkdir(parents=True, exist_ok=True)

    def no_locks(fd, how):
        raise OSError(errno.ENOLCK, "no locks available")

    monkeypatch.setattr(fcntl, "flock", no_locks)
    client.post("/api/update", json={"version": NEW})
    assert "no locks available" in wait(client.updater)["error"] and not ran
    assert (assets.home() / "update.lock").exists() and not assets.update_underway()


def test_looks_at_the_lock_dont_take_each_other_for_an_update(monkeypatch, tmp_path):
    import fcntl

    monkeypatch.setenv("WAKESMITH_HOME", str(tmp_path))
    os.close(assets.hold_update_lock())
    other = os.open(tmp_path / "update.lock", os.O_RDWR)
    fcntl.flock(other, fcntl.LOCK_SH)  # another process, looking right now
    assert not assets.update_underway()
    assert assets.hold_update_lock() is None  # an update tries a few times, then says so
    os.close(other)
    os.close(assets.hold_update_lock())


@pytest.mark.parametrize(
    "st",
    [
        {"status": "running", "pid": "abc"},
        {"status": "running", "pid": 99999999999999999999},
        {"status": "running", "pid": -1},
        {"status": "running", "pid": True},
        {"status": "running", "started": None},
        {"status": "running", "started": "x"},
        {"status": "running", "pid": os.getpid(), "started": 1000.0},  # older version, before boot
    ],
)
def test_a_split_file_unlike_what_is_written_there_isnt_a_running_split(monkeypatch, tmp_path, st):
    from wakesmith import sessions

    monkeypatch.setattr(sessions, "home", lambda: tmp_path)
    monkeypatch.setattr(sessions, "folder", lambda sid: tmp_path / sid)
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "split.json").write_text(json.dumps(st))
    assert sessions.split_status("a")["status"] == "stopped"
    assert not sessions.split_running()


def test_a_split_just_started_counts_before_it_has_noted_itself(monkeypatch, tmp_path):
    from wakesmith import sessions

    monkeypatch.setattr(sessions, "home", lambda: tmp_path)
    monkeypatch.setattr(sessions, "folder", lambda sid: tmp_path / sid)
    (tmp_path / "a").mkdir()
    f = tmp_path / "a" / "split.json"
    f.write_text(json.dumps({"status": "running", "pid": None, "started": time.time()}))
    assert sessions.split_running()
    f.write_text("[]")
    assert sessions.split_status("a") == {"status": None}


def test_a_stage_with_an_odd_start_time_doesnt_stop_the_queue(monkeypatch, tmp_path):
    p = {"status": "running", "pid": os.getpid(), "started": "x"}
    assert jobs._stage_running(p) in (True, False)


@pytest.mark.parametrize(
    "args", [["test", "x"], ["room", "list"], ["clips", "sync", "x"], ["setup"]]
)
def test_commands_wait_for_an_update(tmp_path, monkeypatch, args):
    from typer.testing import CliRunner

    from wakesmith.cli import app

    lock = _busy_lock(monkeypatch, tmp_path)
    try:
        r = CliRunner().invoke(app, args)
        assert r.exit_code != 0 and not isinstance(r.exception, (RuntimeError, ValueError))
    finally:
        os.close(lock)


def test_doctor_isnt_held_up_by_an_update(tmp_path, monkeypatch):
    from wakesmith import cli

    lock = _busy_lock(monkeypatch, tmp_path)
    ctx = type("C", (), {"invoked_subcommand": "doctor"})()
    try:
        cli._every_command(ctx)
    finally:
        os.close(lock)


@pytest.mark.parametrize(
    "args", [["train", "--help"], ["room", "split", "--help"], ["clips", "sync", "--help"]]
)
def test_help_works_during_an_update(tmp_path, monkeypatch, args):
    from typer.testing import CliRunner

    from wakesmith.cli import app

    lock = _busy_lock(monkeypatch, tmp_path)
    monkeypatch.setattr(sys, "argv", ["wakesmith", *args])
    try:
        assert CliRunner().invoke(app, args).exit_code == 0
    finally:
        os.close(lock)


def test_the_messages_say_to_stop_the_server_when_setup_is_needed():
    assert "stop wakesmith serve" in assets.UPDATE_UNDERWAY


def test_a_split_that_says_it_started_at_infinity_goes_stale(monkeypatch, tmp_path):
    from wakesmith import sessions

    monkeypatch.setattr(sessions, "home", lambda: tmp_path)
    monkeypatch.setattr(sessions, "folder", lambda sid: tmp_path / sid)
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "split.json").write_text('{"status": "running", "started": Infinity}')
    assert not sessions.split_running()
    (tmp_path / "a" / "split.json").write_text(
        '{"status": "running", "started": 1' + "0" * 400 + "}"
    )
    assert not sessions.split_running()
