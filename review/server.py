"""Review page for disputed clips: play each original recording and mark what was really said.
Serves only the clips listed in WORK/review/clips.json (made by build_list.py), by index; verdicts
go to WORK/review/decisions.json, which prep_real.py and mww_features.py apply to the training sets.
No authentication: it binds to localhost unless you pass --host (e.g. 0.0.0.0 to review from a phone
on your LAN). Don't expose it beyond a network you trust: it plays your recordings to anyone.
Usage: server.py [--host 127.0.0.1] [--port 8097]"""
import argparse
import json
import os
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "scripts"))
import recipe  # noqa: E402

LIST = json.load(open(recipe.WORK / "review/clips.json"))
CLIPS = LIST["clips"]
DEC_PATH = recipe.DECISIONS
DECISIONS = json.load(open(DEC_PATH)) if DEC_PATH.exists() else {}
LOCK = threading.Lock()
VERDICTS = {"wake", "not_wake", "unsure", "bad_audio"}


def save():
    fd, tmp = tempfile.mkstemp(dir=DEC_PATH.parent, suffix=".tmp")
    with os.fdopen(fd, "w") as f:
        json.dump(DECISIONS, f, indent=1)
    os.replace(tmp, DEC_PATH)


class H(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/":
            return self._send(200, (HERE / "index.html").read_bytes(), "text/html; charset=utf-8")
        if self.path == "/clips.json":
            with LOCK:
                body = json.dumps({**{k: LIST.get(k) for k in ("threshold", "summary", "title", "phrase")},
                                   "clips": CLIPS, "decisions": DECISIONS})
            return self._send(200, body.encode(), "application/json")
        if self.path.startswith("/audio/"):
            try:
                c = CLIPS[int(self.path[len("/audio/"):])]
            except (ValueError, IndexError):
                return self._send(404, b"no such clip", "text/plain")
            return self._send(200, recipe.path(c["src"], c["file"]).read_bytes(), "audio/wav")
        self._send(404, b"not found", "text/plain")

    def do_POST(self):
        if self.path != "/mark":
            return self._send(404, b"not found", "text/plain")
        try:
            d = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
            c = CLIPS[int(d["i"])]
            verdict = d.get("verdict")
            if verdict is not None and verdict not in VERDICTS:
                raise ValueError(verdict)
        except Exception as e:
            return self._send(400, str(e).encode(), "text/plain")
        with LOCK:
            if verdict is None:
                DECISIONS.pop(c["id"], None)
            else:
                DECISIONS[c["id"]] = {"verdict": verdict, "note": str(d.get("note", ""))[:500],
                                      "was_label": c["label"], "score": c["score"], "split": c["split"]}
            save()
        self._send(200, b"ok", "text/plain")

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8097)
    a = ap.parse_args()
    print(f"{len(CLIPS)} clips, {len(DECISIONS)} verdicts so far - http://{a.host}:{a.port}/")
    ThreadingHTTPServer((a.host, a.port), H).serve_forever()
