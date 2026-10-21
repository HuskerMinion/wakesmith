"""A self-signed certificate for `wakesmith serve --https` (browsers only allow the microphone on
localhost or HTTPS). Made once with the openssl command, for localhost, this machine's name and its
current LAN addresses; delete the folder to make a new one after the addresses change."""

from __future__ import annotations

import shutil
import socket
import subprocess
from pathlib import Path

from . import assets


def _addresses() -> list[str]:
    ips = {"127.0.0.1"}
    try:
        out = subprocess.run(
            ["hostname", "-I"], capture_output=True, text=True, timeout=5, check=False
        ).stdout
        ips |= {ip for ip in out.split() if ":" not in ip}
    except (OSError, subprocess.TimeoutExpired):
        pass
    return sorted(ips)


def certificate() -> tuple[Path, Path]:
    d = assets.home() / "tls"
    cert, key = d / "cert.pem", d / "key.pem"
    if cert.exists() and key.exists():
        return cert, key
    if not shutil.which("openssl"):
        raise SystemExit("--https needs the openssl command (sudo apt install openssl)")
    d.mkdir(parents=True, exist_ok=True)
    names = ["DNS:localhost", f"DNS:{socket.gethostname()}"] + [f"IP:{ip}" for ip in _addresses()]
    subprocess.run(
        [
            "openssl",
            "req",
            "-x509",
            "-newkey",
            "rsa:2048",
            "-nodes",
            "-days",
            "825",
            "-keyout",
            str(key),
            "-out",
            str(cert),
            "-subj",
            "/CN=wakesmith",
            "-addext",
            "subjectAltName=" + ",".join(names),
        ],
        check=True,
        capture_output=True,
    )
    key.chmod(0o600)
    return cert, key
