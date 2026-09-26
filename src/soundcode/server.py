"""Local A/B listening server.

Comparing renders is the whole job here — "is strength 0.6 following the mock
more closely than 0.8" is a question about differences, not about any one file.
So the UI is built around a single shared transport with instant A/B switching:
position is preserved when you swap, which is the only way small differences
become audible.

Zero dependencies beyond the stdlib. Range requests are implemented because
browsers will not seek in an audio element without them.

    soundcode serve            # http://127.0.0.1:8720
"""

from __future__ import annotations

import json
import os
import re
import socketserver
import subprocess
import sys
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import unquote, urlparse

WEB = Path(__file__).parent / "web"
_RANGE_RE = re.compile(r"bytes=(\d*)-(\d*)")


def _project_root() -> Path:
    return Path(os.environ.get("SOUNDCODE_ROOT", Path.cwd())).resolve()


def _discover_tracks() -> list[dict]:
    """Everything under out/ that a browser can play, newest first."""
    root = _project_root()
    out = root / "out"
    tracks: list[dict] = []
    if not out.is_dir():
        return tracks
    for path in sorted(out.rglob("*")):
        if path.suffix.lower() not in (".wav", ".mp3", ".flac", ".ogg"):
            continue
        if "compare" in path.relative_to(out).parts:
            continue              # per-stem compare files live in the report
        if "_work" in path.relative_to(out).parts:
            continue              # separation intermediates, not for listening
        rel = path.relative_to(root)
        stat = path.stat()
        label = str(path.relative_to(out).with_suffix(""))
        if label.startswith("ref/") or "reference" in label:
            kind = "ref"          # the original recording we are aiming at
        elif label.startswith("stems/"):
            kind = "stem"
        elif ".render" in label or "render-" in label:
            kind = "render"
        elif "mock" in label:
            kind = "mock"
        elif "cover" in label:
            kind = "cover"
        else:
            kind = "other"
        strength = None
        if m := re.search(r"strength[_-]?([\d.]+)", label):
            strength = float(m.group(1))
        tracks.append({
            "url": "/audio/" + str(rel).replace(os.sep, "/"),
            "label": label,
            "kind": kind,
            "strength": strength,
            "bytes": stat.st_size,
            "mtime": stat.st_mtime,
        })
    order = {"ref": 0, "render": 1, "mock": 2, "cover": 3, "stem": 4, "other": 5}
    tracks.sort(key=lambda t: (order.get(t["kind"], 9), t["strength"] or 0, t["label"]))
    return tracks


def _sc_sources() -> list[dict]:
    """Hand-authored examples plus anything the encoder has produced."""
    root = _project_root()
    paths = sorted((root / "examples").glob("*.sc"))
    paths += sorted((root / "out" / "sc").glob("*.sc"))
    return [
        {"name": str(p.relative_to(root)), "text": p.read_text(encoding="utf-8")}
        for p in paths
    ]


def _compare_file(rel: str) -> Path | None:
    base = (_project_root() / "out" / "compare").resolve()
    target = (base / rel).resolve()
    if base not in target.parents or not target.is_file():
        return None
    return target


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt: str, *args) -> None:      # quieter console
        if "/audio/" not in (args[0] if args else ""):
            sys.stderr.write("  %s\n" % (fmt % args))

    # -- helpers -------------------------------------------------------------

    def _send_json(self, payload) -> None:
        body = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, path: Path, ctype: str) -> None:
        if not path.is_file():
            self.send_error(404)
            return
        size = path.stat().st_size
        rng = self.headers.get("Range")
        start, end = 0, size - 1
        partial = False
        if rng and (m := _RANGE_RE.match(rng)):
            g1, g2 = m.groups()
            if g1:
                start = int(g1)
                end = int(g2) if g2 else size - 1
            elif g2:                                  # suffix range
                start = max(size - int(g2), 0)
            partial = True
        start = max(0, min(start, size - 1))
        end = max(start, min(end, size - 1))
        length = end - start + 1

        self.send_response(206 if partial else 200)
        self.send_header("Content-Type", ctype)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(length))
        if partial:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        with path.open("rb") as fh:
            fh.seek(start)
            remaining = length
            while remaining > 0:
                chunk = fh.read(min(262144, remaining))
                if not chunk:
                    break
                try:
                    self.wfile.write(chunk)
                except (BrokenPipeError, ConnectionResetError):
                    return                            # browser seeked away
                remaining -= len(chunk)

    # -- routes --------------------------------------------------------------

    def do_GET(self) -> None:                          # noqa: N802
        route = unquote(urlparse(self.path).path)

        if route in ("/", "/index.html"):
            self._send_file(WEB / "index.html", "text/html; charset=utf-8")
        elif route == "/api/tracks":
            self._send_json({"tracks": _discover_tracks(), "root": str(_project_root())})
        elif route == "/api/sources":
            self._send_json({"sources": _sc_sources()})
        elif route.startswith("/compare/"):
            target = _compare_file(route[len("/compare/"):])
            if target is None:
                self.send_error(404)
                return
            ctype = {".html": "text/html; charset=utf-8", ".png": "image/png",
                     ".json": "application/json", ".wav": "audio/wav"}.get(target.suffix,
                                                                            "application/octet-stream")
            self._send_file(target, ctype)
        elif route.startswith("/audio/"):
            rel = route[len("/audio/"):]
            target = (_project_root() / rel).resolve()
            if not str(target).startswith(str(_project_root())):
                self.send_error(403)                   # no traversal out of the repo
                return
            ctype = {
                ".wav": "audio/wav", ".mp3": "audio/mpeg",
                ".flac": "audio/flac", ".ogg": "audio/ogg",
            }.get(target.suffix.lower(), "application/octet-stream")
            self._send_file(target, ctype)
        else:
            self.send_error(404)

    def _read_upload(self) -> tuple[str, bytes] | None:
        """Minimal multipart/form-data reader — one file field, no deps."""
        ctype = self.headers.get("Content-Type", "")
        if "multipart/form-data" not in ctype or "boundary=" not in ctype:
            return None
        boundary = ctype.split("boundary=")[1].strip().strip('"').encode()
        body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        for part in body.split(b"--" + boundary):
            head, _, data = part.partition(b"\r\n\r\n")
            if b"filename=" not in head:
                continue
            name = head.split(b"filename=")[1].split(b'"')[1].decode(errors="replace")
            if not name:
                continue
            return Path(name).name, data.rstrip(b"\r\n-")
        return None

    def do_POST(self) -> None:                         # noqa: N802
        route = unquote(urlparse(self.path).path)
        root = _project_root()

        if route == "/api/upload":
            got = self._read_upload()
            if not got:
                self._send_json({"ok": False, "error": "no file in request"})
                return
            name, data = got
            dest = root / "audio" / "uploads" / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(data)
            self._send_json({"ok": True, "path": str(dest.relative_to(root)),
                             "bytes": len(data)})
            return

        if route == "/api/pipeline":
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length) or b"{}")
            src = body.get("path")
            if not src:
                self._send_json({"ok": False, "error": "no path"})
                return
            stem = Path(src).stem
            sc = root / "out" / "sc" / f"{stem}.sc"
            mock = root / "out" / f"{stem}.mock.wav"
            steps = []
            for label, cmd in (
                ("encode", ["encode", src, "-o", str(sc)]),
                ("render", ["render", str(sc), "-o", str(mock)]),
            ):
                proc = subprocess.run(
                    [sys.executable, "-m", "soundcode.cli", *cmd],
                    cwd=root, capture_output=True, text=True,
                    env={**os.environ, "PYTHONPATH": str(root / "src")},
                )
                steps.append({"step": label, "ok": proc.returncode == 0,
                              "out": (proc.stdout + proc.stderr).strip()[-4000:]})
                if proc.returncode != 0:
                    break
            self._send_json({"ok": all(s["ok"] for s in steps), "steps": steps,
                             "sc": str(sc.relative_to(root)) if sc.exists() else None})
            return

        if route == "/api/render":
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length) or b"{}")
            src = body.get("source", "examples/signal-lost.v3.sc")
            dest = root / "out" / (Path(src).stem + ".mock.wav")
            proc = subprocess.run(
                [sys.executable, "-m", "soundcode.cli", "render", src, "-o", str(dest)],
                cwd=root, capture_output=True, text=True,
                env={**os.environ, "PYTHONPATH": str(root / "src")},
            )
            self._send_json({
                "ok": proc.returncode == 0,
                "stdout": proc.stdout.strip(),
                "stderr": proc.stderr.strip(),
            })
            return

        self.send_error(404)


class Server(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


def serve(host: str = "127.0.0.1", port: int = 8720) -> None:
    os.environ.setdefault("SOUNDCODE_ROOT", str(Path.cwd()))
    with Server((host, port), Handler) as httpd:
        print(f"soundcode listening on http://{host}:{port}")
        print(f"  serving audio from {_project_root() / 'out'}")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nstopped")
