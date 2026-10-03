"""Static file server for the browser encoder.

    python tools/serve.py [port] [--isolate]

Plain `python -m http.server` works too. `--isolate` adds the COOP/COEP
headers that make the page cross-origin isolated, which lets onnxruntime-web
use WASM threads (SharedArrayBuffer). COEP is `credentialless` so the CDN
scripts and Hugging Face model files still load.
"""

import functools
import http.server
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


class Handler(http.server.SimpleHTTPRequestHandler):
    isolate = False

    def end_headers(self):
        if self.isolate:
            self.send_header("Cross-Origin-Opener-Policy", "same-origin")
            self.send_header("Cross-Origin-Embedder-Policy", "credentialless")
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def log_message(self, *args):
        pass


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    port = int(args[0]) if args else 8000
    Handler.isolate = "--isolate" in sys.argv
    handler = functools.partial(Handler, directory=str(ROOT))
    with http.server.ThreadingHTTPServer(("127.0.0.1", port), handler) as srv:
        print(f"serving {ROOT} on http://127.0.0.1:{port} (isolated={Handler.isolate})", flush=True)
        srv.serve_forever()


if __name__ == "__main__":
    main()
