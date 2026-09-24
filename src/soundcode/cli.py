"""soundcode CLI.

Implemented so far (milestone M1):
    soundcode check  <file.sc>            parse, validate, report
    soundcode render <file.sc> -o out.wav mock render (Path B stage 1)

Still to come: `encode` (audio -> .sc) and `generate` (.sc -> mock -> Remix).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .expand import build_grid, expand
from .parser import ParseError, parse_file


def _summarise(path: str) -> int:
    doc = parse_file(path)
    notes = expand(doc)
    grid = build_grid(doc)

    title = doc.header.get("title", "(untitled)")
    print(f"{title}  —  sc {doc.pragmas.get('sc', '?')}  "
          f"profile {doc.pragmas.get('profile', '?')}  "
          f"residual {doc.pragmas.get('residual', 'none')}")
    print(f"  duration {doc.duration:.3f}s   sr {doc.sample_rate}   "
          f"streams {len(doc.streams)}   sounding events {len(notes)}")

    if doc.stream("grid"):
        bpm = grid.tempo[0][1] if grid.tempo else 0
        print(f"  grid: {len(grid.tempo)} tempo pts (start {bpm:.2f} bpm), "
              f"{len(grid.anchors)} anchors, meter {grid.meter[0][1]}/{grid.meter[0][2]}")

    print()
    for s in doc.streams:
        counts = []
        if s.events:
            counts.append(f"{len(s.events)} ev")
        if s.patterns:
            counts.append(f"{len(s.patterns)} pat")
        if s.bindings:
            counts.append(f"{len(s.bindings)} bind")
        if s.statements:
            counts.append(f"{len(s.statements)} stmt")
        sounding = sum(1 for n in notes if n.stream == s.name)
        if sounding:
            counts.append(f"-> {sounding} sounding")
        conf = s.meta.get("conf")
        flag = "" if conf in (None, "1.00") else f"  conf={conf}"
        print(f"  :{s.name:<16} {s.fields.get('inst', ''):<14} "
              f"{', '.join(counts)}{flag}")
        for w in s.warns:
            print(f"      warn: {w}")

    lossy = [(s.name, b) for s in doc.streams for b in s.bindings
             if b.similarity is not None]
    if lossy:
        print("\n  lossy pattern bindings (per-instance nuance discarded):")
        for name, b in lossy:
            print(f"    :{name} bars {b.first}-{b.last} ~{b.similarity:.2f}")

    unresolved = [n for n in notes if n.cents is None and n.voice is None]
    if unresolved:
        print(f"\n  WARNING: {len(unresolved)} events failed to resolve")
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="soundcode")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_check = sub.add_parser("check", help="parse and report on a .sc file")
    p_check.add_argument("file")

    p_render = sub.add_parser("render", help="mock-render a .sc file to audio")
    p_render.add_argument("file")
    p_render.add_argument("-o", "--out", default=None)
    p_render.add_argument("--sr", type=int, default=None)

    p_encode = sub.add_parser("encode", help="analyse audio and write a .sc file")
    p_encode.add_argument("file")
    p_encode.add_argument("-o", "--out", default=None)
    p_encode.add_argument("--workdir", default=None)
    p_encode.add_argument("--title", default=None)

    p_serve = sub.add_parser("serve", help="local A/B listening server")
    p_serve.add_argument("--host", default="127.0.0.1")
    p_serve.add_argument("--port", type=int, default=8720)

    args = ap.parse_args(argv)

    try:
        if args.cmd == "check":
            return _summarise(args.file)

        if args.cmd == "render":
            from .render import render_to_file

            doc = parse_file(args.file)
            out = args.out or str(Path(args.file).with_suffix(".mock.wav"))
            count, secs = render_to_file(doc, out, args.sr)
            print(f"rendered {count} events -> {out}  ({secs:.2f}s @ "
                  f"{args.sr or doc.sample_rate} Hz)")
            return 0

        if args.cmd == "encode":
            from .encode import encode

            out = args.out or str(Path(args.file).with_suffix(".sc"))
            encode(args.file, out, args.workdir, args.title)
            return 0

        if args.cmd == "serve":
            from .server import serve

            serve(args.host, args.port)
            return 0
    except ParseError as exc:
        print(f"parse error: {exc}", file=sys.stderr)
        return 2
    except FileNotFoundError as exc:
        print(f"no such file: {exc.filename}", file=sys.stderr)
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
