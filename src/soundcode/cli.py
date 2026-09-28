"""soundcode CLI.

Implemented so far (milestone M1):
    soundcode check  <file.sc>            parse, validate, report
    soundcode render <file.sc> -o out.wav mock render (Path B stage 1)
    soundcode separate <audio> [-o dir]   split into vocal + instrument stems

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
    p_render.add_argument("--engine", choices=("sf2", "mock"), default="sf2",
                          help="sf2: sampled instruments (default); mock: crude synth")
    p_render.add_argument("--voice-ref", default=None,
                          help="the original singer (wav); default out/stems/<source>/lead_vocals.wav")
    p_render.add_argument("--with-vocals", action="store_true",
                          help="include vocal streams (sf2 engine leaves them out by default)")

    p_sep = sub.add_parser("separate", help="split audio into vocal and instrument stems")
    p_sep.add_argument("file")
    p_sep.add_argument("-o", "--out", default=None,
                       help="output folder (default out/stems/<name>)")

    p_cmp = sub.add_parser("compare", help="score a render against the original, per stem")
    p_cmp.add_argument("original")
    p_cmp.add_argument("render", help=".sc file (rendered per stream) or a rendered .wav")
    p_cmp.add_argument("-o", "--out", default=None, help="default out/compare/<name>")
    p_cmp.add_argument("--engine", choices=("sf2", "mock"), default="sf2")
    p_cmp.add_argument("--with-vocals", action="store_true",
                       help="score the sung lead vocal (DiffSinger -> Seed-VC)")
    p_cmp.add_argument("--voice-ref", default=None)

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
            doc = parse_file(args.file)
            if args.engine == "mock":
                from .render import render_to_file
                suffix = ".mock.wav"
            else:
                from . import render_sf

                def render_to_file(doc, out, sr):
                    return render_sf.render_to_file(
                        doc, out, sr, with_vocals=args.with_vocals,
                        voice_ref=Path(args.voice_ref) if args.voice_ref else None)
                suffix = ".render.wav"
            out = args.out or str(Path(args.file).with_suffix(suffix))
            try:
                count, secs = render_to_file(doc, out, args.sr)
            except Exception as exc:
                from .render_sf import SoundFontError
                from .sing_score import SingError
                if isinstance(exc, (SoundFontError, SingError)):
                    print(f"render failed: {exc}", file=sys.stderr)
                    return 2
                raise
            print(f"rendered {count} events -> {out}  ({secs:.2f}s @ "
                  f"{args.sr or doc.sample_rate} Hz, engine {args.engine})")
            return 0

        if args.cmd == "separate":
            from . import separate as sep

            out = args.out or str(sep.default_out_dir(args.file))
            try:
                res = sep.separate(args.file, out, backend=sep.AudioSeparatorBackend())
            except sep.SeparationError as exc:
                print(f"separation failed: {exc}", file=sys.stderr)
                return 2
            for name in (*sep.STEMS, "vocals", "instrumental", "residual"):
                db = res.levels[name]
                print(f"  {name:<15} {db:7.1f} dBFS" if db > -200 else f"  {name:<15}  silent")
            r = res.report
            verdict = "OK" if r.ok else "FAILED"
            print(f"sum check: {verdict}  (level diff {r.level_diff_db:+.2f} dB, "
                  f"residual {r.residual_db:.1f} dB)  -> {out}")
            for w in res.warnings:
                print(f"  warn: {w}")
            return 0 if r.ok else 1

        if args.cmd == "compare":
            from . import compare as cmp

            out = args.out or str(Path("out") / "compare" / Path(args.original).stem)
            from .sing_score import SingError
            try:
                rep = cmp.run(args.original, args.render, out, engine=args.engine,
                              with_vocals=args.with_vocals,
                              voice_ref=Path(args.voice_ref) if args.voice_ref else None)
            except SingError as exc:
                print(f"compare failed: {exc}", file=sys.stderr)
                return 2
            f = cmp._fmt
            print(f"{'stem':<15}{'orig dB':>8}{'rend dB':>8}{'Δ dB':>7}{'noteF1':>8}"
                  f"{'anyOct':>8}{'chroma':>8}{'onsetF1':>8}{'energy':>8}")
            for s, r in rep["stems"].items():
                print(f"{s:<15}{f(r['level_orig_db'], '.1f'):>8}{f(r['level_render_db'], '.1f'):>8}"
                      f"{f(r['level_diff_db'], '+.1f'):>7}{f(r['notes_f1']):>8}"
                      f"{f(r['notes_f1_octave']):>8}{f(r['chroma']):>8}"
                      f"{f(r['onset_f1']):>8}{f(r['energy_corr']):>8}")
            lv = rep["stems"].get("lead_vocals", {})
            if "pitch_cents" in lv:
                print(f"lead_vocals sung: pitch error {f(lv['pitch_cents'], '.0f')} c, "
                      f"voice similarity {f(lv['voice_sim'])}")
            print(f"report: {rep['report']}")
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
