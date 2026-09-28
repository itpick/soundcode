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
    p_render.add_argument("--no-fx", action="store_true",
                          help="skip each part's production profile (fx lines), for A/B")
    p_render.add_argument("--singer", choices=("diffsinger", "soulx"), default=None,
                          help="who sings --with-vocals (default: sing.DEFAULT_SINGER)")
    p_render.add_argument("--voice-ref", default=None,
                          help="the original singer (wav); default out/stems/<source>/lead_vocals.wav")
    p_render.add_argument("--with-vocals", action="store_true",
                          help="include vocal streams (sf2 engine leaves them out by default)")
    p_render.add_argument("--parts", metavar="DIR",
                          help="also write each part (vocals, keys, bass, drums, …) as DIR/<part>.wav")

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
    p_cmp.add_argument("--no-fx", action="store_true", help="render without fx lines (A/B)")
    p_cmp.add_argument("--singer", choices=("diffsinger", "soulx"), default=None)

    p_encode = sub.add_parser("encode", help="analyse audio and write a .sc file")
    p_encode.add_argument("file")
    p_encode.add_argument("-o", "--out", default=None)
    p_encode.add_argument("--workdir", default=None)
    p_encode.add_argument("--title", default=None)
    p_encode.add_argument("--artist", default=None, help="for the published-lyrics lookup")
    p_encode.add_argument("--offset", type=float, default=None,
                          help="where this clip starts in the full song, seconds (default: found from the lyrics)")

    p_score = sub.add_parser("score", help="score a rebuild against the original, per part/section")
    p_score.add_argument("original")
    p_score.add_argument("sc")
    p_score.add_argument("--stems", default=None, help="default out/stems/<original stem>")
    p_score.add_argument("--out", default=None, help="default out/score/<sc stem>")

    p_bench = sub.add_parser("bench", help="run the benchmark set and update the run history")
    p_bench.add_argument("--tier", choices=("A", "B", "C", "all"), default="A")
    p_bench.add_argument("--label", default="")
    p_bench.add_argument("--force", action="store_true")
    p_bench.add_argument("--calibrate", action="store_true",
                         help="re-measure the anchors (not yet implemented)")

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
                if args.parts:
                    print("render: --parts needs the sf2 engine", file=sys.stderr)
                    return 2
                from .render import render_to_file
                suffix = ".mock.wav"
            elif args.parts:
                from . import render_sf

                def render_to_file(doc, out, sr):
                    import numpy as np
                    import soundfile as sf

                    sr = sr or doc.sample_rate
                    audio, parts = render_sf.render_parts(
                        doc, sr, with_vocals=args.with_vocals,
                        voice_ref=Path(args.voice_ref) if args.voice_ref else None,
                        no_fx=args.no_fx, singer=args.singer)
                    sf.write(out, audio, sr, subtype="PCM_16")
                    parts_dir = Path(args.parts)
                    parts_dir.mkdir(parents=True, exist_ok=True)
                    for key, y in parts.items():
                        if float(np.abs(y).max()) > 1e-4:
                            sf.write(str(parts_dir / f"{key}.wav"), y, sr, subtype="PCM_16")
                    return len(expand(doc)), audio.shape[0] / sr
                suffix = ".render.wav"
            else:
                from . import render_sf

                def render_to_file(doc, out, sr):
                    return render_sf.render_to_file(
                        doc, out, sr, with_vocals=args.with_vocals,
                        voice_ref=Path(args.voice_ref) if args.voice_ref else None,
                        no_fx=args.no_fx, singer=args.singer)
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
                              voice_ref=Path(args.voice_ref) if args.voice_ref else None,
                              no_fx=args.no_fx, singer=args.singer)
            except SingError as exc:
                print(f"compare failed: {exc}", file=sys.stderr)
                return 2
            f = cmp._fmt
            print(f"{'stem':<15}{'orig dB':>8}{'rend dB':>8}{'Δ dB':>7}{'noteF1':>8}"
                  f"{'anyOct':>8}{'chroma':>8}{'onsetF1':>8}{'energy':>8}{'specdB':>8}")
            for s, r in rep["stems"].items():
                print(f"{s:<15}{f(r['level_orig_db'], '.1f'):>8}{f(r['level_render_db'], '.1f'):>8}"
                      f"{f(r['level_diff_db'], '+.1f'):>7}{f(r['notes_f1']):>8}"
                      f"{f(r['notes_f1_octave']):>8}{f(r['chroma']):>8}"
                      f"{f(r['onset_f1']):>8}{f(r['energy_corr']):>8}"
                      f"{f(r.get('spectral_db'), '.1f'):>8}")
            lyr = rep.get("lyrics") or {}
            if lyr.get("lyric_wer") is not None or lyr.get("sung_wer") is not None:
                print(f"lyrics: wer vs published {f(lyr.get('lyric_wer'))}, "
                      f"sung intelligibility wer {f(lyr.get('sung_wer'))}")
            lv = rep["stems"].get("lead_vocals", {})
            if "pitch_cents" in lv:
                print(f"lead_vocals sung: pitch error {f(lv['pitch_cents'], '.0f')} c, "
                      f"voice similarity {f(lv['voice_sim'])}")
            print(f"report: {rep['report']}")
            return 0

        if args.cmd == "encode":
            from .encode import encode

            out = args.out or str(Path(args.file).with_suffix(".sc"))
            encode(args.file, out, args.workdir, args.title, args.artist, args.offset)
            return 0

        if args.cmd == "score":
            from . import render_sf
            from .score import report as score_report
            from .score import scorer

            sc_path, original = Path(args.sc), Path(args.original)
            stems_dir = Path(args.stems) if args.stems else Path("out") / "stems" / original.stem
            out_dir = Path(args.out) if args.out else Path("out") / "score" / sc_path.stem
            parts_dir, rebuild = out_dir / "parts", out_dir / "rebuild.wav"

            doc = parse_file(str(sc_path))
            cached = (rebuild.exists() and parts_dir.is_dir() and any(parts_dir.glob("*.wav"))
                     and rebuild.stat().st_mtime >= sc_path.stat().st_mtime)
            if not cached:
                import numpy as np
                import soundfile as sf

                from .sing_score import SingError
                try:
                    sr = doc.sample_rate
                    audio, parts = render_sf.render_parts(doc, sr, with_vocals=True)
                except (render_sf.SoundFontError, SingError) as exc:
                    print(f"score failed: {exc}", file=sys.stderr)
                    return 2
                out_dir.mkdir(parents=True, exist_ok=True)
                sf.write(str(rebuild), audio, sr, subtype="PCM_16")
                parts_dir.mkdir(parents=True, exist_ok=True)
                for key, y in parts.items():
                    if float(np.abs(y).max()) > 1e-4:
                        sf.write(str(parts_dir / f"{key}.wav"), y, sr, subtype="PCM_16")

            cache_dir = (Path("out") / "bench" / "cache").resolve()
            result = scorer.score_song(original, stems_dir, sc_path, parts_dir, rebuild, out_dir,
                                       cache_dir=cache_dir)
            print(score_report.table(result))
            report_path = score_report.html(result, out_dir)
            print(f"report: {report_path}")
            return 0

        if args.cmd == "bench":
            if args.calibrate:
                print("not yet")
                return 0
            from .score import bench

            res = bench.run_bench(args.tier, args.label, Path("."), force=args.force)
            print(f"run: {res['run_dir']}")
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
