#!/usr/bin/env python3
"""Build the demo page data: encode + render the CC-licensed NIN clips, measure, write site/.

Usage: .venv/bin/python scripts/build_site.py [--force]
Preview: python -m http.server -d site   (not deployed; see site/README.md)"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from soundcode import site  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--force", action="store_true", help="re-encode and re-render every song")
args = ap.parse_args()
data = site.build(ROOT, ROOT / "site", ROOT / "out" / "site", force=args.force)
t = data["totals"]
print(f"{len(data['songs'])} songs: WAV {t['wav']/1e6:.1f} MB -> .sc {t['sc']/1e3:.1f} KB "
      f"(gz {t['sc_gz']/1e3:.1f} KB, x{round(t['wav']/t['sc_gz'])}); borrowed audio {t['borrowed']/1e6:.2f} MB")
