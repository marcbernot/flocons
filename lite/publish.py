"""Copy the files the notebooks write that are published on purpose, or check the copies.

Notebooks write to notebooks/out/, which is in neither git nor the site. A file meant for the
README or for readers (a figure, an SVG) is listed in pyproject.toml:

    [tool.lite.publish]
    "docs/figure.svg" = "notebooks/out/figure.svg"

`pixi run publish` runs the notebooks, then copies each file; commit the copies.
`pixi run publish-check` (part of `check`) runs the notebooks, then fails when a copy is missing
or differs from what the notebook wrote, so a committed copy cannot drift from the code.
Only files the notebooks write identically every run belong here (an SVG, a CSV; not a Bokeh
HTML page, whose ids change every run).

Usage: python lite/publish.py [--check]
"""

import argparse
import shutil
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="compare only, copy nothing")
    args = parser.parse_args()
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text())
    files = pyproject.get("tool", {}).get("lite", {}).get("publish", {})
    if not files:
        print("publish: nothing listed in [tool.lite.publish]")
        return 0
    bad = 0
    for dest, source in files.items():
        src, dst = ROOT / source, ROOT / dest
        if not src.exists():
            print(f"FAIL {source}: not written by the notebooks")
            bad += 1
        elif args.check:
            if not dst.exists() or dst.read_bytes() != src.read_bytes():
                print(f"FAIL {dest}: out of date with {source} (pixi run publish, then commit)")
                bad += 1
            else:
                print(f"ok   {dest}")
        else:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dst)
            print(f"copied {source} -> {dest}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
