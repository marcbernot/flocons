"""Screenshot the figures of every _preview/*.html in headless Chromium, one PNG per figure.

For an unattended look at the figures of `pixi run preview`. Waits until the page has stopped
loading and then `--wait` seconds more, so that Bokeh has drawn every figure: a screenshot
taken too early (headless Firefox `--screenshot`) shows figures blank that are fine.

Writes _preview/<notebook>-cellNN.png for each figure output (HTML or image) of a code cell,
cropped to it; NN is the cell number that `test-nb` and _preview/<notebook>.txt use, and a
second figure in the same cell gets -cellNN-2.png. Outputs smaller than `--min` px in height
(the HoloViews / Bokeh logos of `hv.extension`, empty script outputs) are skipped. Read
_preview/<notebook>.txt first: it has the printed numbers and the title of each figure.
A figure still blank in its PNG is really blank (causes: PROJECT.md, "Look at the figures").

With `--pages`, the whole page is taken instead, in pieces `--height` px tall:
_preview/<notebook>-01.png, -02.png, ... (text and code included).

Usage: python lite/screenshot.py _preview [--wait 5] [--min 40] [--pages [--height 1500]]
"""

import argparse
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

FIGURE_TYPES = ("text/html", "image/svg+xml", "image/png", "image/jpeg")
# every cell of the page, in order: the index + 1 is the cell number of the notebook
CELLS = ".jp-Cell"
FIGURES = ", ".join(f'.jp-OutputArea-output[data-mime-type="{t}"]' for t in FIGURE_TYPES)


def clear(folder: Path, stem: str) -> None:
    for pattern in (f"{stem}-[0-9][0-9].png", f"{stem}-cell*.png"):
        for old in folder.glob(pattern):
            old.unlink()


def figures(page, folder: Path, stem: str, min_height: int) -> list[str]:
    done = []
    for i, cell in enumerate(page.locator(CELLS).all()):
        if "jp-CodeCell" not in (cell.get_attribute("class") or ""):
            continue
        k = 0
        for out in cell.locator(FIGURES).all():
            box = out.bounding_box()
            if box is None or box["height"] < min_height:
                continue
            k += 1
            name = f"{stem}-cell{i + 1:02d}{'' if k == 1 else f'-{k}'}.png"
            out.screenshot(path=folder / name)
            done.append(f"cell {i + 1}: {box['width']:.0f} x {box['height']:.0f} -> {name}")
    return done


def pages(page, folder: Path, stem: str, width: int, height: int) -> list[str]:
    total = page.evaluate("() => document.documentElement.scrollHeight")
    done = []
    for top in range(0, total, height):
        name = f"{stem}-{len(done) + 1:02d}.png"
        clip = {"x": 0, "y": top, "width": width, "height": min(height, total - top)}
        page.screenshot(path=folder / name, full_page=True, clip=clip)
        done.append(name)
    return done


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("folder", type=Path, help="folder of the HTML files (_preview)")
    parser.add_argument("--width", type=int, default=1100, help="width of the browser window")
    parser.add_argument("--wait", type=float, default=5, help="seconds after the page loaded")
    parser.add_argument("--min", type=int, default=40, help="smaller outputs are skipped (px)")
    parser.add_argument("--pages", action="store_true", help="whole pages instead of figures")
    parser.add_argument("--height", type=int, default=1500, help="height of a page piece")
    args = parser.parse_args()

    htmls = sorted(args.folder.glob("*.html"))
    if not htmls:
        print(f"no HTML in {args.folder}: run `pixi run preview` first")
        return 1
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        for html in htmls:
            clear(args.folder, html.stem)
            page = browser.new_page(viewport={"width": args.width, "height": args.height})
            page.goto(html.resolve().as_uri(), wait_until="networkidle")
            page.wait_for_timeout(args.wait * 1000)
            if args.pages:
                done = pages(page, args.folder, html.stem, args.width, args.height)
                print(f"{html}: {len(done)} pieces of {args.height} px -> {html.stem}-NN.png")
            else:
                done = figures(page, args.folder, html.stem, args.min)
                print(f"{html}: {len(done)} figures (text and titles: {html.stem}.txt)")
                for line in done:
                    print(f"  {line}")
            page.close()
        browser.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
