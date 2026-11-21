"""Execute notebooks locally (outputs are not saved) and report the slow cells.

The browser runs the same code several times slower than local Python, and has to download
every package first. A cell that takes a few seconds here can take a minute in JupyterLite, and
`test-lite` stops at its timeout. Cells slower than `--slow` seconds are listed with a warning,
except cells with an import line (their time is mostly first-import cost, paid differently in the
browser: keep imports in their own cells so the rest is measured). A failing cell fails the
run, as `jupyter execute` does; a slow cell only warns.

With `--html DIR` the executed notebook is also written to DIR/<notebook>.html (what
`jupyter nbconvert --execute --to html` writes), so one run serves `test-nb`, `publish`,
`publish-check`, `preview` and `shot-preview`. pixi caches the run (`inputs` of the `test-nb`
task): the notebooks run again only when they, `src/`, `pyproject.toml` or the environment change.

It also writes DIR/<notebook>.txt: the text every code cell printed, its errors, and one line
per figure with the titles found in the cell (`[figure: "..."]`). Read it before the figures:
a wrong number (a level, a count, an error) shows there first, for a fraction of the cost.

Usage: python lite/time_notebooks.py notebooks/*.ipynb [--slow 2] [--html _preview]
"""

import argparse
import re
import sys
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError


def imports(source: str) -> bool:
    return any(ln.startswith(("import ", "from ")) for ln in source.splitlines())


FIGURE_TYPES = ("text/html", "image/svg+xml", "image/png", "image/jpeg")
TITLE = re.compile(r"""title=(["'])(.+?)\1""")


def digest(nb) -> str:
    """The text outputs of every code cell, figures as one line each, cells numbered from 1."""
    out = []
    for i, cell in enumerate(nb.cells):
        if cell.cell_type != "code" or not cell.get("outputs"):
            continue
        lines = cell.source.strip().splitlines()
        out.append(f"## cell {i + 1}: {lines[0][:80] if lines else ''}")
        titles = [m.group(2) for m in TITLE.finditer(cell.source)]
        for o in cell.outputs:
            if o.output_type == "stream":
                out.append(o.text.rstrip("\n"))
            elif o.output_type == "error":
                out.append(f"ERROR {o.ename}: {o.evalue}")
            else:
                data = o.get("data", {})
                kind = next((t for t in FIGURE_TYPES if t in data), None)
                if kind is not None:
                    what = ", ".join(f'"{t}"' for t in titles) if titles else kind
                    out.append(f"[figure: {what}]")
                elif "text/plain" in data:
                    out.append(data["text/plain"].rstrip("\n"))
        out.append("")
    return "\n".join(out)


def run(path: Path, slow: float, html: Path | None) -> bool:
    nb = nbformat.read(path, as_version=4)
    times: dict[int, float] = {}
    started: dict[int, float] = {}

    def on_start(cell, cell_index, **_):
        started[cell_index] = time.monotonic()

    def on_done(cell, cell_index, **_):
        times[cell_index] = time.monotonic() - started[cell_index]

    client = NotebookClient(
        nb,
        timeout=600,
        kernel_name="python3",
        resources={"metadata": {"path": str(path.parent)}},
        on_cell_execute=on_start,
        on_cell_executed=on_done,
    )
    start = time.monotonic()
    ok = True
    try:
        client.execute()
    except CellExecutionError as error:
        last = re.sub(r"\x1b\[[0-9;]*m", "", str(error).strip().splitlines()[-1])
        print(f"FAIL {path}: {last}")
        ok = False
    total = time.monotonic() - start
    if html is not None:  # also after a failure: the page shows the error in place
        from nbconvert import HTMLExporter

        html.mkdir(parents=True, exist_ok=True)
        body, _ = HTMLExporter().from_notebook_node(nb)
        (html / f"{path.stem}.html").write_text(body, encoding="utf-8")
        (html / f"{path.stem}.txt").write_text(digest(nb), encoding="utf-8")
    flagged = [
        (i, t)
        for i, t in times.items()
        if t > slow and not imports(nb.cells[i].source)
    ]
    print(f"{'ok  ' if ok else 'FAIL'} {path} ({total:.1f} s)")
    for i, t in sorted(flagged, key=lambda x: -x[1]):
        first = nb.cells[i].source.strip().splitlines()[0][:70]
        print(f"     slow: cell {i + 1}, {t:.1f} s here, more in the browser: {first}")
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("notebooks", nargs="+", type=Path)
    parser.add_argument("--slow", type=float, default=2.0, help="seconds; slower cells are listed")
    parser.add_argument(
        "--html", type=Path, help="also write DIR/<notebook>.html (executed) and .txt (its text)"
    )
    args = parser.parse_args()
    results = [run(p, args.slow, args.html) for p in args.notebooks]
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
