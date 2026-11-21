"""Run every notebook of the built JupyterLite site in a headless browser.

Serves the site, opens each notebook in the interface the site starts on (`appUrl` in
lite/jupyter-lite.json: JupyterLab or Jupyter Notebook), runs all cells on the Pyodide kernel and
fails when a cell raises, when a cell was not executed, when a dialog opens (e.g. "Select Kernel":
a visitor would be stopped by it too: the notebook metadata needs `language_info.name = "python"`,
which JupyterLab writes but `nbformat.v4.new_notebook()` does not), when the tab crashes or the
kernel dies, when no cell finishes for `--stall` seconds once the first one has (the kernel
loading its packages comes before), or when the run does not finish in time. It prints each
cell as it finishes, so a slow or stuck cell is named. A failure leaves a screenshot in
_test-lite/ (when the page can still take one). This is the browser half of "works
locally and in JupyterLite"; `pixi run check-env` is the local half.

Figures: once a notebook has run, every cell whose output is a Bokeh, HoloViews, Panel or Plotly
figure (by its MIME type, not by its looks) must be drawn at least `--min-figure` px high within
`--draw` seconds, or the run fails: a plotting package whose lab extension is missing from the
`lite` environment runs without error and draws nothing. Each figure is saved as
_test-lite/<notebook>-cellNN.png (NN: the cell number of `test-nb` and _preview/*.txt).

Before starting it checks the available memory: a Pyodide tab with the scientific stack needs
about 2 GiB, and below that the tab crashes after minutes instead of failing at once.

The outcome is also written to one line in _test-lite/status, for whoever started the run in the
background: `running ...` (written by `--mark-running`, the first step of `pixi run test-lite`,
before the site build), then `ok ...`, `FAIL ...` or `refused ...` when the browser run ends. A
file still saying `running` after the command has exited means the build failed: read its output.

Usage: python lite/run_notebooks.py _output [--timeout 600] [--stall 300] [--only NAME.ipynb]
                                            [--min-memory 2.0]   (0 skips the memory check)
                                            [--min-figure 40] [--draw 20]
       python lite/run_notebooks.py --mark-running
Needs `"exposeAppInBrowser": true` in lite/jupyter-lite.json (gives the page `window.jupyterapp`).
"""

import argparse
import functools
import http.server
import json
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

from doctor import MIN_MEMORY_GIB, available_gib
from playwright.sync_api import Error, sync_playwright

# The page that opens one notebook, per `appUrl`: "./lab" is JupyterLab, "./tree" is Jupyter
# Notebook (its file browser; a notebook opens in the `notebooks` app).
NOTEBOOK_PAGE = {"lab": "lab/index.html", "tree": "notebooks/index.html"}

IS_OPEN = """
(name) => {
  const widget = window.jupyterapp?.shell?.currentWidget;
  return Boolean(widget?.context?.path?.endsWith(name) && widget.sessionContext);
}
"""
# Starts "run all" without waiting for it; STATE is then polled from Python.
START = """
async () => {
  const app = window.jupyterapp;
  const panel = app.shell.currentWidget;
  window.__run = { done: false, error: null };
  await panel.context.ready;
  await panel.sessionContext.ready;
  app.commands.execute("notebook:run-all-cells").then(
    () => { window.__run.done = true; },
    (e) => { window.__run.error = String(e); }
  );
}
"""
STATE = """
() => {
  const panel = window.jupyterapp.shell.currentWidget;
  const box = document.querySelector(".jp-Dialog-content");
  const cells = panel.context.model.toJSON().cells;
  const text = (c) => (Array.isArray(c.source) ? c.source.join("") : String(c.source)).trim();
  const numbers = (keep) => cells.map((c, i) => (keep(c) ? i + 1 : 0)).filter(Boolean);
  return {
    kernel: panel.sessionContext.session?.kernel?.status ?? "none",
    executed: numbers((c) => c.cell_type === "code" && c.execution_count !== null),
    code: numbers((c) => c.cell_type === "code" && text(c)),
    errored: cells.some((c) => (c.outputs || []).some((o) => o.output_type === "error")),
    dialog: box ? box.innerText.replace(/\\s+/g, " ").trim() : null,
    done: window.__run.done,
    error: window.__run.error,
  };
}
"""
NOTEBOOK = "() => window.jupyterapp.shell.currentWidget.context.model.toJSON()"
# Outputs that are figures drawn by JavaScript in the page: they can run fine and draw nothing.
# (Static images, image/png from matplotlib, are drawn by the kernel: nothing to check.)
FIGURE_MIMES = [
    "application/vnd.holoviews_exec.v0+json",  # HoloViews, hvPlot, Panel
    "application/vnd.bokehjs_exec.v0+json",  # bokeh.plotting.show
    "application/vnd.plotly.v1+json",
]
# The cell of `hv.extension(...)` / `pn.extension(...)` loads the JavaScript and also emits an
# exec output with an empty <div>: not a figure. Keep the extension call in a cell of its own.
LOAD_MIMES = ["application/vnd.holoviews_load.v0+json", "application/vnd.bokehjs_load.v0+json"]
FIGURE_CELLS = """
([mimes, loads]) => {
  const panel = window.jupyterapp.shell.currentWidget;
  const found = [];
  const has = (outputs, list) => outputs.some((o) => o.data && list.some((m) => m in o.data));
  panel.content.widgets.forEach((cell, i) => {
    if (cell.model.type !== "code") return;
    const outputs = cell.model.outputs.toJSON();
    if (has(outputs, mimes) && !has(outputs, loads)) found.push(i);
  });
  return found;
}
"""
# The output area of cell i, scrolled into view (an off-screen output may not be drawn).
OUTPUT_AREA = """
(i) => {
  const cell = window.jupyterapp.shell.currentWidget.content.widgets[i];
  cell.node.scrollIntoView({ block: "start" });
  return cell.outputArea.node;
}
"""
HEIGHT = """
(area) => Math.max(0, ...[...area.querySelectorAll(".jp-OutputArea-output")].map(
  (o) => o.getBoundingClientRect().height))
"""
SHOTS = Path("_test-lite")
STATUS = SHOTS / "status"


def write_status(text: str) -> None:
    """One line, read by whoever waits for a background run: ok / FAIL / refused / running."""
    SHOTS.mkdir(exist_ok=True)
    STATUS.write_text(f"{text} [{datetime.now():%Y-%m-%d %H:%M:%S}]\n")


class QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *args):
        pass


def serve(directory: Path) -> http.server.ThreadingHTTPServer:
    handler = functools.partial(QuietHandler, directory=str(directory))
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def notebook_page(site: Path) -> str:
    config = json.loads((site / "jupyter-lite.json").read_text())
    app = config["jupyter-config-data"].get("appUrl", "./lab").strip("./")
    if app not in NOTEBOOK_PAGE:
        sys.exit(f"appUrl {app!r} in lite/jupyter-lite.json: expected ./lab or ./tree")
    return NOTEBOOK_PAGE[app]


def problems(notebook: dict) -> list[str]:
    """Cells that raised or were never executed."""
    found = []
    for number, cell in enumerate(notebook["cells"], start=1):
        source = cell["source"] if isinstance(cell["source"], str) else "".join(cell["source"])
        if cell["cell_type"] != "code" or not source.strip():
            continue
        errors = [o for o in cell.get("outputs", []) if o["output_type"] == "error"]
        for error in errors:
            found.append(f"cell {number}: {error['ename']}: {error['evalue']}")
        if not errors and cell.get("execution_count") is None:
            found.append(f"cell {number}: not executed")
    return found


def check_figures(page, name: str, min_height: float, draw: float) -> tuple[list[str], int]:
    """Figures drawn too small (problems), and the number of figures saved as PNG."""
    found, saved = [], 0
    stem = name.removesuffix(".ipynb").replace("/", "__")
    for i in page.evaluate(FIGURE_CELLS, [FIGURE_MIMES, LOAD_MIMES]):
        area = page.evaluate_handle(OUTPUT_AREA, i).as_element()
        deadline = time.monotonic() + draw
        height = area.evaluate(HEIGHT)
        while height < min_height and time.monotonic() < deadline:
            time.sleep(0.5)
            height = area.evaluate(HEIGHT)
        if height < min_height:
            found.append(
                f"cell {i + 1}: figure drawn {height:.0f} px high after {draw:.0f} s "
                f"(--min-figure {min_height:.0f}): a lab extension missing from the lite env?"
            )
        shot = SHOTS / f"{stem}-cell{i + 1:02d}.png"
        try:
            area.screenshot(path=shot, timeout=10_000)
            saved += 1
        except Error:
            pass
    return found, saved


def memory_note() -> str:
    memory = available_gib()
    return "" if memory is None else f" (memory available now: {memory:.1f} GiB)"


def run_one(page, timeout: float, stall: float) -> list[str]:
    """Run all cells of the open notebook, printing each cell as it finishes."""
    page.evaluate(START)
    start = last = time.monotonic()
    seen: set[int] = set()
    alive = False  # the kernel has reported idle or busy at least once
    while True:
        time.sleep(1)
        state = page.evaluate(STATE)
        now = time.monotonic()
        new = [c for c in state["executed"] if c not in seen]
        for cell in new:
            print(f"     cell {cell} done (+{now - last:.0f} s)", flush=True)
            seen.add(cell)
        if new:
            last = now
        if state["dialog"]:
            return [f"dialog: {state['dialog']}"]
        if state["error"]:
            return [f"run-all failed: {state['error']}"]
        if state["done"] or state["errored"]:
            return problems(page.evaluate(NOTEBOOK))
        alive |= state["kernel"] in ("idle", "busy")
        if state["kernel"] == "dead" or (alive and state["kernel"] in ("unknown", "none")):
            return [f"kernel {state['kernel']} after {now - start:.0f} s{memory_note()}"]
        if seen and now - last > stall:  # from the first finished cell: loading is not a stall
            pending = [c for c in state["code"] if c not in seen]
            where = f"cell {pending[0]}" if pending else "the kernel"
            return [f"{where} still running after {now - last:.0f} s (--stall {stall:.0f})"]
        if now - start > timeout:
            return [f"not finished after {timeout:.0f} s (--timeout)"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("site", type=Path, nargs="?", help="the built site (_output)")
    parser.add_argument("--timeout", type=float, default=600, help="seconds per notebook")
    parser.add_argument("--stall", type=float, default=300, help="seconds without a finished cell")
    parser.add_argument("--only", help="run this notebook only (path inside the site contents)")
    parser.add_argument(
        "--min-memory", type=float, default=MIN_MEMORY_GIB, help="GiB needed to start; 0: no check"
    )
    parser.add_argument(
        "--min-figure", type=float, default=40, help="px; a figure drawn smaller fails the run"
    )
    parser.add_argument(
        "--draw", type=float, default=20, help="seconds a figure has to reach --min-figure"
    )
    parser.add_argument(
        "--mark-running",
        action="store_true",
        help=f"only write `running` to {STATUS} (first step of test-lite, before the build)",
    )
    args = parser.parse_args()
    if args.mark_running:
        write_status("running: build-lite, then the browser run")
        return 0
    if args.site is None:
        parser.error("the built site is required (_output)")
    try:
        code, status = run_all(args)
    except BaseException as error:  # interrupted or crashed: the status must not say running
        write_status(f"FAIL {type(error).__name__}: {error}".splitlines()[0])
        raise
    write_status(status)
    return code


def run_all(args) -> tuple[int, str]:
    """Exit code and the status line."""
    memory = available_gib()
    if args.min_memory and memory is not None and memory < args.min_memory:
        print(
            f"only {memory:.1f} GiB of memory available, the browser test needs "
            f"{args.min_memory:.1f}: the tab would crash. Close browsers or other large apps, "
            "or pass --min-memory 0 to try anyway (pixi run doctor shows what else to check)."
        )
        return 2, f"refused: {memory:.2f} GiB of memory available, needs {args.min_memory:.1f}"

    contents = args.site / "files"
    names = sorted(p.relative_to(contents).as_posix() for p in contents.rglob("*.ipynb"))
    if args.only:
        names = [n for n in names if n == args.only]
    if not names:
        print(f"no notebook found in {contents}")
        return 1, f"FAIL no notebook found in {contents}"

    opener = notebook_page(args.site)
    SHOTS.mkdir(exist_ok=True)
    for old in SHOTS.glob("*.png"):  # screenshots of an earlier run would pass for this one's
        old.unlink()
    mem = "" if memory is None else f", {memory:.1f} GiB of memory available"
    print(f"opening notebooks with {opener}{mem}", flush=True)
    server = serve(args.site)
    base = f"http://127.0.0.1:{server.server_port}"
    failed: list[str] = []
    began = time.monotonic()
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        for name in names:
            # a new context per notebook: empty browser storage, as for a first-time visitor.
            # In a shared context JupyterLab restores the previous notebook and its dialogs.
            context = browser.new_context(viewport={"width": 1300, "height": 900})
            page = context.new_page()
            crashed = []
            page.on("crash", lambda *_, crashed=crashed: crashed.append(True))
            start = time.monotonic()
            print(f"run  {name}", flush=True)
            shots = 0
            try:
                page.goto(f"{base}/{opener}?path={name}")
                page.wait_for_function(IS_OPEN, arg=name, timeout=120_000)
                found = run_one(page, args.timeout, args.stall)
                if not found:
                    found, shots = check_figures(page, name, args.min_figure, args.draw)
            except Error as error:
                first = str(error).splitlines()[0]
                if crashed:
                    first = f"the browser tab crashed (out of memory?){memory_note()}: {first}"
                found = [first]
            if found:
                shot = SHOTS / (name.replace("/", "__") + ".png")
                try:  # a crashed page cannot take one: keep the first error, not this one
                    page.screenshot(path=shot, timeout=10_000)
                    found.append(f"screenshot: {shot}")
                except Error:
                    found.append("no screenshot: the page is gone")
            context.close()
            if found:
                failed.append(f"{name}: {found[0]}")
            figs = f", {shots} figure{'s' * (shots != 1)} drawn" if not found and shots else ""
            print(
                f"{'FAIL' if found else 'ok  '} {name} ({time.monotonic() - start:.0f} s{figs})",
                flush=True,
            )
            for line in found:
                print(f"     {line}", flush=True)
        browser.close()
    server.shutdown()
    took = f"{time.monotonic() - began:.0f} s"
    if failed:
        return 1, f"FAIL {len(failed)} of {len(names)} notebooks ({took}): " + "; ".join(failed)
    return 0, f"ok {len(names)} notebook{'s' * (len(names) != 1)} ({took})"


if __name__ == "__main__":
    sys.exit(main())
