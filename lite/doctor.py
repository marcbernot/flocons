"""Check that this machine can run the browser test, before spending minutes on it.

Reports free memory, free disk, whether Playwright's Chromium is installed and whether the
Pyodide CDN answers for the packages of the built site. Exits 1 when a check fails, so it can
gate a step. Standard library only: runs in any environment.

Usage: python lite/doctor.py [--site _output] [--min-memory 2.0]

`test-lite` runs the memory check itself (lite/run_notebooks.py imports `available_gib`).
"""

import argparse
import json
import os
import shutil
import sys
import urllib.request
from pathlib import Path

# A Pyodide tab with numpy + scipy + scikit-image + bokeh + panel crashed with 1.0-1.2 GiB
# available (flocons, 2026-11-19) and ran in 55 s once memory was freed: 2 GiB is the floor.
MIN_MEMORY_GIB = 2.0
MIN_DISK_GIB = 2.0


def available_gib() -> float | None:
    """Memory available to a new process, in GiB (Linux: MemAvailable); None if unknown."""
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / 1024**2
    except OSError:
        pass
    return None  # macOS / Windows: not measured


def playwright_browsers() -> list[str]:
    """Browser builds Playwright has downloaded (its default cache, or PLAYWRIGHT_BROWSERS_PATH)."""
    custom = os.environ.get("PLAYWRIGHT_BROWSERS_PATH")
    if custom:
        root = Path(custom)
    elif sys.platform == "darwin":
        root = Path.home() / "Library" / "Caches" / "ms-playwright"
    elif sys.platform == "win32":
        root = Path(os.environ.get("LOCALAPPDATA", "")) / "ms-playwright"
    else:
        root = Path.home() / ".cache" / "ms-playwright"
    return sorted(p.name for p in root.glob("[!.]*")) if root.is_dir() else []


def cdn_url(site: Path) -> str | None:
    """One package URL of the built site's lock that is fetched from the CDN."""
    lock = site / "static" / "pyodide-lock" / "pyodide-lock.json"
    if not lock.exists():
        return None
    packages = json.loads(lock.read_text())["packages"].values()
    remote = [p["file_name"] for p in packages if p["file_name"].startswith("https://")]
    return min(remote, key=len) if remote else None


def reachable(url: str) -> str | None:
    """None when the URL answers 200 to a HEAD request, else the problem."""
    try:
        request = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(request, timeout=20) as resp:
            return None if resp.status == 200 else f"HTTP {resp.status}"
    except Exception as error:  # any network failure is the answer
        return str(error)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--site", type=Path, default=Path("_output"), help="the built site")
    parser.add_argument("--min-memory", type=float, default=MIN_MEMORY_GIB, help="GiB")
    args = parser.parse_args()
    failed = False

    def report(ok: bool | None, what: str) -> None:
        nonlocal failed
        failed |= ok is False
        mark = {True: "ok  ", False: "FAIL", None: "?   "}[ok]
        print(f"{mark} {what}")

    memory = available_gib()
    if memory is None:
        report(None, "memory: not measured on this OS; the browser test needs ~2 GiB free")
    else:
        enough = memory >= args.min_memory
        advice = "" if enough else ": close browsers or other large apps"
        report(
            enough,
            f"memory: {memory:.1f} GiB available, test-lite needs {args.min_memory:.1f}{advice}",
        )
    disk = shutil.disk_usage(Path.cwd()).free / 1024**3
    report(disk >= MIN_DISK_GIB, f"disk: {disk:.1f} GiB free here")
    browsers = playwright_browsers()
    report(
        any(b.startswith("chromium") for b in browsers) or None,
        f"playwright browsers: {', '.join(browsers) or 'none yet (test-lite downloads Chromium)'}",
    )
    url = cdn_url(args.site)
    if url is None:
        report(None, f"CDN: not checked ({args.site} not built yet: pixi run build-lite)")
    else:
        problem = reachable(url)
        report(problem is None, f"CDN: {url} -> {problem or 'ok'}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
