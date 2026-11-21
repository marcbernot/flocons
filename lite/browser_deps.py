"""Can the browser have [project].dependencies? Seconds, before any code (DUAL-ENV.md, 4).

For each package of [project].dependencies in pyproject.toml:

  ok    Pyodide builds it: its version in the Pyodide of this kernel, read from the official
        pyodide-lock.json on the CDN (the Pyodide version comes from jupyterlite-pyodide-kernel)
  ok    not built by Pyodide, but PyPI has a pure-Python wheel (py3-none-any): the lock step
        takes it from PyPI
  FAIL  neither: the browser cannot have it. Find another package, or keep it desktop-only
        (the `local` extra, imported inside functions)

A version specifier that the Pyodide version does not satisfy fails too (a semver pin newer than
Pyodide). Only the direct dependencies are checked, and only by name and version:
`pixi run build-lite` (run by `test-lite`) stays the full check, dependencies of dependencies
included. Needs the network (the CDN and pypi.org).

Usage: python lite/browser_deps.py
"""

import json
import sys
import tomllib
import urllib.request
from pathlib import Path

from jupyterlite_pyodide_kernel.constants import PYODIDE_CDN_URL, PYODIDE_VERSION
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
from packaging.version import InvalidVersion, Version

ROOT = Path(__file__).resolve().parents[1]


def get_json(url: str) -> dict | None:
    try:
        with urllib.request.urlopen(url, timeout=30) as resp:
            return json.load(resp)
    except OSError:
        return None


def pure_wheel_on_pypi(name: str, req: Requirement) -> str | None:
    """Newest version that satisfies ``req`` and has a py3-none-any wheel, or None."""
    data = get_json(f"https://pypi.org/pypi/{name}/json")
    if data is None:
        return None
    versions = []
    for version, files in data.get("releases", {}).items():
        try:
            wanted = req.specifier.contains(Version(version), prereleases=False)
        except InvalidVersion:  # old releases with non-PEP 440 versions
            continue
        pure = any(f["filename"].endswith("-none-any.whl") and not f.get("yanked") for f in files)
        if wanted and pure:
            versions.append(version)
    if not versions:
        return None
    return str(max(versions, key=Version))


def main() -> int:
    deps = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["dependencies"]
    lock = get_json(f"{PYODIDE_CDN_URL}/pyodide-lock.json")
    if lock is None:
        print(f"FAIL cannot read {PYODIDE_CDN_URL}/pyodide-lock.json (network? `pixi run doctor`)")
        return 1
    packages = {canonicalize_name(k): v for k, v in lock["packages"].items()}
    print(f"Pyodide {PYODIDE_VERSION}: {len(packages)} packages")
    bad = 0
    for dep in deps:
        req = Requirement(dep)
        name = canonicalize_name(req.name)
        if name in packages:
            version = packages[name]["version"]
            if req.specifier.contains(version, prereleases=True):
                print(f"ok   {req.name} {version} (built by Pyodide)")
            else:
                print(f"FAIL {req.name}: Pyodide has {version}, outside {req.specifier}")
                bad += 1
            continue
        version = pure_wheel_on_pypi(name, req)
        if version:
            print(f"ok   {req.name} {version} (pure-Python wheel on PyPI)")
        else:
            print(
                f"FAIL {req.name}: not built by Pyodide {PYODIDE_VERSION} and no pure-Python "
                f"wheel on PyPI{' within ' + str(req.specifier) if req.specifier else ''}"
            )
            bad += 1
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
