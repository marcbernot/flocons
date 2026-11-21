import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).parents[1]

# conda-forge name -> PyPI name, where they differ (add yours: pytables -> tables, ...)
CONDA_TO_PYPI = {"matplotlib-base": "matplotlib", "pytables": "tables"}


def _norm(name: str) -> str:
    name = re.sub(r"[-_.]+", "-", name).lower()
    return CONDA_TO_PYPI.get(name, name)


def test_runtime_feature_matches_project_dependencies():
    """[feature.runtime] in pixi.toml (local) == [project].dependencies (what the browser gets).

    Both directions: a package missing locally makes a notebook fail here; a package missing
    from [project].dependencies makes it fail in the browser only, which is worse.
    """
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    runtime = tomllib.loads((ROOT / "pixi.toml").read_text())["feature"]["runtime"]
    local = {_norm(n) for n in runtime.get("dependencies", {})}
    local |= {_norm(n) for n in runtime.get("pypi-dependencies", {})}
    browser = {_norm(re.split(r"[<>=!~;\[ ]", d, maxsplit=1)[0]) for d in project["dependencies"]}
    assert not browser - local, (
        f"in [project].dependencies, not in [feature.runtime] of pixi.toml: "
        f"{sorted(browser - local)} (pixi add --feature runtime <pkg>)"
    )
    assert not local - browser, (
        f"in [feature.runtime] of pixi.toml, not in [project].dependencies: "
        f"{sorted(local - browser)} (the browser would not get them; a tool goes in "
        f"[dependencies] instead)"
    )
