"""Write a notebook from a py:percent file, then delete the .py (one-way: no jupytext pairing).

    pixi run nb-new notebooks/analysis.py [--keep] [--force]

The .py holds cells separated by `# %%` (code) and `# %% [markdown]` (markdown, each line
commented with `# `). The .ipynb gets the metadata JupyterLite needs (kernelspec and
language_info: without them the browser stops on a "Select Kernel" dialog), whatever the .py
header says, and no jupytext metadata, so nothing pairs it back to a text file.

Until the notebook is first committed it has no outputs and no edits of its own to lose, so the
quickest loop is to keep editing the .py and convert again:

    pixi run nb-new notebooks/analysis.py --keep --force    # each round: the .py stays
    pixi run nb-new notebooks/analysis.py --force           # last time: the .py goes

Once the .ipynb is committed, `--force` refuses (a regenerated notebook would drop what was
changed in it since): from then on the .ipynb is the only file, edit it.
"""

import argparse
import subprocess
import sys
from pathlib import Path

import jupytext

METADATA = {
    "kernelspec": {"display_name": "Python 3 (ipykernel)", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
}


def committed(path: Path) -> bool:
    """True when the last commit holds the file (staged by `tidy` only does not count)."""
    try:
        done = subprocess.run(
            ["git", "ls-tree", "--name-only", "HEAD", "--", path.name],
            cwd=path.parent,
            capture_output=True,
            text=True,
        )
    except OSError:  # no git: nothing to protect
        return False
    return done.returncode == 0 and bool(done.stdout.strip())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("py", type=Path, help="py:percent file, e.g. notebooks/analysis.py")
    parser.add_argument("--keep", action="store_true", help="keep the .py afterwards")
    parser.add_argument("--force", action="store_true", help="overwrite an existing .ipynb")
    args = parser.parse_args()
    src: Path = args.py
    out = src.with_suffix(".ipynb")
    if src.suffix != ".py" or not src.is_file():
        print(f"not a .py file: {src}")
        return 1
    if out.exists() and not args.force:
        print(f"{out} exists: edit it, or pass --force to overwrite it (before its first commit)")
        return 1
    if out.exists() and committed(out):
        print(f"{out} is committed: edit it, not the .py (--force: before the first commit only)")
        return 1
    nb = jupytext.read(src, fmt="py:percent")
    nb.metadata.pop("jupytext", None)
    nb.metadata.update(METADATA)
    for cell in nb.cells:
        cell.metadata.pop("lines_to_next_cell", None)
    jupytext.write(nb, out, fmt="ipynb")
    kinds = [c.cell_type for c in nb.cells]
    print(f"wrote {out}: {kinds.count('code')} code, {kinds.count('markdown')} markdown cells")
    if args.keep:
        print(
            f"kept {src}: edit it and run `pixi run nb-new {src} --keep --force` again; "
            "before the first commit, once without --keep to remove it"
        )
    else:
        src.unlink()
        print(f"removed {src}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
