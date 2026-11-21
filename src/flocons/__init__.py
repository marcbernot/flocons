"""flocons: the same code runs in a local Python and in JupyterLite (Pyodide)."""

from ._runtime import IN_BROWSER, afetch, fetch

__version__ = "0.0.1"
__all__ = ["IN_BROWSER", "afetch", "fetch"]
