# flocons

TODO: one paragraph on what this is.

## Run locally

```bash
pixi run lab        # installs the environment on first run
pixi task list -s   # the other tasks, per environment
```

The notebooks in `notebooks/` run unchanged locally and in the browser (JupyterLite):
`pixi run build-lite && pixi run serve-lite` serves the site on <http://localhost:8000>.
