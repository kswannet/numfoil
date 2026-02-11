# Interactive Airfoils (Experimental)

These interactive plots are designed to work on a *remote/headless* Linux workstation.
Instead of GUI-based matplotlib widgets, they use a **Bokeh server app** that you interact with in a web browser.

## Setup

The apps require `bokeh`.

- If you use the repo venv: `pip install bokeh`

## Run (PARSEC)

Simplest:

```bash
python -m numfoil.interactive.run_parsec_app --port 5006
```

Equivalent manual command:

```bash
python -m bokeh serve $(python -c "import numfoil, pathlib; print(pathlib.Path(numfoil.__file__).parent / 'interactive' / 'bokeh_parsec_app.py')") --port 5006
```

## Run (Modified Kulfan/CST)

Simplest:

```bash
python -m numfoil.interactive.run_kulfan_app --port 5007
```

Equivalent manual command:

```bash
python -m bokeh serve $(python -c "import numfoil, pathlib; print(pathlib.Path(numfoil.__file__).parent / 'interactive' / 'bokeh_kulfan_app.py')") --port 5007
```

If you already have a forwarded port (e.g. 5006), you can reuse it:

```bash
python -m bokeh serve $(python -c "import numfoil, pathlib; print(pathlib.Path(numfoil.__file__).parent / 'interactive' / 'bokeh_kulfan_app.py')") --port 5006
```

## Common gotcha (VS Code linkification)

If you copy a command from the VS Code terminal history or chat, VS Code may turn
file paths into clickable Markdown-like links. If you paste something like:

```text
python -m bokeh serve [bokeh_kulfan_app.py](http://...)
```

then Bash will error with:

```text
bash: syntax error near unexpected token `('
```

Make sure you paste the *plain path*:

```bash
python -m bokeh serve $(python -c "import numfoil, pathlib; print(pathlib.Path(numfoil.__file__).parent / 'interactive' / 'bokeh_kulfan_app.py')") --port 5007
```

## Viewing in VS Code Remote

1. Start the server (above).
2. In VS Code, open the **Ports** panel and forward `5006` or `5007`.
3. Open the forwarded URL in your local browser.

If you need the direct URL path, Bokeh uses the script filename by default:

- PARSEC: `http://127.0.0.1:<port>/bokeh_parsec_app`
- Kulfan: `http://127.0.0.1:<port>/bokeh_kulfan_app`

## Notes

- The Kulfan app works around a current `numfoil` issue where passing `t_te` as a Python float can throw a `torch.all(bool)` TypeError.
  The experiment always passes torch tensors for these params.
- Overlays are intentionally approximate/fast (interactive-first). If you want “publication-accurate” properties, we can add a second mode.
