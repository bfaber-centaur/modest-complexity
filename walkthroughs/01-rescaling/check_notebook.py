"""Run this walkthrough in order without Jupyter sockets; save real cell outputs.

This checks Python cells and assertions, not interactive Jupyter UI behavior.
Run: uv run --locked python check_notebook.py
"""
from pathlib import Path
import os

for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS"):
    os.environ[name] = "1"

import nbformat
from IPython.terminal.interactiveshell import TerminalInteractiveShell
from IPython.utils.capture import capture_output


def main():
    path = Path(__file__).resolve().with_name("rescaling.ipynb")
    os.chdir(path.parent)
    notebook = nbformat.read(path, as_version=4)
    nbformat.validate(notebook)
    shell = TerminalInteractiveShell.instance()
    shell.run_line_magic("matplotlib", "inline")
    shell.display_formatter.active_types = [
        "text/plain", "text/html", "text/markdown", "image/png"
    ]
    count = 0
    for cell in notebook.cells:
        if cell.cell_type != "code":
            continue
        count += 1
        with capture_output() as captured:
            result = shell.run_cell(cell.source, store_history=False)
        result.raise_error()  # leave the saved notebook untouched on failure
        cell.execution_count = count
        cell.outputs = []
        for name, text in (("stdout", captured.stdout), ("stderr", captured.stderr)):
            if text:
                cell.outputs.append(nbformat.v4.new_output("stream", name=name, text=text))
        for output in captured.outputs:
            cell.outputs.append(nbformat.v4.new_output(
                "display_data", data=output.data, metadata=output.metadata
            ))
        print(f"Cell {count}: passed")
    notebook.metadata["execution_method"] = "in-process IPython; check_notebook.py"
    nbformat.validate(notebook)
    nbformat.write(notebook, path)
    print(f"Saved {count} checked code cells: {path}")


if __name__ == "__main__":
    main()
