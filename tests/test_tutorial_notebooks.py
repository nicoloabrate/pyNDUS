"""Smoke-test tutorial notebooks by executing their cells."""

import json
import os
import sys
from pathlib import Path

import pytest

nbformat = pytest.importorskip("nbformat")
preprocessors = pytest.importorskip("nbconvert.preprocessors")
ExecutePreprocessor = preprocessors.ExecutePreprocessor

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = ROOT / "docs" / "tutorials"
NOTEBOOKS = [
    NOTEBOOK_DIR / "sensitivity_serpent.ipynb",
    NOTEBOOK_DIR / "sensitivity_eranos.ipynb",
    NOTEBOOK_DIR / "sensitivity_algebra.ipynb",
    NOTEBOOK_DIR / "covariance.ipynb",
    NOTEBOOK_DIR / "sandwich.ipynb",
]


def _make_current_python_kernel(tmp_path):
    """Create a temporary kernelspec pointing to the current interpreter."""
    kernel_name = "pyndus-current-python"
    kernel_dir = tmp_path / "kernels" / kernel_name
    kernel_dir.mkdir(parents=True)
    kernel_json = {
        "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
        "display_name": f"pyNDUS tests ({Path(sys.executable).parent.parent.name})",
        "language": "python",
    }
    (kernel_dir / "kernel.json").write_text(json.dumps(kernel_json))
    return kernel_name, tmp_path


@pytest.mark.parametrize("notebook", NOTEBOOKS, ids=lambda path: path.name)
def test_tutorial_notebook_executes(notebook, tmp_path, monkeypatch):
    """Execute each tutorial notebook as a lightweight integration test."""
    kernel_name, kernel_root = _make_current_python_kernel(tmp_path)
    previous_jupyter_path = os.environ.get("JUPYTER_PATH")
    if previous_jupyter_path:
        monkeypatch.setenv("JUPYTER_PATH", f"{kernel_root}{os.pathsep}{previous_jupyter_path}")
    else:
        monkeypatch.setenv("JUPYTER_PATH", str(kernel_root))

    nb = nbformat.read(notebook, as_version=4)
    executor = ExecutePreprocessor(timeout=240, kernel_name=kernel_name)
    executor.preprocess(nb, {"metadata": {"path": str(notebook.parent)}})
