"""Sphinx configuration for the pyNDUS documentation."""

from __future__ import annotations

import os
import sys
from datetime import datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

project = "pyNDUS"
author = "Nicolò Abrate"
copyright = f"{datetime.now().year}, {author}"

try:
    release = version("pyNDUS")
except PackageNotFoundError:
    release = "0.0.1"
version = release

extensions = [
    "sphinx.ext.autodoc", "sphinx.ext.autosummary", "sphinx.ext.intersphinx", "sphinx.ext.mathjax",
    "sphinx.ext.napoleon", "sphinx.ext.viewcode", "nbsphinx", ]

autosummary_generate = True
autodoc_typehints = "description"
autodoc_member_order = "bysource"
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False

# Keep documentation builds independent from local NJOY/SANDY installations.
autodoc_mock_imports = ["sandy", "serpentTools", "matplotlib", "matplotlib.pyplot", "h5py", ]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# Tutorial notebooks depend on local benchmark data; render them without execution.
nbsphinx_execute = "never"
# Use a built-in lexer as fallback for notebook code cells on hosted builds.
nbsphinx_codecell_lexer = "python3"

highlight_language = "python"
pygments_style = "default"

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_css_files = ["notebook-syntax.css"]
html_title = "pyNDUS documentation"

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None), }

rst_prolog = """
.. |project| replace:: pyNDUS
"""
