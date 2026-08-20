"""Pytest configuration for the local src-layout package."""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

pythonpath = os.environ.get("PYTHONPATH")
paths = [] if not pythonpath else pythonpath.split(os.pathsep)
if str(SRC) not in paths:
    os.environ["PYTHONPATH"] = os.pathsep.join([str(SRC), *paths])
