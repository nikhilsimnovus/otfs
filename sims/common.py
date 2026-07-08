"""Shared setup for simulation scripts."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

import matplotlib
matplotlib.use("Agg")
