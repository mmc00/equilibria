"""The GTAP door: everything to load, build, shock and read a GTAP exercise."""

from __future__ import annotations

from equilibria.gtap._load import load
from equilibria.gtap._results import GTAPResults, results

__all__ = ["GTAPResults", "load", "results"]
