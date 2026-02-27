#!/usr/bin/env python3
"""Backward-compatible entrypoint for systemic parity runner."""

from __future__ import annotations

import runpy
from pathlib import Path


if __name__ == "__main__":
    target = Path(__file__).resolve().parent / "parity" / "run_pep_systemic_parity.py"
    runpy.run_path(str(target), run_name="__main__")

