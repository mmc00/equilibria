#!/usr/bin/env python3
"""Compatibility wrapper for relocated script."""

from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
runpy.run_path(str(ROOT / "scripts" / "dev" / "analyze_sam_storage.py"), run_name="__main__")
