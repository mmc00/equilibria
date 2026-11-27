"""The GEMPACK translation MAR-249 moves into the package gives what main gave.

The oracle (``tests/compare/golden/mar249_translation_main.json.gz``) was recorded on
main (2c84f805) by ``scripts/dev/record_mar249_translation.py`` before the translator
left ``scripts/gtap/compare_gempack_levels.py``; the test runs that recorder on this
checkout in a child process and must give the same ``equilibria_side`` levels and
``pct_changes`` of the AlterTax nus333 run, within 1e-12 relative + 1e-12 absolute.
"""

from __future__ import annotations

import gzip
import importlib.util
import json
from pathlib import Path

import pytest

from tests._golden import mismatches

ROOT = Path(__file__).resolve().parents[2]
with gzip.open(
    ROOT / "tests" / "compare" / "golden" / "mar249_translation_main.json.gz", "rt"
) as _f:
    GOLDEN = json.load(_f)
pytest.importorskip("mumps")


def _recorder():
    path = ROOT / "scripts" / "dev" / "record_mar249_translation.py"
    spec = importlib.util.spec_from_file_location("record_mar249_translation", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_translation_matches_main() -> None:
    from equilibria._local_refs import nus333_dir

    if not (nus333_dir() / "basedata.har").exists():
        pytest.skip("nus333 data missing")
    got = _recorder().run_case(ROOT)
    assert set(got) == set(GOLDEN)
    for section in ("levels", "pct"):
        bad = mismatches(got[section], GOLDEN[section], rel=1e-12, abs_tol=1e-12)
        assert bad == [], f"{section}: {len(bad)} cells, first: {bad[:5]}"
