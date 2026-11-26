"""The AlterTax outputs MAR-246 speeds up give what main gave.

The oracle (``tests/family/golden/mar246_headers_main.json.gz``) was recorded on main
(1bfcb8ee) by ``scripts/dev/record_mar246_headers.py`` before any MAR-246 change; each
case here runs that recorder on this checkout in a child process and must give the
same codes, margin params, ifSUB report variables, value, percent, volume and
decomposition headers, within 1e-12 relative + 1e-12 absolute (a run is bit-identical
on one machine). The decomposition and volume headers get the 1e-7 absolute floor of
``test_altertax_on_core.py``: they are sums of level differences that cancel, so
another CPU moves them more (Linux CI vs the Mac golden: volume up to 5e-9 absolute on
values up to 4.2, every other section within 1e-12).
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
    ROOT / "tests" / "family" / "golden" / "mar246_headers_main.json.gz", "rt"
) as _f:
    GOLDEN = json.load(_f)
pytest.importorskip("mumps")


def _recorder():
    path = ROOT / "scripts" / "dev" / "record_mar246_headers.py"
    spec = importlib.util.spec_from_file_location("record_mar246_headers", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _as_dict(x):
    """Lists (header arrays) as ``{index: value}``, for ``mismatches``."""
    if isinstance(x, dict):
        return {k: _as_dict(v) for k, v in x.items()}
    if isinstance(x, list):
        return {i: _as_dict(v) for i, v in enumerate(x)}
    return x


@pytest.fixture(scope="module")
def recorder():
    from equilibria._local_refs import nus333_dir

    if not (nus333_dir() / "basedata.har").exists():
        pytest.skip("nus333 data missing")
    return _recorder()


def test_the_recorder_and_the_golden_have_the_same_cases(recorder) -> None:
    assert set(recorder.CASES) == set(GOLDEN)


@pytest.mark.parametrize("case", sorted(GOLDEN))
def test_outputs_match_main(case: str, recorder) -> None:
    got = recorder.run_case(case, ROOT)
    want = GOLDEN[case]
    assert set(got) == set(want)
    assert got["codes"] == want["codes"]
    for section in ("margins", "report_vars", "headers", "pct"):
        bad = mismatches(
            _as_dict(got[section]), _as_dict(want[section]), rel=1e-12, abs_tol=1e-12
        )
        assert bad == [], f"{section}: {len(bad)} cells, first: {bad[:5]}"
    for section in ("volume", "decomp"):
        bad = mismatches(
            _as_dict(got[section]), _as_dict(want[section]), rel=1e-12, abs_tol=1e-7
        )
        assert bad == [], f"{section}: {len(bad)} cells, first: {bad[:5]}"
