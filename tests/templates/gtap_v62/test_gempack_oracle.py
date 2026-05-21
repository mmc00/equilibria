"""Tests for the GEMPACK oracle wrapper (``scripts/gtap_v62/run_gempack_oracle.py``).

Marked ``@pytest.mark.gempack`` so they only run on hosts that have a
RunGTAP install. CI on Linux/macOS skips these automatically.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pytest

# Make scripts/ importable
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from scripts.gtap_v62.run_gempack_oracle import (  # noqa: E402
    DEFAULT_RUNGTAP_DIR,
    is_rungtap_available,
    run_gempack_experiment,
)


BOOK3X3 = Path("C:/runGTAP375/BOOK3X3")


pytestmark = [
    pytest.mark.gempack,
    pytest.mark.skipif(
        not (is_rungtap_available() and BOOK3X3.exists()),
        reason="RunGTAP v6.2 / BOOK3X3 dataset not available",
    ),
]


def test_oracle_runs_book3x3_exp1a(tmp_path: Path) -> None:
    """gtap.exe runs Exp1a (10% tariff cut on US food→EU) end-to-end.

    Verifies that:
    1. The CMF-prep step injects GTAPSETS / GTAPDATA / GTAPPARM correctly.
    2. gtap.exe terminates with returncode 0.
    3. sltohta.exe converts the .sl4 to .har and the .har is readable.
    """
    result = run_gempack_experiment(
        "Exp1a",
        dataset_dir=BOOK3X3,
        workdir=tmp_path / "oracle_run",
    )

    assert result.status == "ok", (
        f"Oracle failed (rc={result.returncode}): {result.message}\n"
        f"See {result.log_path}"
    )
    assert result.sl4_path is not None and result.sl4_path.exists()
    assert result.solution_har is not None and result.solution_har.exists()


def test_oracle_solution_har_has_expected_variables(tmp_path: Path) -> None:
    """The generated solution HAR contains v6.2 endogenous variables."""
    from equilibria.babel.har import read_har

    result = run_gempack_experiment(
        "Exp1a",
        dataset_dir=BOOK3X3,
        workdir=tmp_path / "oracle_solhar",
    )
    assert result.status == "ok"

    sol = read_har(result.solution_har)

    # v6.2 endogenous variables that must show up post-solve
    headers_text = " ".join(
        (arr.long_name or "") for arr in sol.values() if hasattr(arr, "long_name")
    )
    for token in ("qo ", "qfe ", "qfd ", "qpd ", "qxs ", "pm ", "ps "):
        assert token in headers_text, f"Missing variable {token!r} in solution HAR"


def test_oracle_reports_missing_exe(tmp_path: Path) -> None:
    """If gtap.exe is missing, oracle returns 'missing_exe' rather than crashing."""
    bogus = tmp_path / "no_rungtap_here"
    bogus.mkdir()

    result = run_gempack_experiment(
        "Exp1a",
        dataset_dir=BOOK3X3,
        workdir=tmp_path / "oracle_no_exe",
        rungtap_dir=bogus,
    )

    assert result.status == "missing_exe"
    assert result.sl4_path is None
    assert "not found" in result.message.lower()


def test_oracle_idempotent_workdir(tmp_path: Path) -> None:
    """Re-running with a clean workdir produces a fresh solution."""
    workdir = tmp_path / "oracle_idempotent"
    r1 = run_gempack_experiment("Exp1a", dataset_dir=BOOK3X3, workdir=workdir)
    assert r1.status == "ok"

    shutil.rmtree(workdir)
    r2 = run_gempack_experiment("Exp1a", dataset_dir=BOOK3X3, workdir=workdir)
    assert r2.status == "ok"
    assert r2.solution_har is not None and r2.solution_har.exists()
