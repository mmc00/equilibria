"""Regression test for diff_mcp_pairing.py's GAMS<->Python pairing resolution.

The name-only heuristic (_py_family_state) produced 3 FALSE POSITIVES on
gtap7_3x3/check:
  - pseq.ps      -> mapped to eq_ps (legacy, deactivated) instead of the real
                    homolog eq_xs (the CES domestic-supply equation, active);
  - dintxeq.dintx, mintxeq.mintx -> equation deactivated BUT the variable is
                    fixed (exogenous wedge / holdfix-equivalent), which is a valid
                    pairing, not a free variable.
The only REAL anomaly is pfteq: a GAMS free-row that Python keeps active+paired
(the factor-2 class).

This test gates on the resolver returning exactly the pfteq violation.
Requires the local GAMS reference + the model build path, so it is skipped when
the reference .gms is absent.
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "gtap"))

GMS = Path(
    "/Users/marmol/proyectos2/equilibria_refs/gtap7_3x3_altertax_cd/"
    "model_altertax_ifsub0.gms"
)

pytestmark = pytest.mark.skipif(
    not GMS.exists(), reason="local GAMS reference .gms not present"
)


def _run_check():
    import argparse
    from diff_mcp_pairing import _work

    args = argparse.Namespace(
        dataset="gtap7_3x3",
        period="check",
        ifsub=0,
        model_name="gtap",
        apply_closure=True,
    )
    return _work(args)


def test_check_reports_only_pfteq():
    res = _run_check()
    viols = res["violations"]
    entities = sorted(v["entity"] for v in viols)
    assert entities == ["pfteq"], (
        f"expected only pfteq, got {entities} "
        f"(false positives from name-heuristic pairing not eliminated)"
    )
    assert res["status"] == "dirty"  # pfteq is a real anomaly -> still dirty


def test_pseq_resolves_to_active_homolog_not_legacy_eq_ps():
    """pseq must resolve to eq_xs (active), not eq_ps (deactivated legacy)."""
    res = _run_check()
    # pseq must NOT appear as a violation
    assert "pseq" not in {v["entity"] for v in res["violations"]}


def test_fixed_var_with_deactivated_eq_is_not_a_violation():
    """dintxeq/mintxeq: eq deactivated but var fixed -> valid exogenous pairing."""
    res = _run_check()
    bad = {v["entity"] for v in res["violations"]} & {"dintxeq", "mintxeq"}
    assert not bad, f"fixed-var wedges wrongly flagged: {bad}"
