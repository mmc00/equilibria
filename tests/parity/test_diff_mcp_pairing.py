"""Regression test for diff_mcp_pairing.py's GAMS<->Python pairing resolution.

The name-only heuristic (_py_family_state) produced 3 FALSE POSITIVES on
gtap7_3x3/check:
  - pseq.ps      -> mapped to eq_ps (legacy, deactivated) instead of the real
                    homolog eq_xs (the CES domestic-supply equation, active);
  - dintxeq.dintx, mintxeq.mintx -> equation deactivated BUT the variable is
                    fixed (exogenous wedge / holdfix-equivalent), which is a valid
                    pairing, not a free variable.
pfteq is a GAMS free-row that Python keeps active+paired. It was the factor-2
class while eq_pfteq read pft^2 == K (two roots); since ef807eb it is the single
positive root pft == K^(1/2), so it is benign and the check is clean.

The .gms is the one recorded with the cascade's reference (data/, fbep=header);
the tests that build the model are skipped when it is absent.
"""

import sys
from pathlib import Path

import pytest

from equilibria.sources import find_reference

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "gtap"))

_FOLDER = find_reference(
    "gtap7_3x3", "gams", form="mcp", mode="altertax", ifsub=0, fbep="header"
)
GMS = None if _FOLDER is None else _FOLDER / "model.gms"
needs_gms = pytest.mark.skipif(
    GMS is None or not GMS.exists(), reason="GAMS reference .gms not present"
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


def test_pfteq_single_root_free_row_is_benign():
    from diff_mcp_pairing import _classify_free_row

    kind, severity, _ = _classify_free_row("pfteq", "eq_pfteq", 15, 15)
    assert (kind, severity) == ("benign_active", 0.0)


def test_the_gms_is_the_cascade_reference_model():
    from diff_mcp_pairing import _resolve_gms_path

    path, _ = _resolve_gms_path("gtap7_3x3", 0)
    assert GMS is not None and path == GMS


@needs_gms
def test_check_is_clean():
    res = _run_check()
    entities = sorted(v["entity"] for v in res["violations"])
    assert entities == [], (
        f"expected no violation, got {entities} "
        f"(false positives from name-heuristic pairing not eliminated)"
    )
    assert res["status"] == "clean"


@needs_gms
def test_pseq_resolves_to_active_homolog_not_legacy_eq_ps():
    """pseq must resolve to eq_xs (active), not eq_ps (deactivated legacy)."""
    res = _run_check()
    # pseq must NOT appear as a violation
    assert "pseq" not in {v["entity"] for v in res["violations"]}


@needs_gms
def test_fixed_var_with_deactivated_eq_is_not_a_violation():
    """dintxeq/mintxeq: eq deactivated but var fixed -> valid exogenous pairing."""
    res = _run_check()
    bad = {v["entity"] for v in res["violations"]} & {"dintxeq", "mintxeq"}
    assert not bad, f"fixed-var wedges wrongly flagged: {bad}"


# MAR-219: the Python side is the case the MCP gate builds, read at the solve of
# the period on the core driver, so only that period's rows and cells count.
def _two_period_model():
    import pyomo.environ as pyo

    m = pyo.ConcreteModel()
    m.t = pyo.Set(initialize=["check", "shock"])
    m.pf = pyo.Var(m.t, initialize=1.0)
    m.eq_pf = pyo.Constraint(m.t, rule=lambda m, t: m.pf[t] == 1.0)
    m.eq_pf["shock"].deactivate()
    m.pf["shock"].fix()
    return m


def test_family_state_counts_only_the_period():
    from diff_mcp_pairing import _py_family_state

    m = _two_period_model()
    assert _py_family_state(m, "pfeq", "pf", "check") == ("eq_pf", 1, 1, (0, 1))
    assert _py_family_state(m, "pfeq", "pf", "shock") == ("eq_pf", 0, 1, (1, 1))


@needs_gms
@pytest.mark.needs_path
@pytest.mark.needs_gdxdump
@pytest.mark.parametrize("period", ["check", "shock"])
def test_the_gate_case_is_clean(period):
    import argparse

    from diff_mcp_pairing import _work

    res = _work(
        argparse.Namespace(
            dataset="gtap7_3x3",
            period=period,
            ifsub=0,
            model_name="gtap",
            apply_closure=True,
            mode="altertax",
            gdx=None,
        )
    )
    assert res["status"] == "clean", res["headline"]
    assert res["meta"]["case"] == "mcp_gate"


def test_a_gams_variable_is_read_under_its_python_name():
    # GAMS peq.p: Python's p is p_rai; fixed with eq_peq off is a valid pairing.
    import pyomo.environ as pyo
    from diff_mcp_pairing import _var_fixed_frac

    m = pyo.ConcreteModel()
    m.p_rai = pyo.Var(["check"], initialize=1.0)
    m.p_rai["check"].fix()
    assert _var_fixed_frac(m, "p", "check") == (1, 1)
