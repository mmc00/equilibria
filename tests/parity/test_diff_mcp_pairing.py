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
@pytest.mark.needs_gdxdump  # the gate case seeds from the GAMS GDX through it
def test_check_is_clean():
    res = _run_check()
    entities = sorted(v["entity"] for v in res["violations"])
    assert entities == [], (
        f"expected no violation, got {entities} "
        f"(false positives from name-heuristic pairing not eliminated)"
    )
    assert res["status"] == "clean"


@needs_gms
@pytest.mark.needs_gdxdump  # the gate case seeds from the GAMS GDX through it
def test_pseq_resolves_to_active_homolog_not_legacy_eq_ps():
    """pseq must resolve to eq_xs (active), not eq_ps (deactivated legacy)."""
    res = _run_check()
    # pseq must NOT appear as a violation
    assert "pseq" not in {v["entity"] for v in res["violations"]}


@needs_gms
@pytest.mark.needs_gdxdump  # the gate case seeds from the GAMS GDX through it
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


def test_the_state_is_read_at_the_solve_not_after_the_run(monkeypatch):
    """The driver fixes every cell of a period after its solve (freeze_period) and
    the next period switches its rows off (freeze_inactive). The Pyomo model is
    shared, so the state must be copied at the solve, or check would always read
    "every row off, every cell fixed" and the layer could never fail."""
    import types

    import _core_run
    import diff_mcp_pairing as DMP
    import seed_and_solve

    from equilibria.templates.gtap import recipe

    m = _two_period_model()
    m.eq_pf["shock"].activate()
    m.pf["shock"].unfix()

    def fake_solve_on_core(*_a, solver, **_k):
        for per in ("check", "shock"):
            solver(
                types.SimpleNamespace(pyomo=m, staged=types.SimpleNamespace(period=per))
            )
            m.pf[per].fix()  # freeze_period
            m.eq_pf[per].deactivate()  # the next freeze_inactive

    monkeypatch.setattr(_core_run, "solve_on_core", fake_solve_on_core)
    monkeypatch.setattr(recipe, "_by_backend", lambda model: "ok")
    monkeypatch.setattr(seed_and_solve, "_case", lambda *a: (None, None, "ROW"))
    state = DMP.gate_model_at_solve("gtap7_3x3", "check", "altertax", 0, Path("x"))
    assert DMP._py_family_state(state, "pfeq", "pf", "check") == ("eq_pf", 1, 1, (0, 1))


def test_a_recalibrated_nd_row_counts_as_its_family():
    """check/shock swap eq_nd[r,a,t] for _eq_nd_recal_<r>_<a>_<t> (pyomo_ops)."""
    import pyomo.environ as pyo
    from diff_mcp_pairing import _freeze_state, _py_family_state

    m = pyo.ConcreteModel()
    m.nd = pyo.Var([("EU_28", "Food", "check")], initialize=1.0)
    m.eq_nd = pyo.Constraint(
        [("EU_28", "Food", "check")], rule=lambda m, *i: m.nd[i] == 1.0
    )
    m.eq_nd["EU_28", "Food", "check"].deactivate()
    m.add_component(
        "_eq_nd_recal_EU_28_Food_check",
        pyo.Constraint(expr=m.nd["EU_28", "Food", "check"] == 1.0),
    )
    state = _freeze_state(m, "check")
    assert _py_family_state(state, "ndeq", "nd", "check") == ("eq_nd", 1, 1, (0, 1))
