"""Phase C.3 tests — solve_sequential multi-instance loop.

Test scenarios:
1. Single-period parity: t_set=("base",) bit-identical to existing single solve.
2. Two-period no-shock identity: t_set=("base","check") converges, yc unchanged.
3. PeriodResult dataclass shape.
4. Mock-solver smoke test (no PATH needed).
"""
from __future__ import annotations

import contextlib
import json
import os
import sys
from pathlib import Path

import pytest
from pyomo.environ import ConcreteModel, Param, Reals, Set, Var, value as pyo_value

os.environ['EQUILIBRIA_GTAP_RRES'] = 'USA'
sys.path.insert(0, str(Path('src').resolve()))
sys.path.insert(0, str(Path('scripts/gtap').resolve()))

from equilibria.templates.gtap.gtap_solve_sequential import (
    PeriodResult,
    solve_sequential,
)
from equilibria.templates.gtap.gtap_iterloop import LAGGED_VARS


GDX_15X10 = Path('datasets/gtap7_15x10/GDX/basedata.gdx').resolve()
BASELINE_15X10 = Path('tests/gtap/baselines/baseline_15x10_base.json').resolve()


# ---------------------------------------------------------------------------
# Test #4 — mock-solver smoke. Runs without GDX or PATH.
# ---------------------------------------------------------------------------


def _make_mock_params_and_contract():
    """Build a fake params + closure pair where build_model returns a tiny
    ConcreteModel that has every LAGGED_VARS name as an indexed Var on a
    common 1-element index set. This lets us exercise the fix_lagged_state
    call path without depending on GTAP-specific structure.
    """
    class _MockSets:
        pass

    class _MockClosure:
        pass

    class _MockParams:
        sets = _MockSets()

    # We need to monkey-patch GTAPModelEquations inside the module so the
    # function builds our mock model.
    def _build_one() -> ConcreteModel:
        m = ConcreteModel()
        m.r = Set(initialize=["USA"])
        m.a = Set(initialize=["agr"])
        m.i = Set(initialize=["agr"])
        m.aa = Set(initialize=["agr", "hh"])
        m.f = Set(initialize=["lab"])
        m.rp = Set(initialize=["USA"])
        m.m_set = Set(initialize=["trp"])
        m.axp = Var(m.r, m.a, within=Reals, initialize=1.0)
        m.lambdand = Var(m.r, m.a, within=Reals, initialize=1.0)
        m.lambdava = Var(m.r, m.a, within=Reals, initialize=1.0)
        m.aioall = Var(m.r, m.i, m.a, within=Reals, initialize=1.0)
        m.lambdaf = Var(m.r, m.f, m.a, within=Reals, initialize=1.0)
        m.pf = Var(m.r, m.f, m.a, within=Reals, initialize=1.0)
        m.xf = Var(m.r, m.f, m.a, within=Reals, initialize=1.0)
        m.pa = Var(m.r, m.i, m.aa, within=Reals, initialize=1.0)
        m.xaa = Var(m.r, m.i, m.aa, within=Reals, initialize=1.0)
        m.pe = Var(m.r, m.i, m.rp, within=Reals, initialize=1.0)
        m.pefob = Var(m.r, m.i, m.rp, within=Reals, initialize=1.0)
        m.pmcif = Var(m.rp, m.i, m.r, within=Reals, initialize=1.0)
        m.pm = Var(m.rp, m.i, m.r, within=Reals, initialize=1.0)
        m.xw = Var(m.r, m.i, m.rp, within=Reals, initialize=1.0)
        m.ptmg = Var(m.m_set, within=Reals, initialize=1.0)
        m.psave = Var(m.r, within=Reals, initialize=1.0)
        m.pi = Var(m.r, within=Reals, initialize=1.0)
        m.uh = Var(m.r, within=Reals, initialize=1.0)
        m.pabs = Var(m.r, within=Reals, initialize=1.0)
        m.pfact = Var(m.r, within=Reals, initialize=1.0)
        m.pwfact = Var(within=Reals, initialize=1.0)
        m.gdpmp = Var(m.r, within=Reals, initialize=1.0)
        m.rgdpmp = Var(m.r, within=Reals, initialize=1.0)
        m.pgdpmp = Var(m.r, within=Reals, initialize=1.0)
        m.pmuv = Param(within=Reals, initialize=1.0, mutable=True)
        return m

    return _MockParams(), _MockClosure(), _build_one


def test_mock_solver_smoke(monkeypatch):
    """Stub solver_fn + monkey-patched builder. Verify wiring across 3 periods."""
    import equilibria.templates.gtap.gtap_solve_sequential as mod

    params, closure, build_one = _make_mock_params_and_contract()

    class _FakeBuilder:
        def __init__(self, *a, **kw):
            pass
        def build_model(self):
            return build_one()

    monkeypatch.setattr(mod, "GTAPModelEquations", _FakeBuilder)

    fix_calls: list[tuple] = []
    real_fix = mod.fix_lagged_state

    def _spy_fix(new, prev, *a, **kw):
        fix_calls.append((id(new), id(prev)))
        return real_fix(new, prev, *a, **kw)

    monkeypatch.setattr(mod, "fix_lagged_state", _spy_fix)

    seen_models: list[ConcreteModel] = []

    def _stub_solver(model, params_arg, *, closure_config, equation_scaling,
                     path_capi_convergence_tol):
        seen_models.append(model)
        assert isinstance(model, ConcreteModel)
        # Seed every Var so prev->new copy has values
        for name in LAGGED_VARS:
            v = getattr(model, name, None)
            if v is None or not isinstance(v, Var):
                continue
            if v.is_indexed():
                for idx in v:
                    v[idx].value = 1.0
            else:
                v.value = 1.0
        return {"residual": 0.0}

    out = solve_sequential(
        params, closure,
        t_set=("base", "t1", "t2"),
        solver_fn=_stub_solver,
    )

    assert len(out) == 3
    assert set(out.keys()) == {"base", "t1", "t2"}
    assert len(seen_models) == 3
    # fix_lagged_state called once each for t1, t2 (not for base)
    assert len(fix_calls) == 2
    for p in ("base", "t1", "t2"):
        assert isinstance(out[p], PeriodResult)
        assert out[p].period == p
        assert isinstance(out[p].model, ConcreteModel)
        assert out[p].residual == 0.0
        assert isinstance(out[p].solver_metadata, dict)


def test_period_result_shape():
    """PeriodResult is a dataclass with the documented fields."""
    m = ConcreteModel()
    pr = PeriodResult(period="base", model=m, residual=1e-10, solver_metadata={"foo": 1})
    assert pr.period == "base"
    assert pr.model is m
    assert pr.residual == 1e-10
    assert pr.solver_metadata == {"foo": 1}


# ---------------------------------------------------------------------------
# Tests #1 & #2 — real solver. Slow. Need GDX + PATH.
# ---------------------------------------------------------------------------


def _load_baseline():
    if not BASELINE_15X10.exists():
        pytest.skip(f"baseline file missing: {BASELINE_15X10}")
    return json.loads(BASELINE_15X10.read_text())


def _real_solver_fn():
    """Lazy import of the production solver."""
    from run_gtap import _run_path_capi_nonlinear_full
    return _run_path_capi_nonlinear_full


def _real_setup():
    if not GDX_15X10.exists():
        pytest.skip(f"GDX file missing: {GDX_15X10}")
    from equilibria.templates.gtap.gtap_parameters import GTAPParameters
    from equilibria.templates.gtap.gtap_contract import default_gtap_contract
    contract = default_gtap_contract()
    p = GTAPParameters()
    with contextlib.redirect_stdout(sys.stderr):
        p.load_from_gdx(GDX_15X10)
    return p, contract


@pytest.mark.slow
def test_single_period_parity():
    """solve_sequential(("base",)) ≡ single-period solve. BIT-IDENTICAL."""
    baseline = _load_baseline()
    p, contract = _real_setup()
    solver = _real_solver_fn()

    with contextlib.redirect_stdout(sys.stderr):
        out = solve_sequential(
            p, contract.closure,
            t_set=("base",),
            solver_fn=solver,
        )

    assert "base" in out
    pr = out["base"]
    assert pr.period == "base"
    assert isinstance(pr.model, ConcreteModel)

    # Residual must match the snapshot within 1e-13 (bit-identical for our purposes).
    expected_res = baseline["residual"]
    assert abs(pr.residual - expected_res) < 1e-13, (
        f"residual drift: got {pr.residual!r}, expected {expected_res!r}"
    )

    # Verify a sample of USA endogenous values
    yc = float(pyo_value(pr.model.yc['USA']))
    assert abs(yc - baseline['usa']['yc']) < 1e-10, (
        f"yc[USA] drift: got {yc!r}, expected {baseline['usa']['yc']!r}"
    )


@pytest.mark.slow
def test_two_period_no_shock_identity():
    """t_set=("base","check") with NO shock — check period must match base."""
    p, contract = _real_setup()
    solver = _real_solver_fn()

    with contextlib.redirect_stdout(sys.stderr):
        out = solve_sequential(
            p, contract.closure,
            t_set=("base", "check"),
            solver_fn=solver,
        )

    assert set(out.keys()) == {"base", "check"}
    base_pr = out["base"]
    check_pr = out["check"]

    assert check_pr.residual < 1e-8, (
        f"check period did not converge: residual={check_pr.residual!r}"
    )

    # yc[USA] should be ~identical between base and check (no shock = no change)
    yc_base = float(pyo_value(base_pr.model.yc['USA']))
    yc_check = float(pyo_value(check_pr.model.yc['USA']))
    diff = abs(yc_base - yc_check)
    assert diff < 1e-6, (
        f"yc[USA] differs between base and check despite no shock: "
        f"base={yc_base!r}, check={yc_check!r}, diff={diff!r}"
    )
