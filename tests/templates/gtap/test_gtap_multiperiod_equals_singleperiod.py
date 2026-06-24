"""Fidelity gate: for the pure gtap model, MP shock == SP shock.

gtap does not recalibrate (no betaCal), so the multi-period check is idempotent
and the multi-period shock must equal the single-period shock. This equality is
the proof that licenses deleting the single-period path.

LOCAL-ONLY: needs the PATH solver (path_capi_python) + the dataset HAR.  It SKIPS
(never fails) when either is absent, so it is safe on CI (no self-hosted runner /
no solver there) -- mirror of test_altertax_multiperiod_parity.py.  Run by hand on
a machine that has PATH + datasets/gtap7_3x3 to validate the MP==SP equality:

    uv run pytest tests/templates/gtap/test_gtap_multiperiod_equals_singleperiod.py -v
"""
from __future__ import annotations

import copy
import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
DATASETS_DIR = ROOT / "datasets"

sys.path.insert(0, str(ROOT / "scripts/gtap"))

# Local PATH solver lives outside the venv; add its src dir so find_spec can
# locate it.  This does NOT affect CI (the dir won't exist there, find_spec will
# still return None, and the tests SKIP as intended).
_PATH_CAPI_SRC = Path("/Users/marmol/proyectos/path-capi-python/src")
if _PATH_CAPI_SRC.exists() and str(_PATH_CAPI_SRC) not in sys.path:
    sys.path.insert(0, str(_PATH_CAPI_SRC))

# Exclusion sets (denominator), identical to test_altertax_multiperiod_parity.py
# lines 49-57.
SKIP = {"walras", "ev", "cv", "uh", "u", "ug", "us"}
RF = {
    "pfa", "pfy", "pm", "pmcif", "pefob", "pwmg", "pp", "pdp", "pmp",
    "xwmg", "xmgm", "lambdamg", "imptx", "exptx",
}
ALIAS = {
    "xa": "xaa", "xd": "xda", "xm": "xma", "pp": "pp_rai", "p": "p_rai",
    "ytaxInd": "ytax_ind", "ytaxind": "ytax_ind",
}


def _has_path_solver() -> bool:
    """The PATH MCP solver lives in a local package, absent on ubuntu-latest."""
    return importlib.util.find_spec("path_capi_python") is not None


def _gtap_closure():
    from equilibria.templates.gtap.gtap_contract import GTAPClosureConfig

    return GTAPClosureConfig(
        name="base", closure_type="MCP", capital_mobility="sluggish",
        fix_endowments=False, fix_taxes=False, fix_technology=False,
        if_sub=False, numeraire="pnum",
    )


def _load_params(dataset: str):
    from equilibria.templates.gtap import GTAPParameters

    d = DATASETS_DIR / dataset
    p = GTAPParameters()
    p.load_from_har(
        basedata_path=d / "basedata.har",
        sets_path=d / "sets.har",
        default_path=d / "default.prm",
        baserate_path=d / "baserate.har",
    )
    return p


def _solve_mp_gtap_shock(dataset: str):
    """Build + solve the pure-gtap MP model. Returns (m, codes_dict)."""
    from equilibria.templates.gtap.gtap_model_multiperiod import (
        GTAPMultiPeriodModel,
        PERIODS,
    )
    from equilibria.templates.gtap.gtap_multiperiod_driver import solve_multiperiod

    p = _load_params(dataset)
    rr = list(p.sets.r)[-1]
    gc = _gtap_closure()
    mp = GTAPMultiPeriodModel(p.sets, p, gc, residual_region=rr)
    m = mp.build_sets()
    mp.build_vars(m)
    for per in PERIODS:
        mp.build_equations_intra(m, per)
    mp.build_equations_fisher(m)
    m._residual_region = rr
    res = solve_multiperiod(
        m, p, gc, skip_base_solve=False, mute_welfare=True,
        seed_from_prior=True, holdfix_cd=False, mode="gtap",
    )
    codes = {k: res[k]["code"] for k in res}
    return m, codes


def _solve_sp_gtap_shock(dataset: str):
    """In-process single-period gtap shock: build -> tm_pct 10% -> PATH solve.

    Returns {varname_lower: {idx_tuple: value}} for populated (non-None) Var cells.
    No CLI, no GDX -- the same machinery the SP path uses internally
    (GTAPModelEquations.build_model + _run_path_capi_nonlinear_full), so the proof
    is unaffected by deleting the validate-shock CLI subcommand later.
    """
    from equilibria.templates.gtap import GTAPModelEquations
    from equilibria.templates.gtap.gtap_multiperiod_driver import _apply_imptx_shock
    from pyomo.environ import Var, value as V
    import run_gtap  # scripts/gtap on sys.path

    p = _load_params(dataset)
    rr = list(p.sets.r)[-1]
    p_shock = copy.deepcopy(p)
    _apply_imptx_shock(p_shock, factor=0.10)  # same tm_pct shock the driver uses
    gc = _gtap_closure()
    m_sp = GTAPModelEquations(
        p_shock.sets, p_shock, gc, residual_region=rr
    ).build_model()
    # closure is KEYWORD-ONLY via closure_config=; equation_scaling=True is mandatory
    # (without it the base lands code=2 -- project constraint).
    run_gtap._run_path_capi_nonlinear_full(
        m_sp, p_shock, closure_config=gc, equation_scaling=True,
    )

    out: dict[str, dict] = {}
    for v in m_sp.component_objects(Var, active=True):
        cells: dict = {}
        for idx in v:
            vd = v[idx]
            if vd.value is None:
                continue
            try:
                key = idx if isinstance(idx, tuple) else (idx,)
                cells[key] = float(V(vd))
            except Exception:
                continue
        if cells:
            out[v.name.lower()] = cells
    return out


@pytest.mark.skipif(not _has_path_solver(), reason="PATH solver unavailable")
def test_gtap_mp_shock_converges_3x3():
    """All 3 MP periods of the pure-gtap shock converge (code == 1)."""
    dataset = "gtap7_3x3"
    if not (DATASETS_DIR / dataset / "basedata.har").exists():
        pytest.skip(f"dataset HAR missing for {dataset}")
    _m, codes = _solve_mp_gtap_shock(dataset)
    assert all(c == 1 for c in codes.values()), (
        f"not all periods converged: {codes}"
    )


@pytest.mark.skipif(not _has_path_solver(), reason="PATH solver unavailable")
def test_gtap_mp_shock_equals_sp_shock_3x3():
    """MP shock values == SP shock values for pure gtap (the fidelity proof)."""
    dataset = "gtap7_3x3"
    if not (DATASETS_DIR / dataset / "basedata.har").exists():
        pytest.skip(f"dataset HAR missing for {dataset}")
    from pyomo.environ import value as V

    m, codes = _solve_mp_gtap_shock(dataset)
    assert all(c == 1 for c in codes.values()), f"MP not converged: {codes}"
    sp = _solve_sp_gtap_shock(dataset)

    tot = match = 0
    worst: list[tuple[float, str, tuple, float, float]] = []
    for vn, cells in sp.items():
        if vn in SKIP or vn in RF:
            continue
        mp_name = ALIAS.get(vn, vn)
        pv = getattr(m, mp_name, None)
        if pv is None:
            pv = getattr(m, vn, None)
        if pv is None:
            continue
        for sp_idx, sp_val in cells.items():
            # MP cell = SP idx + trailing "shock" period label.
            mp_idx = (*sp_idx, "shock")
            try:
                mp_val = float(V(pv[mp_idx]))
            except Exception:
                continue
            tot += 1
            d_abs = abs(mp_val - sp_val)
            rel = d_abs / abs(sp_val) if abs(sp_val) > 1e-12 else (
                0.0 if d_abs < 1e-6 else 9e9
            )
            if d_abs <= 1e-6 or rel <= 1e-2:
                match += 1
            else:
                worst.append((rel, vn, sp_idx, sp_val, mp_val))

    match_pct = 100.0 * match / max(tot, 1)
    worst.sort(reverse=True)
    worst_str = "; ".join(
        f"{vn}{idx}: SP={sv:.5g} MP={mv:.5g} rel={r:.3g}"
        for r, vn, idx, sv, mv in worst[:10]
    )
    assert tot > 0, "no comparable cells found"
    assert match_pct >= 99.5, (
        f"MP!=SP for gtap7_3x3: {match_pct:.2f}% over {tot} cells. "
        f"worst: {worst_str}"
    )
