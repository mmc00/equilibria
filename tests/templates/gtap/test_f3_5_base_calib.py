"""F3.5 base-calibrated mode — adopt the settled (check) point as base so the
shock response matches GEMPACK instead of the check-contaminated GAMS path."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts" / "gtap"))
_PATH_CAPI = Path("/Users/marmol/proyectos/path-capi-python/src")
if _PATH_CAPI.exists() and str(_PATH_CAPI) not in sys.path:
    sys.path.insert(0, str(_PATH_CAPI))

DATASETS = ROOT / "datasets"
REF_GDX = ROOT / "tests/fixtures/gtap7/gtap7_3x3/out_gtap_shock_ifsub1.gdx"


def _has_solver():
    return importlib.util.find_spec("path_capi_python") is not None


def _ref_slice(period: str) -> dict[tuple[str, tuple], float]:
    from _diff_core import gams_levels, split_t

    out: dict[tuple[str, tuple], float] = {}
    for vn in ("pft", "xf", "xp", "va", "nd", "xda", "xma", "xaa"):
        try:
            g = gams_levels(REF_GDX, vn)
        except Exception:
            continue
        for fk, val in g.items():
            body, t = split_t(fk)
            if t == period:
                out[(vn, tuple(str(x) for x in body))] = val
    return out


def test_settled_base_land_response_matches_gempack():
    """The mechanism, on the reference: shock-vs-SETTLED-base land price is small
    (~-3%, near GEMPACK's -2.68%); shock-vs-RAW-base is the -18% contaminated path."""
    base = _ref_slice("base")
    check = _ref_slice("check")
    shock = _ref_slice("shock")
    k = ("pft", ("EU_28", "Land"))
    raw_resp = 100.0 * (shock[k] - base[k]) / base[k]
    settled_resp = 100.0 * (shock[k] - check[k]) / check[k]
    assert raw_resp < -15.0, (
        f"expected contaminated raw path <-15%, got {raw_resp:.2f}%"
    )
    assert -8.0 < settled_resp < 0.0, (
        f"expected clean settled response, got {settled_resp:.2f}%"
    )
    assert abs(settled_resp - (-2.68)) < abs(raw_resp - (-2.68))


def _load_params(dataset="gtap7_3x3"):
    from equilibria.templates.gtap import GTAPParameters

    d = DATASETS / dataset
    p = GTAPParameters()
    p.load_from_har(
        basedata_path=d / "basedata.har",
        sets_path=d / "sets.har",
        default_path=d / "default.prm",
        baserate_path=d / "baserate.har",
    )
    return p


def _base_closure(p):
    from equilibria.templates.gtap.gtap_contract import GTAPClosureConfig

    return GTAPClosureConfig(
        name="base",
        closure_type="MCP",
        capital_mobility="sluggish",
        fix_endowments=False,
        fix_taxes=False,
        fix_technology=False,
        if_sub=False,
        numeraire="pnum",
    )


@pytest.mark.skipif(not _has_solver(), reason="PATH solver not available")
def test_calibrate_base_returns_settled_land_price():
    """calibrate_base runs the settle solve and returns the CHECK-period point;
    the Land price in it is the settled ~0.845, not the raw 1.0."""
    from equilibria.blocks.gtap.factor import FactorBlock

    p = _load_params()
    rr = list(p.sets.r)[-1]
    fb = FactorBlock(sets=p.sets, params=p)
    settled = fb.calibrate_base(p, p.sets, _base_closure(p), rr, ref_gdx=REF_GDX)
    pft = settled["pft"][("EU_28", "Land")]
    assert pft == pytest.approx(0.84476, abs=5e-3), f"settled Land price off: {pft}"


@pytest.mark.skipif(not _has_solver(), reason="PATH solver not available")
def test_composer_stamps_settled_seed():
    """build_block_model(base_calibrated=True) runs the settle and stamps the
    settled point + the flag; base_calibrated=False leaves both empty (default)."""
    from equilibria.templates.gtap.gtap_block_model import build_block_model

    p = _load_params()
    rr = list(p.sets.r)[-1]
    m_raw, _ = build_block_model(p, p.sets, _base_closure(p), rr, base_calibrated=False)
    assert m_raw._base_calibrated is False
    assert m_raw._settled_seed is None

    m_cal, _ = build_block_model(p, p.sets, _base_closure(p), rr, base_calibrated=True)
    assert m_cal._base_calibrated is True
    assert m_cal._settled_seed["pft"][("EU_28", "Land")] == pytest.approx(
        0.84476, abs=5e-3
    )


@pytest.mark.skipif(not _has_solver(), reason="PATH solver not available")
def test_base_calibrated_shock_response_is_clean():
    """End-to-end: base-calibrated → base seeded to settled ~0.845, shock land
    response ~-3% (near GEMPACK), NOT -18%, all solved code=1, no check phase."""
    from pyomo.environ import value as V

    from equilibria.templates.gtap.gtap_block_model import build_block_model
    from equilibria.templates.gtap.gtap_multiperiod_driver import solve_multiperiod

    p = _load_params()
    rr = list(p.sets.r)[-1]
    gc = _base_closure(p)
    m, mp = build_block_model(p, p.sets, gc, rr, base_calibrated=True)
    # skip_base_solve=True: the settled point IS the given base (already an
    # equilibrium from the settle solve), like the faithful-to-GAMS base.
    res = solve_multiperiod(
        m, p, gc, ref_gdx=REF_GDX, skip_base_solve=True, mode="gtap"
    )
    assert res["base"]["code"] == 1
    assert res["shock"]["code"] == 1
    assert res.get("check") is None
    base = float(V(m.pft["EU_28", "Land", "base"]))
    shock = float(V(m.pft["EU_28", "Land", "shock"]))
    assert base == pytest.approx(0.84476, abs=1e-2), f"base not settled: {base}"
    resp = 100.0 * (shock - base) / base
    assert -8.0 < resp < 0.0, f"land response not clean: {resp:.2f}%"


@pytest.mark.skipif(not _has_solver(), reason="PATH solver not available")
def test_calibrated_land_response_beats_default_vs_gempack():
    """The base-calibrated land response (~-3%) is far closer to GEMPACK's -2.68%
    than the check-contaminated raw path (~-18%). GEMPACK number read from the
    committed sl4dump fixture (pfe[Land,Food,EU_28])."""
    # GEMPACK's own number from the committed fixture
    from gempack_reference import sl4_levels
    from pyomo.environ import value as V

    from equilibria.templates.gtap.gtap_block_model import build_block_model
    from equilibria.templates.gtap.gtap_multiperiod_driver import solve_multiperiod

    sl4 = ROOT / "tests/fixtures/gtap7_gempack/sl4dump_gtap7_3x3_tm10.har"
    if not sl4.exists():
        pytest.skip(f"GEMPACK fixture missing: {sl4}")
    gem = None
    for _var in ("pfe", "pes"):
        try:
            for _k, _v in sl4_levels(str(sl4), _var).items():
                if "Land" in _k and "Food" in _k and "EU_28" in _k:
                    gem = _v
                    break
        except Exception:
            continue
        if gem is not None:
            break
    assert gem is not None and gem == pytest.approx(-2.681, abs=1e-2)

    p = _load_params()
    rr = list(p.sets.r)[-1]
    gc = _base_closure(p)
    m, _ = build_block_model(p, p.sets, gc, rr, base_calibrated=True)
    solve_multiperiod(m, p, gc, ref_gdx=REF_GDX, skip_base_solve=True, mode="gtap")
    b = float(V(m.pft["EU_28", "Land", "base"]))
    s = float(V(m.pft["EU_28", "Land", "shock"]))
    cal_resp = 100.0 * (s - b) / b

    RAW_PATH = -18.09  # shock vs raw-1.0 base (the check-contaminated GAMS path)
    assert abs(cal_resp - gem) < 6.0, (
        f"calibrated response {cal_resp:.2f}% not near GEMPACK {gem}"
    )
    assert abs(cal_resp - gem) < abs(RAW_PATH - gem)
