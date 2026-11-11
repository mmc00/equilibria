"""F3.5 base-calibrated mode — adopt the settled (check) point as base so the
shock response matches GEMPACK instead of the check-contaminated GAMS path.

2026-11-01: the contamination was the FBEP sign. The old GAMS fixtures injected
the subsidy with the HAR's sign, so the base was no equilibrium and the check
moved EU_28 Land from 1.0 to 0.845. With the data's sign (the fbep=header
reference, Python's va_subsidy_basis="gempack") the base is an equilibrium: the
settle leaves it where it is, and GAMS's raw shock response already matches
GEMPACK. These tests now pin that; base_calibrated stays a harmless no-op here.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from equilibria._local_refs import path_capi_src
from equilibria.sources import dataset_dir, reference

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts" / "gtap"))
_PATH_CAPI = path_capi_src()
if _PATH_CAPI.exists() and str(_PATH_CAPI) not in sys.path:
    sys.path.insert(0, str(_PATH_CAPI))

# the GAMS MCP reference with FBEP split out; Python takes the data's subsidy
# sign to match it (the default "gams" basis keeps the old sign, #191)
REF_GDX = (
    reference("gtap7_3x3", "gams", form="mcp", mode="gtap", ifsub=1, fbep="header")
    / "out.gdx"
)
# GEMPACK con el MISMO cierre que estos tests (capFix = RORDELTA 0). El nombre sin
# sufijo (sl4dump_gtap7_3x3_tm10.har) es capFlex (RORDELTA 1) desde 06dd878, que
# regenero los fixtures con el cierre estandar: comparado contra capFix, el match de
# cantidades cayo de 96,3% a 73,2% sin que cambiara el modelo (git bisect, 2026-10-04).
GEMPACK_CAPFIX = (
    reference("gtap7_3x3", "gempack", shock=10, steps="8-16-32", closure="capFix")
    / "sl4dump.har"
)


def _has_solver():
    return importlib.util.find_spec("path_capi_python") is not None


def _ref_slice(period: str) -> dict[tuple[str, tuple], float]:
    from _diff_core import gams_levels, split_t

    out: dict[tuple[str, tuple], float] = {}
    for vn in ("pft", "xf", "xp", "va", "nd", "xda", "xma", "xaa"):
        # Un simbolo ausente del GDX es legitimo (se salta); que NO se pueda
        # leer el GDX no lo es. Sin esta distincion, un gdxdump ausente dejaba
        # el dict vacio y el test moria con un KeyError opaco tres lineas mas
        # abajo, en vez de decir que falta la herramienta.
        try:
            g = gams_levels(REF_GDX, vn)
        except FileNotFoundError as exc:
            raise RuntimeError(
                f"no se pudo leer {REF_GDX} ({exc}); gdxdump es imprescindible "
                "para este test — marcalo con @pytest.mark.needs_gdxdump"
            ) from exc
        except Exception:
            continue
        for fk, val in g.items():
            body, t = split_t(fk)
            if t == period:
                out[(vn, tuple(str(x) for x in body))] = val
    return out


@pytest.mark.needs_gdxdump
def test_settled_base_land_response_matches_gempack():
    """On the reference: the check stays at the base (no contamination; the old
    fixture had it at 0.845), and the raw shock response of the Land price is
    GEMPACK's (pfe[Land,Food,EU_28] = -3.6126% in the capFix fixture)."""
    base = _ref_slice("base")
    check = _ref_slice("check")
    shock = _ref_slice("shock")
    k = ("pft", ("EU_28", "Land"))
    assert check[k] == pytest.approx(base[k], rel=1e-5), (
        f"check moved off the base: {base[k]} -> {check[k]}"
    )
    raw_resp = 100.0 * (shock[k] - base[k]) / base[k]
    assert raw_resp == pytest.approx(-3.6126, abs=0.05), (
        f"GAMS land response {raw_resp:.3f}% is not GEMPACK's -3.6126%"
    )


def _load_params(dataset="gtap7_3x3"):
    from equilibria.templates.gtap import GTAPParameters

    d = dataset_dir(dataset)
    p = GTAPParameters()
    p.load_from_har(
        basedata_path=d / "basedata.har",
        sets_path=d / "sets.har",
        default_path=d / "default.prm",
        baserate_path=d / "baserate.har",
    )
    p.set_va_subsidy_basis("gempack")
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
        va_subsidy_basis="gempack",
    )


@pytest.mark.skipif(not _has_solver(), reason="PATH solver not available")
@pytest.mark.needs_gdxdump
def test_calibrate_base_returns_settled_land_price():
    """calibrate_base runs the settle solve and returns the CHECK-period point;
    with the data's subsidy sign the base is already an equilibrium, so the
    settled Land price is the data's 1.0 (it was 0.845 with the old sign)."""
    from equilibria.blocks.gtap.factor import FactorBlock

    p = _load_params()
    rr = list(p.sets.r)[-1]
    fb = FactorBlock(sets=p.sets, params=p)
    settled = fb.calibrate_base(p, p.sets, _base_closure(p), rr, ref_gdx=REF_GDX)
    pft = settled["pft"][("EU_28", "Land")]
    assert pft == pytest.approx(1.0, abs=1e-5), f"settled Land price off: {pft}"


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
    assert m_cal._settled_seed["pft"][("EU_28", "Land")] == pytest.approx(1.0, abs=1e-5)


@pytest.mark.skipif(not _has_solver(), reason="PATH solver not available")
def test_base_calibrated_shock_response_is_clean():
    """End-to-end: base-calibrated → base seeded to the settled point (the
    data's 1.0 with the data's subsidy sign), shock land response ~-3.6% (near
    GEMPACK), NOT -18%, all solved code=1, no check phase."""
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
    assert base == pytest.approx(1.0, abs=1e-5), f"base not settled: {base}"
    resp = 100.0 * (shock - base) / base
    assert -8.0 < resp < 0.0, f"land response not clean: {resp:.2f}%"


@pytest.mark.skipif(not _has_solver(), reason="PATH solver not available")
def test_calibrated_land_response_beats_default_vs_gempack():
    """The base-calibrated land response is far closer to GEMPACK's number
    than the check-contaminated raw path (~-18%). GEMPACK number read from the
    committed sl4dump fixture (pfe[Land,Food,EU_28])."""
    # GEMPACK's own number from the committed fixture
    from gempack_reference import sl4_levels
    from pyomo.environ import value as V

    from equilibria.templates.gtap.gtap_block_model import build_block_model
    from equilibria.templates.gtap.gtap_multiperiod_driver import solve_multiperiod

    sl4 = GEMPACK_CAPFIX
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
    # pfe[Land,Food,EU_28] del fixture capFix (el cierre de este test). El -2.681
    # original era del fixture anterior a 06dd878 (cierre con el swap de dpsave); el
    # capFlex sin sufijo da -3.618.
    assert gem is not None and gem == pytest.approx(-3.6126, abs=1e-3)

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


@pytest.mark.skipif(not _has_solver(), reason="PATH solver not available")
@pytest.mark.needs_gdxdump
def test_overall_quantity_match_vs_gempack_is_clean_either_way():
    """The OVERALL against-GEMPACK quantity match (all Q_TO_VAR vars, ~190 cells)
    is clean with or without base calibration: 99.5% within 1pp both. The old
    ~76% -> ~96% "lift" from calibrating the base was the FBEP sign, like the
    check contamination above; with the old sign the default path falls to ~76%."""
    sys.path.insert(0, str(ROOT / "scripts" / "gtap"))
    from measure_gempack_blocks import _quantity_match

    gem = GEMPACK_CAPFIX

    default = _quantity_match("gtap7_3x3", False, REF_GDX, gem)
    calib = _quantity_match("gtap7_3x3", True, REF_GDX, gem)
    assert "within_1pp_pct" in default and "within_1pp_pct" in calib
    assert default["within_1pp_pct"] >= 94.0, f"default match too low: {default}"
    assert calib["within_1pp_pct"] >= 94.0, f"calibrated match too low: {calib}"
    assert abs(calib["within_1pp_pct"] - default["within_1pp_pct"]) <= 1.0
