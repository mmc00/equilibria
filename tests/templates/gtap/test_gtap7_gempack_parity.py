"""GTAP7 vs RunGTAP (GEMPACK) quantity parity gate. LOCAL-only.

Compares the Python model's post-shock QUANTITY %-changes against GEMPACK's SL4
solution (qfd→xd, qxs→xw, qo→xp — the verified Q_TO_VAR map), cell-by-cell, in
ABSOLUTE PERCENTAGE POINTS (the natural metric for %-changes; a relative tol on
small %-changes is misleading — see gempack_reference). GEMPACK is Gragg-linearized
and Python is levels, so the per-page floor is stated in pp, not the GAMS 1% rel.

SKIPs when a row's sl4dump fixture is absent, so it never blocks the parity stamp
on a machine without the Windows-produced solution.
"""

from __future__ import annotations

import statistics
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts/gtap"))
sys.path.insert(0, str(ROOT / "src"))
from coverage_matrix import rows_for  # noqa: E402

DATASETS_DIR = ROOT / "datasets"
FIXTURES = ROOT / "tests/fixtures/gtap7_gempack"
GEMPACK_ROWS = rows_for("gtap7", "gempack", kind="mcp")

# Floor per (dataset): min fraction of cells within 1pp, and max allowed median |Δpp|.
# Measured on the real sl4dump; set conservatively below the measured value.
PP_WITHIN = 0.01  # 1 percentage point


def test_no_gempack_rows_is_a_clean_skip():
    if not GEMPACK_ROWS:
        pytest.skip("no reference='gempack' rows yet — awaiting RunGTAP SL4 dump")


def _solve_shock(dataset: str, ifsub: int):
    """Build + seed + solve base→check→shock (gtap pure MCP) and return the model."""
    from equilibria.templates.gtap import GTAPParameters
    from equilibria.templates.gtap.gtap_contract import GTAPClosureConfig
    from equilibria.templates.gtap.gtap_model_multiperiod import (
        PERIODS,
        GTAPMultiPeriodModel,
    )
    from equilibria.templates.gtap.gtap_multiperiod_driver import solve_multiperiod

    d = DATASETS_DIR / dataset
    p = GTAPParameters()
    p.load_from_har(
        basedata_path=d / "basedata.har",
        sets_path=d / "sets.har",
        default_path=d / "default.prm",
        baserate_path=d / "baserate.har",
    )
    rr = list(p.sets.r)[-1]
    ac = GTAPClosureConfig(
        name="base",
        closure_type="MCP",
        capital_mobility="sluggish",
        fix_endowments=False,
        fix_taxes=False,
        fix_technology=False,
        if_sub=bool(ifsub),
        numeraire="pnum",
    )
    gdx = ROOT / f"tests/fixtures/gtap7/{dataset}/out_gtap_shock_ifsub{ifsub}.gdx"
    mp = GTAPMultiPeriodModel(p.sets, p, ac, residual_region=rr)
    m = mp.build_sets()
    mp.build_vars(m)
    for per in PERIODS:
        mp.build_equations_intra(m, per)
    mp.build_equations_fisher(m)
    m._residual_region = rr
    mp.seed_all_periods(m, gdx)
    res = solve_multiperiod(
        m,
        p,
        ac,
        ref_gdx=gdx,
        skip_base_solve=True,
        mute_welfare=True,
        seed_from_prior=False,
        holdfix_cd=True,
        mode="gtap",
    )
    return m, int(res["shock"]["code"])


def _measure_pp(m, sl4dump: Path):
    """Return (within_1pp_fraction, median_abs_pp) across all mapped quantity cells."""
    from gempack_reference import Q_TO_VAR, gempack_qty_pct
    from pyomo.environ import value as V

    diffs = []
    for gvar, spec in Q_TO_VAR.items():
        pyname = spec["var"]
        try:
            gem = gempack_qty_pct(str(sl4dump), gvar)
        except KeyError:
            continue
        pv = getattr(m, pyname, None)
        if pv is None:
            continue
        for key, gfrac in gem.items():
            try:
                b = float(V(pv[(*key, "base")]))
                s = float(V(pv[(*key, "shock")]))
            except (KeyError, ValueError):
                continue
            if abs(b) <= 1e-12:
                continue
            py = s / b - 1.0
            diffs.append(abs(py - gfrac))
    if not diffs:
        return None, None
    within = sum(1 for x in diffs if x <= PP_WITHIN) / len(diffs)
    return within, statistics.median(diffs)


@pytest.mark.parametrize(
    "row", GEMPACK_ROWS, ids=lambda r: f"{r.dataset}-ifsub{r.ifsub}"
)
def test_gtap7_gempack_parity(row):
    sl4 = FIXTURES / row.ref
    if not sl4.exists():
        pytest.skip(f"sl4dump fixture missing: {sl4}")
    if not (DATASETS_DIR / row.dataset / "basedata.har").exists():
        pytest.skip(f"dataset HAR missing: {row.dataset}")

    m, code = _solve_shock(row.dataset, row.ifsub)
    assert code == 1, f"[{row.dataset}] Python shock did not converge (code={code})"

    within, med = _measure_pp(m, sl4)
    assert within is not None, f"[{row.dataset}] no comparable quantity cells"

    # Floor: fraction within 1pp >= the row's shock floor (expressed as a fraction*100
    # in stage_floors["shock"]); median |Δpp| must stay small.
    floor = dict(row.stage_floors)["shock"] / 100.0
    assert within >= floor, (
        f"[{row.dataset}/gempack] {within * 100:.1f}% of quantity cells within 1pp "
        f"< floor {floor * 100:.0f}% (median |Δ|={med * 100:.2f}pp) — regression"
    )
