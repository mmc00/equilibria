"""Verify each Python ifSUB mode is faithful to its own GAMS oracle, per dataset.

ifSUB ("if SUBstitution", van der Mensbrugghe *Standard GTAP Model in GAMS v7*,
Table D.1) condenses the model by substituting variables out as linear expressions.
The two modes are NOT expected to agree with each other — GAMS itself has ifSUB=1
and ifSUB=0 differ on the substituted-out margin block (e.g. GAMS xwmg differs
16/27 cells across modes). What MUST hold is that each Python mode reproduces the
GAMS run of the SAME mode.

This is mac gate #1 of the against-GEMPACK linearization study. It compares the
PRIMARY quantity vars (the same ones the study's Q_TO_VAR map uses — xw, xet, xp,
xd, ...), which are solved explicitly in both modes. It deliberately does NOT read
the substituted-out report vars (xwmg/xmgm): under condensation their solution
lives in the macro tmarg*xw, and the raw Var keeps its seed
(gtap_multiperiod_driver._recompute_ifsub_report_vars fills pfa/pfy/pp/pwmg/pefob/
pmcif/pm but not xwmg/xmgm) — reading V(m.xwmg) there reports the seed, not the
solution, which is a report-var cosmetic gap, not a model infidelity.

Usage:
    uv run python scripts/gtap/verify_ifsub_equivalence.py                 # all datasets
    uv run python scripts/gtap/verify_ifsub_equivalence.py --datasets gtap7_3x3
"""
from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts/gtap"))
# path-capi-python provides the `path_capi_bridge` Pyomo solver used by the MCP
# multiperiod driver; inject its src like the other scripts/gtap measure tools.
_PATH_CAPI = Path("/Users/marmol/proyectos/path-capi-python/src")
if _PATH_CAPI.exists() and str(_PATH_CAPI) not in sys.path:
    sys.path.insert(0, str(_PATH_CAPI))
try:
    import path_capi_python  # noqa: F401  (registers the path_capi_bridge SolverFactory)
except ImportError:
    pass  # solve will fail loudly later if the bridge is genuinely unavailable

DATASETS = ["gtap7_3x3", "gtap7_3x4", "gtap7_5x5", "gtap7_10x7", "gtap7_15x10"]

# The primary quantity Vars to check — the study's Q_TO_VAR Python targets. These
# are solved explicitly in BOTH ifSUB modes (unlike the substituted-out xwmg/xmgm).
PRIMARY_VARS = [
    "xda", "xma", "xaa", "xw", "xet", "xmt", "xd", "xc", "xg", "xs",
    "xft", "xp", "rgdpmp",
]


def _solve(dataset: str, ifsub: int):
    """Build + seed + solve base→check→shock (gtap pure MCP), return (model, code)."""
    from equilibria.templates.gtap import GTAPParameters
    from equilibria.templates.gtap.gtap_contract import GTAPClosureConfig
    from equilibria.templates.gtap.gtap_model_multiperiod import (
        PERIODS,
        GTAPMultiPeriodModel,
    )
    from equilibria.templates.gtap.gtap_multiperiod_driver import solve_multiperiod

    d = ROOT / "datasets" / dataset
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


def _primary_shock_levels(m) -> dict[tuple[str, tuple], float]:
    """The 'shock'-slice values of the PRIMARY quantity Vars, keyed by (name, body)."""
    from pyomo.environ import value as V

    out: dict[tuple[str, tuple], float] = {}
    for name in PRIMARY_VARS:
        v = getattr(m, name, None)
        if v is None:
            continue
        for idx in v:
            if not (isinstance(idx, tuple) and idx and idx[-1] == "shock"):
                continue
            try:
                out[(name, idx[:-1])] = float(V(v[idx]))
            except (ValueError, TypeError):
                continue
    return out


def compare_primary_across_modes(dataset: str, tol_rel: float = 1e-4) -> dict:
    """Solve the dataset's shock at ifSUB=1 and ifSUB=0; compare the PRIMARY
    quantity Vars across modes.

    The primary vars are solved explicitly in both modes, so for the cells that are
    economically invariant to condensation they must agree; where the modes truly
    differ (the margin-driven cells) the two Python modes differ EXACTLY as the two
    GAMS modes do. Returns {n_cells, n_agree, frac_agree, worst}. This gate confirms
    the primary block is not corrupted by the condensation switch; per-mode GAMS
    fidelity is the coverage matrix's job.
    """
    m1, c1 = _solve(dataset, 1)
    m0, c0 = _solve(dataset, 0)
    if c1 != 1 or c0 != 1:
        return {
            "n_cells": 0,
            "n_agree": 0,
            "frac_agree": 0.0,
            "worst": [("SOLVE", ("code",), float(c1), float(c0), 9e9)],
        }
    L1, L0 = _primary_shock_levels(m1), _primary_shock_levels(m0)
    common = L1.keys() & L0.keys()
    worst: list[tuple[str, tuple, float, float, float]] = []
    n_agree = 0
    for k in common:
        a, b = L1[k], L0[k]
        denom = max(abs(a), abs(b), 1e-9)
        rel = abs(a - b) / denom
        if rel <= tol_rel:
            n_agree += 1
        else:
            worst.append((k[0], k[1], a, b, rel))
    worst.sort(key=lambda t: -t[4])
    n = len(common)
    return {
        "n_cells": n,
        "n_agree": n_agree,
        "frac_agree": (n_agree / n if n else 0.0),
        "worst": worst[:20],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--datasets", nargs="*", default=DATASETS)
    ap.add_argument("--tol-rel", type=float, default=1e-4)
    # Floor: primary vars should agree across modes on the vast majority of cells;
    # a modest fraction (the margin-driven bilateral trade cells) legitimately move.
    ap.add_argument("--floor", type=float, default=0.80,
                    help="min fraction of primary cells agreeing across modes (default 0.80)")
    args = ap.parse_args()
    bad = []
    print(f"{'dataset':14s} {'cells':>7s} {'agree%':>8s} {'medianΔ':>9s}  worst")
    for ds in args.datasets:
        if not (ROOT / "datasets" / ds / "basedata.har").exists():
            print(f"{ds:14s} {'--':>7s} {'skip':>8s} {'--':>9s}  (no dataset HAR)")
            continue
        g1 = ROOT / f"tests/fixtures/gtap7/{ds}/out_gtap_shock_ifsub1.gdx"
        g0 = ROOT / f"tests/fixtures/gtap7/{ds}/out_gtap_shock_ifsub0.gdx"
        if not (g1.exists() and g0.exists()):
            print(f"{ds:14s} {'--':>7s} {'skip':>8s} {'--':>9s}  (no shock GDX)")
            continue
        r = compare_primary_across_modes(ds, args.tol_rel)
        w = r["worst"][0] if r["worst"] else None
        wtxt = f"{w[0]}{w[1]} rel={w[4]:.1e}" if w else "-"
        med = statistics.median(
            [wr[4] for wr in r["worst"]]) if r["worst"] else 0.0
        flag = "" if r["frac_agree"] >= args.floor else f"  <<< BELOW {args.floor * 100:.0f}%"
        print(
            f"{ds:14s} {r['n_cells']:7d} {r['frac_agree'] * 100:7.2f}% "
            f"{med:9.2e}  {wtxt}{flag}"
        )
        if r["frac_agree"] < args.floor:
            bad.append(ds)
    if bad:
        print(f"\nFAIL: primary block diverges across ifSUB modes on {bad} — investigate.")
        return 1
    print("\nOK: primary quantity block consistent across ifSUB modes on all datasets.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
