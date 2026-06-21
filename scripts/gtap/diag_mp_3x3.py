"""Task-6 decision gate: 3x3 multi-period model → does shock reach code=1?

Loads gtap7_3x3, builds GTAPMultiPeriodModel, seeds from GAMS reference GDX,
solves via solve_multiperiod, then compares shock-period vars against GAMS.

Gate: shock code==1 AND residual<1e-6 AND match>=96.80%
"""
from __future__ import annotations
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "gtap"))

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATASET_DIR = ROOT / "datasets" / "gtap7_3x3"
REF = Path("/Users/marmol/proyectos2/equilibria_refs/gtap7_3x3_altertax_cd/out_altertax_ifsub0.gdx")

PASS_CODE = 1
PASS_RES = 1e-6
PASS_MATCH = 96.80  # percent


def main():
    t0_wall = time.perf_counter()

    # ------------------------------------------------------------------
    # 1. Load parameters
    # ------------------------------------------------------------------
    print("=== [1/5] Loading gtap7_3x3 parameters ===")
    from equilibria.templates.gtap import GTAPParameters
    p_b_raw = GTAPParameters()
    p_b_raw.load_from_har(
        basedata_path=DATASET_DIR / "basedata.har",
        sets_path=DATASET_DIR / "sets.har",
        default_path=DATASET_DIR / "default.prm",
        baserate_path=DATASET_DIR / "baserate.har",
    )
    regions = list(p_b_raw.sets.r)
    rr = regions[-1]
    print(f"  Regions: {regions}")
    print(f"  Residual region: {rr}")

    # ------------------------------------------------------------------
    # 2. Apply altertax elasticities (CD: sigmav=sigmap=1)
    # ------------------------------------------------------------------
    print("\n=== [2/5] Applying altertax elasticities (CD) ===")
    from equilibria.templates.gtap.altertax import apply_altertax_elasticities
    p_alt = apply_altertax_elasticities(p_b_raw, in_place=False)

    # ------------------------------------------------------------------
    # 3. Build multi-period model
    # ------------------------------------------------------------------
    print("\n=== [3/5] Building GTAPMultiPeriodModel ===")
    from equilibria.templates.gtap.gtap_contract import GTAPClosureConfig
    from equilibria.templates.gtap.gtap_model_multiperiod import GTAPMultiPeriodModel, PERIODS

    alt_closure = GTAPClosureConfig(
        name="altertax", closure_type="MCP",
        capital_mobility="mobile", fix_endowments=False,
        fix_taxes=True, fix_technology=True, if_sub=False,
        numeraire="pnum",
    )

    mp = GTAPMultiPeriodModel(p_alt.sets, p_alt, alt_closure, residual_region=rr)

    print("  build_sets ...")
    m = mp.build_sets()

    print("  build_vars ...")
    t_bv = time.perf_counter()
    mp.build_vars(m)
    print(f"  build_vars done in {time.perf_counter()-t_bv:.1f}s")

    print("  build_equations_intra (3 periods) ...")
    t_ei = time.perf_counter()
    for period in PERIODS:
        print(f"    period={period} ...")
        mp.build_equations_intra(m, period)
    print(f"  build_equations_intra done in {time.perf_counter()-t_ei:.1f}s")

    print("  build_equations_fisher ...")
    t_ef = time.perf_counter()
    mp.build_equations_fisher(m)
    print(f"  build_equations_fisher done in {time.perf_counter()-t_ef:.1f}s")

    # Store residual region on model for driver
    m._residual_region = rr

    # ------------------------------------------------------------------
    # 4. Seed all periods from GAMS reference GDX
    # ------------------------------------------------------------------
    print(f"\n=== [4/5] Seeding all periods from GAMS reference ===")
    print(f"  REF: {REF}")
    if not REF.exists():
        print(f"  ERROR: reference GDX not found: {REF}")
        sys.exit(1)

    t_seed = time.perf_counter()
    mp.seed_all_periods(m, REF)
    print(f"  seed_all_periods done in {time.perf_counter()-t_seed:.1f}s")

    # ------------------------------------------------------------------
    # 5. Solve via solve_multiperiod
    # ------------------------------------------------------------------
    print("\n=== [5/5] solve_multiperiod (base → check → shock) ===")
    print("  (This may take several minutes for PATH to solve each period)")
    from equilibria.templates.gtap.gtap_multiperiod_driver import solve_multiperiod

    t_solve = time.perf_counter()
    results = solve_multiperiod(m, p_b_raw, alt_closure, ref_gdx=REF)
    t_elapsed = time.perf_counter() - t_solve
    print(f"  solve_multiperiod done in {t_elapsed:.1f}s")

    # Print per-period results
    print("\n--- Per-period results ---")
    for period in PERIODS:
        r = results.get(period, {})
        code = r.get("code", "?")
        res = r.get("residual", float("inf"))
        print(f"  {period}: code={code}, residual={res:.3e}")

    # ------------------------------------------------------------------
    # 6. Compare shock-period vars against GAMS reference
    # ------------------------------------------------------------------
    print("\n--- Shock period comparison vs GAMS reference ---")
    from _diff_core import (
        list_populated_vars, gams_levels, split_t,
    )

    # Direct comparison: iterate GAMS vars, look up the 'shock' period cell in m.
    # We don't use the proxy pattern (it breaks find_py_var's component_objects scan).
    # Instead: for each GAMS symbol, get the matching Pyomo Var from m by name,
    # then look up the ('shock') index directly.

    tol_rel = 1e-2  # 1% tolerance for match%
    tol_abs = 1e-6

    # GAMS → Python name aliases (same as diff_altertax.GAMS_TO_PY_NAME + _NAME_ALIAS)
    _GAMS_TO_PY = {
        "xa": "xaa", "xd": "xda", "xm": "xma",
        "pp": "pp_rai", "p": "p_rai",
        "ytaxInd": "ytax_ind", "ytaxind": "ytax_ind",
    }

    def _strip_pfx(s):
        if isinstance(s, str) and len(s) > 2 and s[1] == "_" and s[0] in "acfr":
            return s[2:]
        return s

    var_names = list(list_populated_vars(REF))
    print(f"  {len(var_names)} populated vars in REF GDX")

    # Skip welfare/derived vars not in multi-period model
    _SKIP = {"walras", "ev", "cv", "uh"}

    n_total = n_match = n_diverge = n_missing = 0
    worst_cells = []

    for vn in var_names:
        if vn.lower() in _SKIP:
            continue
        try:
            gams_all = gams_levels(REF, vn)
        except Exception:
            continue
        if not gams_all:
            continue

        # Find the matching Pyomo Var on m
        py_name = _GAMS_TO_PY.get(vn, _GAMS_TO_PY.get(vn.lower(), vn))
        py_var = getattr(m, py_name, None)
        if py_var is None:
            py_var = getattr(m, vn.lower(), None)
        if py_var is None:
            shock_n = sum(1 for k in gams_all if split_t(k)[1] == "shock")
            n_total += shock_n
            n_missing += shock_n
            continue

        # Compare only shock-period cells
        for full_key, g_val in gams_all.items():
            body, t = split_t(full_key)
            if t != "shock":
                continue
            n_total += 1
            # Build the multi-period index: strip GAMS prefixes from body, add 'shock'
            stripped = tuple(_strip_pfx(k) for k in body)
            if len(stripped) == 0:
                mp_idx = ("shock",)
            elif len(stripped) == 1:
                mp_idx = (stripped[0], "shock")
            else:
                mp_idx = (*stripped, "shock")

            # Try to get value from the multi-period model
            p_val = None
            candidates = [mp_idx]
            # Also try without prefix stripping
            raw_mp = (*body, "shock") if body else ("shock",)
            if raw_mp != mp_idx:
                candidates.append(raw_mp)
            # And with hhd dropped
            if "hhd" in stripped:
                no_hhd = tuple(e for e in stripped if e != "hhd")
                candidates.append((*no_hhd, "shock") if no_hhd else ("shock",))

            for cand in candidates:
                try:
                    from pyomo.environ import value as pyo_val
                    p_val = float(pyo_val(py_var[cand]))
                    break
                except Exception:
                    pass

            if p_val is None:
                n_missing += 1
                continue

            d = p_val - g_val
            rel = abs(d) / abs(g_val) if abs(g_val) > 1e-12 else (0.0 if abs(d) < tol_abs else float("inf"))
            if abs(d) <= tol_abs or rel <= tol_rel:
                n_match += 1
            else:
                n_diverge += 1
                worst_cells.append((vn, full_key, p_val, g_val, d, rel))

    match_pct = 100.0 * n_match / max(n_total, 1)

    print(f"  Shock cells: total={n_total}, match={n_match}, diverge={n_diverge}, missing={n_missing}")
    print(f"  Match%: {match_pct:.2f}% (tol_rel={tol_rel:.0%})")

    if worst_cells:
        worst_cells.sort(key=lambda x: abs(x[4]), reverse=True)
        print(f"\n  Top-5 worst diverging cells:")
        for vn, gk, pv, gv, d, rel in worst_cells[:5]:
            print(f"    {vn}[{gk}]: py={pv:.6g} gams={gv:.6g} reldiff={rel:.3%}")

    # ------------------------------------------------------------------
    # Verdict
    # ------------------------------------------------------------------
    shock_r = results.get("shock", {})
    shock_code = shock_r.get("code", -1)
    shock_res = shock_r.get("residual", float("inf"))

    total_wall = time.perf_counter() - t0_wall
    print(f"\n=== Total wall time: {total_wall:.1f}s ===")

    print("\n" + "=" * 68)
    print(f"GATE RESULT: shock code={shock_code}, residual={shock_res:.3e}, match={match_pct:.2f}%")
    print(f"PASS criteria: shock code==1 AND residual<1e-6 AND match>={PASS_MATCH}%")

    passed = (shock_code == PASS_CODE) and (shock_res < PASS_RES) and (match_pct >= PASS_MATCH)
    verdict = "PASS" if passed else "FAIL"
    print(f"VERDICT: {verdict}")
    print("=" * 68)

    if not passed:
        reasons = []
        if shock_code != PASS_CODE:
            reasons.append(f"code={shock_code} (need 1)")
        if shock_res >= PASS_RES:
            reasons.append(f"residual={shock_res:.3e} (need <{PASS_RES:.0e})")
        if match_pct < PASS_MATCH:
            reasons.append(f"match={match_pct:.2f}% (need >={PASS_MATCH}%)")
        print(f"FAIL reasons: {'; '.join(reasons)}")

    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
