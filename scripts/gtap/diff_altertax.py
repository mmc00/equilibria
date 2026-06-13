"""Cell-by-cell diff: Python altertax vs GAMS NEOS altertax out.gdx.

Mirrors GAMS comp_altertax.gms three-period structure:
  base   → Python baseline (standard GTAP, no shock)
  check  → Python altertax re-solve (CD elasticities, all factors mobile, no imptx shock)
  shock  → Python altertax shock (+10% imptx, warm-started from check)

Compares shock-period levels cell-by-cell against the NEOS altertax reference GDX.

Usage:
    uv run python scripts/gtap/diff_altertax.py
    uv run python scripts/gtap/diff_altertax.py --dataset gtap7_3x3
    uv run python scripts/gtap/diff_altertax.py --dataset gtap7_5x5
    uv run python scripts/gtap/diff_altertax.py --gdx output/9x10_altertax_neos_bundle/out.gdx
    uv run python scripts/gtap/diff_altertax.py --show-worst --tol-rel 1e-3
"""
from __future__ import annotations
import argparse, sys, time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "gtap"))

from _diff_core import (
    list_populated_vars, gams_levels, find_py_var, compare_phase,
    diff_phase_rows, diff_params_rows, ALTERTAX_PARAM_NAMES,
    write_csv, git_short_sha, build_derived,
)

GDX_9X10 = ROOT / "src/equilibria/templates/reference/gtap/data/basedata-9x10.gdx"
DEFAULT_NEOS_GDX = ROOT / "output/9x10_altertax_neos_bundle/out.gdx"

# Dataset registry: name → (data_gdx_or_har_dir, neos_bundle_dir)
DATASET_REGISTRY = {
    "9x10": (GDX_9X10, ROOT / "output/9x10_altertax_neos_bundle", "gdx"),
    "gtap7_3x3": (ROOT / "datasets/gtap7_3x3", ROOT / "output/gtap7_3x3_altertax_neos_bundle", "har"),
    "gtap7_3x4": (ROOT / "datasets/gtap7_3x4", ROOT / "output/gtap7_3x4_altertax_neos_bundle", "har"),
    "gtap7_5x5": (ROOT / "datasets/gtap7_5x5", ROOT / "output/gtap7_5x5_altertax_neos_bundle", "har"),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="9x10",
                    choices=list(DATASET_REGISTRY.keys()),
                    help="Dataset to use (default: 9x10)")
    ap.add_argument("--gdx", type=Path, default=None,
                    help="GAMS altertax reference GDX (overrides --dataset default)")
    ap.add_argument("--tol-rel", type=float, default=1e-3)
    ap.add_argument("--tol-abs", type=float, default=1e-6)
    ap.add_argument("--show-worst", action="store_true",
                    help="Print the worst diverging cell for each variable")
    ap.add_argument("--csv", type=Path, default=None)
    ap.add_argument("--no-gams-warm", action="store_true",
                    help="Skip GAMS warm-start for check period (use getData init only)")
    ap.add_argument("--use-gams-check", action="store_true",
                    help="Skip Python check solve; seed shock from GAMS check period values directly")
    args = ap.parse_args()

    data_path, bundle_dir, loader = DATASET_REGISTRY[args.dataset]
    _default_gdx = bundle_dir / "out_local.gdx"
    if not _default_gdx.exists():
        _default_gdx = bundle_dir / "out.gdx"
    gdx_path = args.gdx or _default_gdx

    from equilibria.templates.gtap import (
        GTAPParameters, GTAPModelEquations,
    )
    from equilibria.templates.gtap.altertax import (
        apply_altertax_elasticities,
    )
    from equilibria.templates.gtap.gtap_contract import GTAPClosureConfig
    from equilibria.templates.gtap.gtap_parity_pipeline import GTAPVariableSnapshot

    import importlib.util as _u
    spec = _u.spec_from_file_location(
        "run_gtap", str(ROOT / "scripts" / "gtap" / "run_gtap.py")
    )
    run_gtap = _u.module_from_spec(spec)
    sys.modules["run_gtap"] = run_gtap
    spec.loader.exec_module(run_gtap)

    # Load parameters for selected dataset
    print(f"=== Loading dataset: {args.dataset} ===")
    p_b_raw = GTAPParameters()
    if loader == "gdx":
        p_b_raw.load_from_gdx(data_path)
    else:
        p_b_raw.load_from_har(
            basedata_path=data_path / "basedata.har",
            sets_path=data_path / "sets.har",
            default_path=data_path / "default.prm",
            baserate_path=data_path / "baserate.har",
        )

    # Residual region: last region (HAR convention) or NAmerica (9x10)
    res_region = "NAmerica" if args.dataset == "9x10" else list(p_b_raw.sets.r)[-1]

    # Build closures
    if args.dataset == "9x10":
        contract = run_gtap._build_gtap_contract_with_calibration("gtap_standard7_9x10")
        base_closure = contract.closure.model_copy(update={"if_sub": False})
    else:
        base_closure = GTAPClosureConfig(
            name="base", closure_type="MCP",
            capital_mobility="sluggish", fix_endowments=False,
            fix_taxes=False, fix_technology=False, if_sub=False,
            numeraire="pnum",
        )

    # Altertax closure: all factors mobile, taxes fixed, standard numeraire
    alt_closure = GTAPClosureConfig(
        name="altertax",
        closure_type="MCP",
        capital_mobility="mobile",
        fix_endowments=False,
        fix_taxes=True,
        fix_technology=True,
        if_sub=False,
        numeraire=base_closure.numeraire,
        rmuv=getattr(base_closure, "rmuv", None),
        imuv=getattr(base_closure, "imuv", None),
    )

    # =========================================================================
    # [1/3] Python altertax betaCal period
    # =========================================================================
    # GAMS altertax does NOT run a full solve for the base period.
    # Instead, GAMS runs betaCal: a 4-equation MCP that only solves for
    # phi/betaP/betaG/betaS with holdfixed=1 (everything else at getData init).
    # After betaCal, gf/aft are recalibrated from getData initialization values.
    # Python must mirror this: set getData init values and use existing betaP calibration.
    #
    # getData initialization values (from comp_altertax.gms lines 2840-2860):
    #   pft = 1, pfy = 1, pf = 1/(1-kappa), xf = evfb*(1-kappa), xft = sum(xf)
    #   gf  = xf / (xscale * xft)  (since pft=pfy=1: (pft/pfy)^omega = 1)
    print(f"=== [1/3] Python {args.dataset} altertax betaCal (getData init + betaP calibration) ===")

    # Apply altertax elasticities before betaCal — mirrors GAMS order:
    # parameter_altertax.gms sets sigmav=1 etc., then betaCal is solved.
    p_alt = apply_altertax_elasticities(p_b_raw, in_place=False)

    # Build model with base_closure to get betaP/phi0 calibration.
    # Python's initial betaP/phi0 already match GAMS betaCal output (verified:
    # betap[EU_28]=0.7545, phi0[EU_28]=0.8380 match GAMS base values to 0.01%).
    eq_b = GTAPModelEquations(p_alt.sets, p_alt, base_closure, residual_region=res_region)
    m_b = eq_b.build_model()

    from pyomo.environ import value as pyo_value
    # Python's build_model already sets correct getData init values:
    #   xf = evfb*(1-kappa)*xscale, xft = sum_a(xf/xscale) = sum_a(xf_raw)
    #   gf_share already calibrated to match GAMS gf = xf_raw/xft_raw
    # No manual override needed — GAMS warm-start seeds the non-trivial equilibrium basin.
    print(f"  base: using Python build_model init (gf_share matches GAMS, xft = sum_a(xf/xscale))")
    for _r in p_alt.sets.r:
        _yc = float(pyo_value(m_b.yc[_r])) if hasattr(m_b, "yc") else 0.
        _betaP = float(pyo_value(m_b.betap[_r])) if hasattr(m_b, "betap") else 0.
        _phi = float(pyo_value(m_b.phi[_r])) if hasattr(m_b, "phi") else 0.
        _regy = float(pyo_value(m_b.regy[_r])) if hasattr(m_b, "regy") else 0.
        print(f"  [base diag] yc[{_r}]={_yc:.6f}  betaP={_betaP:.6f}  phi={_phi:.6f}  regy={_regy:.6f}")

    # =========================================================================
    # [2/3] Python altertax check period
    #   - NO imptx shock (same as GAMS 'check' period)
    #   - t0_snapshot=m_b: pf0/xf0 are getData init values (pf=1/(1-kappa), xf=evfb*(1-kappa))
    #   - Warm-start from betaCal output (m_b has getData init values + solved phi/betaP)
    # =========================================================================
    print("\n=== [2/3] Python altertax check period (no shock, warm-start from betaCal) ===")

    # m_b now has getData initialization values (pf=1/(1-kappa), pft=1, xf=evfb*(1-kappa))
    # plus betaCal output (phi, betaP, betaG, betaS solved). Use as t0_snapshot so that
    # pf0=1/(1-kappa) matches GAMS betaCal exactly.
    #
    # Critical: Update phip in p_alt to betaCal phi values (phi from base period).
    # GAMS sets phiP[check] = phi[base] (the CPI deflator from betaCal). Python initializes
    # phip=1 (calibration assumption), but the check period must use betaCal phi as phip
    # phiP[check] = pcons[base] = 1.0 (GAMS convention: phiP is the CPI deflator
    # normalized to the base period, so phiP[base]=1 by construction).
    # Python mistakenly used phi[base] (the CPI level ≈0.838), which is wrong.
    for _r in p_alt.sets.r:
        try:
            if hasattr(p_alt, "calibration") and hasattr(p_alt.calibration, "phip"):
                p_alt.calibration.phip[(_r,)] = 1.0
        except Exception:
            pass
    eq_chk = GTAPModelEquations(
        p_alt.sets, p_alt, alt_closure,
        residual_region=res_region,
        t0_snapshot=m_b,
    )
    m_chk = eq_chk.build_model()

    # Unfix regy: GAMS regYeq.regY is endogenous in compStat (cal.gms:645-650).
    # Python's altertax closure may pin regy=base value via is_counterfactual gate
    # (same bug as the prior regY-fixed-in-shock issue). Unfix so PATH + warm-start
    # can find the GAMS check equilibrium (regy[EU_28]≈13.70 vs base 15.70).
    _regy_unfixed = 0
    for _r in p_alt.sets.r:
        try:
            if hasattr(m_chk, "regy") and m_chk.regy[_r].fixed:
                m_chk.regy[_r].unfix()
                _regy_unfixed += 1
        except Exception:
            pass
    if _regy_unfixed:
        print(f"  [regy] Unfixed regy for {_regy_unfixed} regions (mirrors GAMS regYeq.regY endogenous)")

    # xft is already correctly initialized by build_model: sum_a(xf/xscale) = sum_a(xf_raw)
    # which equals GAMS xft_init = sum_a(pfy*xf_raw). No override needed.

    # Override phip=1.0 on the built model: GAMS phiP[check]=pcons[base]=1.0
    _phip_updated = 0
    for _r in p_alt.sets.r:
        try:
            if hasattr(m_chk, "phip"):
                m_chk.phip[_r].set_value(1.0)
                _phip_updated += 1
        except Exception:
            pass
    if _phip_updated:
        print(f"  [phip] Set phip=1.0 for {_phip_updated} regions (GAMS phiP[check]=pcons[base]=1.0)")

    # Warm-start check period from GAMS check period values if available.
    # This places PATH near the non-trivial equilibrium basin.
    _n_warm_set = 0
    if args.no_gams_warm:
        print("  [GAMS-warm] SKIPPED (--no-gams-warm)")
    else:
        try:
            try:
                from _diff_core import gams_levels  # type: ignore
            except ImportError:
                import sys as _sys
                _sys.path.insert(0, str(Path(__file__).resolve().parent))
                from _diff_core import gams_levels  # type: ignore
            try:
                from _diff_core import list_populated_vars  # type: ignore
            except ImportError:
                pass
            # Warm-start from ALL GAMS check period variables found in both GDX and Python model
            # Try both GAMS camelCase name and lowercase (GAMS "regY" → Python "regy")
            # Warm-start pft + pf: pft seeds the equilibrium basin, and pf is needed
            # because eq_pfteq: pft^2 = sum(gf*pfy^2) = sum(gf*(pf*(1-kap))^2).
            # With pfy=1 (base), the only solution is pft=1 (trivial).
            # With pf seeded at GAMS check values, pfy=pf*(1-kap) ≈ 0.662 for
            # EU_28/Land/Food, making pft=0.662 the consistent equilibrium.
            # pft + pf + pfy: all three needed to break the trivial equilibrium.
            # pft=1 is always consistent with pfy=1 (build_model init).
            # Seeding pfy=GAMS-check makes eq_pfteq inconsistent at pft=1,
            # pushing PATH toward the GAMS equilibrium pft≈0.662.
            _WARM_PRICES = {"pft", "pf", "pfy", "pfact", "regy", "yc", "yg", "rsav",
                            "facty", "ytaxind", "ytax_ind", "phi", "pcons", "rore", "rorc", "arent",
                            "xft", "xf", "pabs", "pi", "psave", "savf", "chif",
                            "ytaxtot", "kstock", "kapend"}
            # GDX camelCase → Python underscore/lowercase mapping
            _GAMS_TO_PY_NAME = {
                "ytaxInd": "ytax_ind",   # Var in Python, Expression alias separate
                "ytaxTot": "ytaxTot",    # same in Python
                "factY": "facty",        # lowercase in Python
                "phiP": "phip",          # lowercase in Python
                "regY": "regy",          # lowercase in Python
                "kapEnd": "kapEnd",      # same in Python
                "chiSave": "chiSave",    # same in Python
            }
            _WARM_SKIP_ALL_BUT_PRICES = True
            _gams_all_vars = list_populated_vars(gdx_path)
            for _vn in _gams_all_vars:
                if _WARM_SKIP_ALL_BUT_PRICES and _vn.lower() not in _WARM_PRICES:
                    continue
                try:
                    _gvals = gams_levels(gdx_path, _vn)
                except Exception:
                    continue
                # Resolve Python attribute name (handle GAMS camelCase → Python names)
                _py_name = _GAMS_TO_PY_NAME.get(_vn, _vn)
                _pyvar = getattr(m_chk, _py_name, None)
                if _pyvar is None:
                    _pyvar = getattr(m_chk, _vn, None)
                if _pyvar is None:
                    _pyvar = getattr(m_chk, _vn.lower(), None)
                if _pyvar is None:
                    continue
                for _gkey, _gval in _gvals.items():
                    if isinstance(_gkey, tuple) and _gkey[-1] == "check":
                        _pykey = _gkey[:-1]  # strip period dimension
                    else:
                        continue
                    # Strip GAMS set-type prefixes (a_, c_, f_, r_) so that
                    # e.g. ('EU_28','Land','a_Food') → ('EU_28','Land','Food')
                    _pykey = tuple(
                        _k[2:] if isinstance(_k, str) and len(_k) > 2
                               and _k[1] == '_' and _k[0] in 'acfr'
                        else _k
                        for _k in _pykey
                    )
                    try:
                        _v = _pyvar[_pykey] if len(_pykey) > 1 else _pyvar[_pykey[0]]
                        if not _v.fixed:
                            _v.set_value(float(_gval))
                            _n_warm_set += 1
                    except Exception:
                        pass
            if _n_warm_set:
                print(f"  [GAMS-warm] Warm-started check period from GAMS check values: {_n_warm_set} vars set")
        except Exception as _we:
            print(f"  [GAMS-warm] skipped: {_we}")

    warm_b = GTAPVariableSnapshot.from_python_model(m_chk)
    # Verify warm-start captured GAMS values
    try:
        _eu_land_warm = warm_b.get("pft") if hasattr(warm_b, "get") else None
        if _eu_land_warm is not None:
            _v = _eu_land_warm.get(("EU_28", "Land"))
            print(f"  [warm diag] warm_b.pft[EU_28,Land]={_v:.6f}")
        else:
            _v = m_chk.pft["EU_28", "Land"].value
            print(f"  [warm diag] m_chk.pft[EU_28,Land]={_v:.6f}")
    except Exception:
        pass

    # Pre-PATH residual check at GAMS warm-start point
    try:
        from pyomo.environ import value as _pv2
        _big_resids = []
        for _cname in ["eq_pfteq", "eq_pfeq", "eq_xfteq", "eq_yc", "eq_regy", "eq_phip", "eq_phieq", "eq_facty"]:
            _con = getattr(m_chk, _cname, None)
            if _con is None:
                continue
            for _idx in _con:
                try:
                    _c = _con[_idx]
                    if not _c.active:
                        continue
                    _body = _pv2(_c.body)
                    _lb = float(_c.lower) if _c.lower is not None else None
                    _ub = float(_c.upper) if _c.upper is not None else None
                    _rhs = _lb if _lb is not None else _ub
                    if _rhs is None:
                        continue
                    _resid = abs(_body - _rhs)
                    if _resid > 1e-3:
                        _big_resids.append((_cname, _idx, _resid, _body, _rhs))
                except Exception:
                    pass
        _big_resids.sort(key=lambda x: -x[2])
        print(f"  [pre-PATH] Equations with large residual at GAMS warm-start ({len(_big_resids)} total):")
        for _cn, _ci, _rv, _body, _rhs in _big_resids[:8]:
            print(f"    {_cn}{_ci}: body={_body:.4f}  rhs={_rhs:.4f}  residual={_rv:.4e}")
        # Extra diagnostics for eq_yc and eq_regy
        try:
            _betap_eu = float(_pv2(m_chk.betap["EU_28"]))
            _phi_eu = float(_pv2(m_chk.phi["EU_28"]))
            _phip_eu = float(_pv2(m_chk.phip["EU_28"]))
            _regy_eu = float(_pv2(m_chk.regy["EU_28"]))
            _yc_eu_v = float(_pv2(m_chk.yc["EU_28"]))
            print(f"  [eq_yc EU_28] yc={_yc_eu_v:.4f}  betap={_betap_eu:.4f}  phi={_phi_eu:.4f}  phip={_phip_eu:.4f}  regy={_regy_eu:.4f}  rhs={_betap_eu*(_phi_eu/max(_phip_eu,1e-8))*_regy_eu:.4f}")
        except Exception:
            pass
        try:
            _facty_eu = float(_pv2(m_chk.facty["EU_28"])) if hasattr(m_chk, "facty") else None
            _ytaxind_eu = float(_pv2(m_chk.ytaxInd["EU_28"])) if hasattr(m_chk, "ytaxInd") else None
            _regy_eu2 = float(_pv2(m_chk.regy["EU_28"])) if hasattr(m_chk, "regy") else None
            _eq_regy_active = m_chk.eq_regy["EU_28"].active if hasattr(m_chk, "eq_regy") else None
            _eq_facty_active = m_chk.eq_facty["EU_28"].active if hasattr(m_chk, "eq_facty") else None
            _regy_fixed = m_chk.regy["EU_28"].fixed if hasattr(m_chk, "regy") else None
            print(f"  [eq_regy EU_28] facty={_facty_eu:.4f}  ytaxind={_ytaxind_eu:.4f}  sum={(_facty_eu or 0)+(_ytaxind_eu or 0):.4f}  regy={_regy_eu2:.4f}  eq_regy.active={_eq_regy_active}  eq_facty.active={_eq_facty_active}  regy.fixed={_regy_fixed}")
        except Exception as _re:
            print(f"  [eq_regy diag] error: {_re}")
    except Exception as _pre_exc:
        print(f"  [pre-PATH] residual check error: {_pre_exc}")

    if args.use_gams_check:
        # Skip Python check solve — use GAMS check period values directly.
        # This tests whether the shock period can be solved from the GAMS check equilibrium.
        print(f"  [--use-gams-check] Skipping Python check solve; using GAMS check values as seed for shock period")
        r_chk = {"residual": 0.0, "termination_code": 1}
        res_chk = 0.0
        sec_chk = 0.0
        # m_chk already has GAMS check values from the warm-start above
    else:
        t0 = time.perf_counter()
        r_chk = run_gtap._run_path_capi_nonlinear_full(
            m_chk, p_alt,
            enforce_post_checks=False, strict_path_capi=False,
            closure_config=alt_closure, equation_scaling=True,
            solution_hint=warm_b,
        )
        sec_chk = time.perf_counter() - t0
        res_chk = float(r_chk.get("residual") or 0.0)
    print(f"  check residual={res_chk:.3e}  code={r_chk.get('termination_code')}  t={sec_chk:.2f}s")
    # Diagnostics: print pft/pf/yc for first few vars
    try:
        from pyomo.environ import value as _pv
        r0 = list(p_alt.sets.r)[0]
        f0 = list(p_alt.sets.mf)[0] if p_alt.sets.mf else list(p_alt.sets.f)[0]
        a0 = list(p_alt.sets.a)[0]
        if hasattr(m_chk, "pft"):
            pft_var = m_chk.pft[r0,f0]
            print(f"  [chk diag] pft[{r0},{f0}]={_pv(pft_var):.6f}  fixed={pft_var.fixed}")
        if hasattr(m_chk, "pf"):  print(f"  [chk diag] pf[{r0},{f0},{a0}]={_pv(m_chk.pf[r0,f0,a0]):.6f}")
        if hasattr(m_chk, "yc"):  print(f"  [chk diag] yc[{r0}]={_pv(m_chk.yc[r0]):.6f}")
        if hasattr(m_chk, "xft"): print(f"  [chk diag] xft[{r0},{f0}]={_pv(m_chk.xft[r0,f0]):.6f}")
        print(f"  [chk diag] gf_share[{r0},{f0},{a0}]={float(m_chk.gf_share[r0,f0,a0]):.6f}")
        if hasattr(m_chk, "eq_pfteq"):
            try:
                eq = m_chk.eq_pfteq[r0, f0]
                print(f"  [chk diag] eq_pfteq[{r0},{f0}] active={eq.active}")
            except Exception as _e2: print(f"  [chk diag] eq_pfteq error: {_e2}")
        # Print all pft values and pf sample
        try:
            print(f"  [chk diag] mf set size: {len(list(p_alt.sets.mf))}")
            _pft_count = sum(1 for _ in m_chk.pft)
            print(f"  [chk diag] pft var size: {_pft_count}")
            for _idx in m_chk.pft:
                _v = m_chk.pft[_idx]
                print(f"    pft{_idx}={float(_pv(_v)):.6f}  fixed={_v.fixed}")
        except Exception as _ex2:
            print(f"  [chk diag] pft iter err: {_ex2}")
        try:
            _pf_count = 0
            for _idx in m_chk.pf:
                if _pf_count > 6: break
                _v = m_chk.pf[_idx]
                print(f"    pf{_idx}={float(_pv(_v)):.6f}")
                _pf_count += 1
        except Exception as _ex3:
            print(f"  [chk diag] pf iter err: {_ex3}")
    except Exception as _de:
        print(f"  [chk diag] error: {_de}")

    # =========================================================================
    # [3/3] Python altertax shock period
    #   - Apply +10% imptx shock on top of altertax params
    #   - Warm-start from check period solution
    # =========================================================================
    print("\n=== [3/3] Python altertax shock (+10% imptx, warm-started from check) ===")
    import copy
    p_alt_shock = copy.deepcopy(p_alt)
    # GAMS comp.gms:3909: imptx.fx = (1 + imptx.l)*1.10 - 1  (tm_pct: scale the
    # tariff POWER, not the rate). The old `imptx*1.10` scaled the rate, giving a
    # ~10x smaller shock on low-tariff goods and diverging the whole shock period.
    for key in list(p_alt_shock.taxes.imptx.keys()):
        old = float(p_alt_shock.taxes.imptx[key] or 0.0)
        p_alt_shock.taxes.imptx[key] = (1.0 + old) * 1.10 - 1.0

    warm_chk = GTAPVariableSnapshot.from_python_model(m_chk)
    # NOTE: do NOT pass t0_snapshot=m_chk here. With t0=m_chk, pf0/xf0 (the Fisher-
    # index benchmark) are pinned to the check solution, which is inconsistent with
    # the shocked imptx and steers PATH into a degenerate basin (prices→0.04,
    # code=2). Building WITHOUT t0 (so pf0/xf0 come from the calibrated benchmark)
    # and warm-starting from the check snapshot converges (verified: same model
    # converges to ~3e-9 from a full seed).
    eq_alt = GTAPModelEquations(
        p_alt_shock.sets, p_alt_shock, alt_closure,
        residual_region=res_region,
    )
    m_alt = eq_alt.build_model()
    t0 = time.perf_counter()
    r_alt = run_gtap._run_path_capi_nonlinear_full(
        m_alt, p_alt_shock,
        enforce_post_checks=False, strict_path_capi=False,
        closure_config=alt_closure, equation_scaling=True,
        solution_hint=warm_chk,
    )
    sec_alt = time.perf_counter() - t0
    res_alt = float(r_alt.get("residual") or 0.0)
    print(f"  shock residual={res_alt:.3e}  code={r_alt.get('termination_code')}  t={sec_alt:.2f}s")

    var_names = list_populated_vars(gdx_path)
    print(f"\nPopulated GAMS Vars in {gdx_path.name}: {len(var_names)}")

    # =========================================================================
    # [base diff] Compare Python base altertax solve vs GAMS period='base'
    # Confirms whether the starting point diverges before check/shock.
    # =========================================================================
    print(f"\n{'='*60}")
    print("BASE PERIOD: Python m_b vs GAMS period='base' (pf, pfact, yc, regY)")
    print(f"{'='*60}")
    for vname in ["pf", "pfact", "yc", "regY", "betaP", "phi"]:
        gams_all = gams_levels(gdx_path, vname)
        if not gams_all:
            continue
        py_var, py_name = find_py_var(m_b, vname, derived=build_derived(m_b))
        if py_var is None:
            print(f"  {vname}: no py var")
            continue
        s = compare_phase(py_var, gams_all, "base", tol_rel=args.tol_rel, tol_abs=args.tol_abs)
        status = "ok" if s["n_diverge"] == 0 and s["n_missing"] == 0 else "DIFF"
        print(f"  {vname:<8s} cells={s['n_total']:>4d} match={s['n_match']:>4d} "
              f"diverge={s['n_diverge']:>4d} miss={s['n_missing']:>4d} "
              f"max_rel={s['max_rel']:.3e}  {status}")
        if s["worst"] and status == "DIFF":
            w = s["worst"]
            print(f"    worst: {w[0]}  py={w[1]:+.6e}  gams={w[2]:+.6e}  rel={w[4]*100:.2f}%")

    # =========================================================================
    # [check diff] Compare Python check solve vs GAMS period='check'
    # =========================================================================
    print(f"\n{'='*60}")
    print("CHECK PERIOD: Python m_chk vs GAMS period='check' (pf, pfact, yc, regY, pcons)")
    print(f"{'='*60}")
    for vname in ["pf", "pfact", "yc", "regY", "pcons", "rore", "rorc", "arent"]:
        gams_all = gams_levels(gdx_path, vname)
        if not gams_all:
            continue
        py_var, py_name = find_py_var(m_chk, vname, derived=build_derived(m_chk))
        if py_var is None:
            print(f"  {vname}: no py var")
            continue
        s = compare_phase(py_var, gams_all, "check", tol_rel=args.tol_rel, tol_abs=args.tol_abs)
        status = "ok" if s["n_diverge"] == 0 and s["n_missing"] == 0 else "DIFF"
        print(f"  {vname:<8s} cells={s['n_total']:>4d} match={s['n_match']:>4d} "
              f"diverge={s['n_diverge']:>4d} miss={s['n_missing']:>4d} "
              f"max_rel={s['max_rel']:.3e}  {status}")
        if s["worst"] and status == "DIFF":
            w = s["worst"]
            print(f"    worst: {w[0]}  py={w[1]:+.6e}  gams={w[2]:+.6e}  rel={w[4]*100:.2f}%")

    # NEOS out.gdx uses period "shock" for the +10% tariff altertax result
    phase = "shock"
    print(f"\n{'='*120}")
    print(f"PHASE: altertax → comparing Python m_alt vs GAMS {gdx_path.name} period='{phase}'")
    print(f"  tol_rel={args.tol_rel}  tol_abs={args.tol_abs}")
    print(f"{'='*120}")
    print(f"{'gams_var':<14s} {'py_var':<14s} {'cells':>7s} {'match':>7s} {'diverge':>8s} {'missing':>8s} {'max_abs':>10s} {'max_rel':>10s}  status")
    print("-" * 120)

    git_sha = git_short_sha(ROOT)
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    rows, agg = diff_phase_rows(
        dataset=f"{args.dataset}_altertax", phase=phase, var_names=var_names,
        gdx_path=gdx_path, model_py=m_alt,
        tol_rel=args.tol_rel, tol_abs=args.tol_abs,
        residual=res_alt, git_sha=git_sha, generated_at=generated_at,
        derived=build_derived(m_alt),
        solve_seconds=sec_alt,
    )

    diverge_details = []
    for r in rows:
        if r["var"] == "__SUMMARY__":
            continue
        cells = int(r["cells"])
        match = int(r["match"])
        diverge = int(r["diverge"])
        missing = int(r["missing"])
        mx_abs = r["max_abs_err"] or "—"
        mx_rel = r["max_rel_err"] or "—"
        if not r["py_var"]:
            status = "no-py"
            py = "<n/a>"
        elif diverge == 0 and missing == 0:
            status = "ok"
            py = r["py_var"]
        else:
            status = "diff" if diverge else "miss"
            py = r["py_var"]
        print(f"{r['var']:<14s} {py:<14s} {cells:>7d} {match:>7d} "
              f"{diverge:>8d} {missing:>8d} {mx_abs:>10s} {mx_rel:>10s}  {status}")

        if args.show_worst and (diverge > 0 or missing > 0) and r["py_var"]:
            gams_all = gams_levels(gdx_path, r["var"])
            py_var, _ = find_py_var(m_alt, r["var"], derived=build_derived(m_alt))
            if py_var is not None:
                s = compare_phase(py_var, gams_all, phase,
                                  tol_rel=args.tol_rel, tol_abs=args.tol_abs)
                if s["worst"]:
                    diverge_details.append((r["var"], r["py_var"], s))

    print("-" * 120)
    print(f"  Vars total:           {agg['vars_total']}")
    print(f"  Vars all-match:       {agg['vars_match_all']}")
    print(f"  Vars partial/diverge: {agg['vars_partial']}")
    print(f"  Vars not in Python:   {agg['vars_no_py']}")
    print(f"  Cells total:          {agg['cells_total']}")
    print(f"  Cells match:          {agg['cells_match']}")
    print(f"  Cells diverge:        {agg['cells_diverge']}")
    print(f"  Cells missing/no-py:  {agg['cells_missing']}")
    coverage = (agg["cells_match"] / agg["cells_total"] * 100.0) if agg["cells_total"] else 0.0
    print(f"  Match rate:           {coverage:.2f}%")

    # =========================================================================
    # Parameter diff: GAMS params (stored as vars in GDX) vs Python Params
    # Catches recalibrated values like alphaa, auh that are invisible to the
    # standard variable diff.
    # =========================================================================
    print(f"\n{'='*120}")
    print(f"PARAMETERS: Python Params vs GAMS period='{phase}'")
    print(f"{'='*120}")
    print(f"{'gams_par':<16s} {'py_param':<22s} {'cells':>7s} {'match':>7s} {'diverge':>8s} {'missing':>8s} {'max_abs':>10s} {'max_rel':>10s}  status")
    print("-" * 120)

    param_rows, param_agg = diff_params_rows(
        dataset=f"{args.dataset}_altertax", phase=phase,
        param_names=ALTERTAX_PARAM_NAMES,
        gdx_path=gdx_path, model_py=m_alt,
        tol_rel=args.tol_rel, tol_abs=args.tol_abs,
        residual=res_alt, git_sha=git_sha, generated_at=generated_at,
        solve_seconds=sec_alt,
    )
    param_diverge_details = []
    for r in param_rows:
        cells  = int(r["cells"])
        match  = int(r["match"])
        diverge = int(r["diverge"])
        missing = int(r["missing"])
        mx_abs = r["max_abs_err"] or "—"
        mx_rel = r["max_rel_err"] or "—"
        py_lbl = r["py_var"] or "<n/a>"
        if not r["py_var"]:
            status = "no-py"
        elif diverge == 0 and missing == 0:
            status = "ok"
        else:
            status = "diff" if diverge else "miss"
        print(f"{r['var']:<16s} {py_lbl:<22s} {cells:>7d} {match:>7d} "
              f"{diverge:>8d} {missing:>8d} {mx_abs:>10s} {mx_rel:>10s}  {status}")
        if args.show_worst and (diverge > 0 or missing > 0) and r["py_var"]:
            gams_all = gams_levels(gdx_path, r["var"].lstrip("[par]"))
            from _diff_core import compare_phase_param, _find_py_param
            py_p, _ = _find_py_param(m_alt, r["var"].lstrip("[par]"))
            if py_p is not None:
                s = compare_phase_param(py_p, gams_all, phase,
                                        tol_rel=args.tol_rel, tol_abs=args.tol_abs)
                if s["worst"]:
                    param_diverge_details.append((r["var"], r["py_var"], s))
    print("-" * 120)

    if args.show_worst and diverge_details:
        print(f"\n  Worst diverging cell per variable (altertax vars):")
        for name, py_name, stats in diverge_details[:30]:
            w = stats["worst"]
            if w is None:
                continue
            key, p_val, g_val, d, rel = w
            rel_str = f"{rel*100:.3f}%" if rel != float("inf") else "inf"
            print(f"    {name:<12s} {str(key):<60s}  py={p_val:+.6e}  gams={g_val:+.6e}  Δ={d:+.3e}  rel={rel_str}")

    if args.show_worst and param_diverge_details:
        print(f"\n  Worst diverging cell per parameter (altertax params):")
        for name, py_name, stats in param_diverge_details[:30]:
            w = stats["worst"]
            if w is None:
                continue
            key, p_val, g_val, d, rel = w
            rel_str = f"{rel*100:.3f}%" if rel != float("inf") else "inf"
            print(f"    {name:<14s} {str(key):<60s}  py={p_val:+.6e}  gams={g_val:+.6e}  Δ={d:+.3e}  rel={rel_str}")

    if args.csv:
        write_csv(args.csv, rows)
        print(f"\nWrote {len(rows)} rows to {args.csv}")


if __name__ == "__main__":
    main()
