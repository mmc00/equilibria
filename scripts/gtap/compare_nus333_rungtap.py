"""NUS333 head-to-head: equilibria (Python) vs RunGTAP/GEMPACK (this machine).

Runs the same 10% uniform shock to the import tariff power (tm) on the NUS333
dataset using the local equilibria template, computes Huff/McDougall welfare
decomposition, and dumps both percent-change macro variables and welfare
components to JSON so they can be compared against the RunGTAP output produced
by `runs/nus333_compare/rungtap/tm10.cmf` (gtapv7.exe).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
SRC = ROOT / "src"
SCRIPTS_GTAP = ROOT / "scripts" / "gtap"
# Sibling repo per docs/site/guide/installation.md.
PATH_CAPI_SRC = ROOT.parent / "path-capi-python" / "src"
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(SCRIPTS_GTAP))
if PATH_CAPI_SRC.exists():
    sys.path.insert(0, str(PATH_CAPI_SRC))

# Point the loader at GAMS 53's PATH 5.2 DLL on this Windows machine, and add
# the GAMS install dir to PATH so dependent DLLs (msc*, etc.) resolve.
import os
GAMS_DIR = Path(r"C:\GAMS\53")
PATH_DLL = GAMS_DIR / "path52.dll"
if PATH_DLL.exists():
    os.environ.setdefault("PATH_CAPI_LIBPATH", str(PATH_DLL))
    os.environ["PATH"] = str(GAMS_DIR) + os.pathsep + os.environ.get("PATH", "")

# PATH solver license: must be set in env before the DLL initializes its license
# context. The wrapper does not call License_SetString; PATH_LICENSE_STRING is
# the standard mechanism for the PATH C API to pick up the license.
if not os.environ.get("PATH_LICENSE_STRING"):
    raise SystemExit(
        "PATH_LICENSE_STRING is not set. Export it before running this script."
    )

NUS333 = ROOT / "src" / "equilibria" / "templates" / "reference" / "gtap" / "data" / "nus333"
OUT_DIR = ROOT / "runs" / "nus333_compare" / "equilibria"
OUT_DIR.mkdir(parents=True, exist_ok=True)

from equilibria.templates.gtap import GTAPParameters
from equilibria.templates.gtap.gtap_model_equations import GTAPModelEquations
from equilibria.templates.gtap.gtap_contract import GTAPClosureConfig
from equilibria.templates.gtap.welfare_decomp import (
    compute_welfare_decomposition,
    compute_welfare_decomposition_homotopy,
)

# Use the proven helpers from compare_nus333_vs_neos so we hit the same solver path.
import compare_nus333_vs_neos as _cmp
# Point PATH_LIB at the GAMS-bundled Windows PATH 5.2 DLL on this machine.
if PATH_DLL.exists():
    _cmp.PATH_LIB = PATH_DLL
from compare_nus333_vs_neos import _solve, _apply_tariff_shock, _copy_var_levels, _extract_key

# Use run_gtap's full snapshot collector (the welfare module needs this format).
from run_gtap import _collect_key_quantities


def _augment_snapshot(model, snap: dict) -> None:
    """Add `uh` (and any other Var-of-(r,)) variables needed by the welfare
    decomposition, since the stock `_collect_key_quantities` omits them.
    """
    from pyomo.environ import value, Var
    for vname in ("uh", "ug", "us", "u", "cv", "ev", "pop"):
        comp = getattr(model, vname, None)
        if comp is None:
            continue
        data: dict[str, float] = {}
        try:
            for idx in comp:
                key = "|".join(str(v) for v in idx) if isinstance(idx, tuple) else str(idx)
                try:
                    data[key] = float(value(comp[idx]))
                except (ValueError, TypeError):
                    continue
        except TypeError:
            try:
                data["__scalar__"] = float(value(comp))
            except (ValueError, TypeError):
                pass
        if data:
            snap[vname] = data


def main() -> int:
    print(f"NUS333 path: {NUS333}")
    params = GTAPParameters()
    params.load_from_har(
        basedata_path=NUS333 / "basedata.har",
        sets_path=NUS333 / "sets.har",
        default_path=NUS333 / "default.prm",
        baserate_path=NUS333 / "baserate.har",
    )
    base_params = GTAPParameters()
    base_params.load_from_har(
        basedata_path=NUS333 / "basedata.har",
        sets_path=NUS333 / "sets.har",
        default_path=NUS333 / "default.prm",
        baserate_path=NUS333 / "baserate.har",
    )
    print(f"Sets: r={params.sets.r}, i={params.sets.i}, f={params.sets.f}")

    # Closure: capFix — savf pinned to baseline (savf = pigbl * savf_bar) for
    # non-residual regions; residual region (ROW) absorbs the capital-account
    # identity. RunGTAP side mirrors this via `swap psaveslack("USA") =
    # del_tbalry("USA")` so both engines run the same closure.
    closure = GTAPClosureConfig(if_sub=False, savf_flag="capFix")

    # Baseline
    builder_b = GTAPModelEquations(params.sets, params, residual_region="ROW", closure=closure)
    model_b = builder_b.build_model()
    _solve(model_b, params, label="base")
    base_macro = _extract_key(model_b, params)
    base_snap = _collect_key_quantities(model_b, params, scale_for_gams=True)
    _augment_snapshot(model_b, base_snap)

    # Check (warm-start, no shock) — needed for the iterloop convergence pattern
    builder_c = GTAPModelEquations(params.sets, params, residual_region="ROW", closure=closure)
    model_c = builder_c.build_model()
    _copy_var_levels(model_b, model_c)
    _solve(model_c, params, label="check")

    # ------------------------------------------------------------------
    # Homotopy: split the 10% tariff shock into N=4 equal increments and
    # solve at each, so the welfare decomposition can integrate Δuh/Δug/Δus
    # along the path. RunGTAP does this implicitly via Gragg extrapolation;
    # equilibria needs the explicit path to pick up CDE curvature in yc.
    # ------------------------------------------------------------------
    HOMOTOPY_N = 4
    step_params_list = [base_params]
    step_snaps = [base_snap]
    prev_model = model_c

    for k in range(1, HOMOTOPY_N + 1):
        cumulative_factor = 1.0 + 0.10 * (k / HOMOTOPY_N)  # 1.025, 1.050, 1.075, 1.100
        step_params = GTAPParameters()
        step_params.load_from_har(
            basedata_path=NUS333 / "basedata.har",
            sets_path=NUS333 / "sets.har",
            default_path=NUS333 / "default.prm",
            baserate_path=NUS333 / "baserate.har",
        )
        _apply_tariff_shock(step_params, factor=cumulative_factor)

        builder_k = GTAPModelEquations(
            step_params.sets, step_params, residual_region="ROW",
            closure=closure, t0_snapshot=model_b,
        )
        model_k = builder_k.build_model()
        _copy_var_levels(prev_model, model_k)
        _solve(model_k, step_params, label=f"shock_step{k}")
        snap_k = _collect_key_quantities(model_k, step_params, scale_for_gams=True)
        _augment_snapshot(model_k, snap_k)

        step_params_list.append(step_params)
        step_snaps.append(snap_k)
        prev_model = model_k

    model_s = prev_model           # final shocked model
    params = step_params_list[-1]  # final shocked params (kept as `params` for downstream)
    shock_macro = _extract_key(model_s, params)
    shock_snap = step_snaps[-1]

    # Welfare decomposition — path-integrated over N=4 homotopy steps. The
    # local yc/yg/rsav at each step weight that step's Δuh/Δug/Δus, so the
    # CDE curvature in the private branch (where yc rises with the shock
    # via tariff revenue) is captured the same way RunGTAP's Gragg
    # extrapolation captures it on the linearized side.
    welfare = compute_welfare_decomposition_homotopy(step_params_list, step_snaps)

    # --- Results ---
    regions = list(params.sets.r)
    macro_table = {
        "regions": regions,
        "base": base_macro,
        "shock": shock_macro,
        "pct": {
            v: {
                r: ((shock_macro[v][r] / base_macro[v][r] - 1.0) * 100.0
                    if base_macro[v][r] else float("nan"))
                for r in regions
            }
            for v in base_macro
        },
    }

    welfare_table = {}
    for r in regions:
        w = welfare[r]
        welfare_table[r] = {
            "EV_USDm": w.EV,
            "EV_priv": w.EV_priv,
            "EV_gov":  w.EV_gov,
            "EV_save": w.EV_save,
            "A_total": w.A_total,
            "T_terms_of_trade": w.T,
            "IS_invest_saving": w.IS,
            "ENDW": w.ENDW,
            "TECH": w.TECH,
            "total_recon": w.total,
            "residual": w.residual,
            "residual_pct_of_EV": (100.0 * w.residual / w.EV) if abs(w.EV) > 1e-9 else 0.0,
        }

    out = {
        "scenario": "NUS333 uniform 10% shock to import tariff power (tm)",
        "macro": macro_table,
        "welfare_huff": welfare_table,
    }
    out_path = OUT_DIR / "equilibria_nus333_tm10.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {out_path}")

    print("\n=== Equilibria macro percent change ===")
    print(f"{'var':<8}{'region':<8}{'base':>14}{'shock':>14}{'Δ %':>10}")
    for v in ("gdpmp", "regy", "u"):
        for r in regions:
            b = base_macro[v][r]; s = shock_macro[v][r]
            d = (s / b - 1.0) * 100 if b else float("nan")
            print(f"{v:<8}{r:<8}{b:>14.4f}{s:>14.4f}{d:>+9.4f}%")

    print("\n=== Equilibria welfare (Huff decomposition, USD millions) ===")
    print(f"{'region':<8}{'EV':>14}{'A_tot':>14}{'T (ToT)':>12}{'IS':>12}{'resid%':>10}")
    for r in regions:
        w = welfare[r]
        rp = (100.0 * w.residual / w.EV) if abs(w.EV) > 1e-9 else 0.0
        print(f"{r:<8}{w.EV:>14.2f}{w.A_total:>14.2f}{w.T:>12.2f}{w.IS:>12.2f}{rp:>+9.2f}%")

    return 0


if __name__ == "__main__":
    sys.exit(main())
