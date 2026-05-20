"""9x10 head-to-head: equilibria (Python MCP) vs RunGTAP under capFix closure.

Mirrors `compare_nus333_rungtap.py` for the 9x10 dataset. Runs:

  1. equilibria baseline + 10% uniform `tm_pct` shock with capFix closure
     (residual_region = "NAmerica").
  2. Reads RunGTAP's solved values (u, EV, dpsave) from
     `runs/9x10_compare/rungtap/sl4dump.xls`.
  3. Side-by-side per-region comparison of macros (u, gdpmp, regy) and
     EV (via welfare_shadow with RunGTAP-supplied dpsave).

Writes JSON output to `runs/9x10_compare/equilibria/equilibria_9x10_tm10.json`.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
SRC = ROOT / "src"
SCRIPTS_GTAP = ROOT / "scripts" / "gtap"
PATH_CAPI_SRC = ROOT.parent / "path-capi-python" / "src"
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(SCRIPTS_GTAP))
if PATH_CAPI_SRC.exists():
    sys.path.insert(0, str(PATH_CAPI_SRC))

# PATH solver lives in the GAMS install on this Windows box.
GAMS_DIR = Path(r"C:\GAMS\53")
PATH_DLL = GAMS_DIR / "path52.dll"
if PATH_DLL.exists():
    os.environ.setdefault("PATH_CAPI_LIBPATH", str(PATH_DLL))
    # GAMS-bundled PATH 5.2 on Windows has LUSOL statically linked into
    # path52.dll, so we just point PATH_CAPI_LIBLUSOL at the same file to
    # satisfy the loader's existence check; loading it twice is harmless.
    os.environ.setdefault("PATH_CAPI_LIBLUSOL", str(PATH_DLL))
    os.environ["PATH"] = str(GAMS_DIR) + os.pathsep + os.environ.get("PATH", "")

if not os.environ.get("PATH_LICENSE_STRING"):
    raise SystemExit("PATH_LICENSE_STRING is not set.")

OUT_DIR = ROOT / "runs" / "9x10_compare" / "equilibria"
OUT_DIR.mkdir(parents=True, exist_ok=True)

GDX = ROOT / "src" / "equilibria" / "templates" / "reference" / "gtap" / "data" / "basedata-9x10.gdx"
RESIDUAL_REGION = "NAmerica"

from equilibria.templates.gtap import GTAPParameters
from equilibria.templates.gtap.gtap_model_equations import GTAPModelEquations
from equilibria.templates.gtap.gtap_contract import GTAPClosureConfig

from run_gtap import (
    _build_gtap_contract_with_calibration,
    _run_path_capi_nonlinear_full,
    _collect_key_quantities,
)


def _augment_snapshot(model, snap: dict) -> None:
    from pyomo.environ import value
    for vname in ("uh", "ug", "us", "u", "cv", "ev", "pop"):
        comp = getattr(model, vname, None)
        if comp is None:
            continue
        data = {}
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
    print(f"9x10 GDX: {GDX}")
    print(f"Residual region: {RESIDUAL_REGION}")
    contract = _build_gtap_contract_with_calibration("gtap_standard7_9x10")
    # Override to capFix (capSFix is default in 9x10 contract; we want capFix
    # to mirror NUS333 + the RunGTAP CMF swap configuration).
    payload = contract.model_dump(mode="python")
    payload["closure"]["savf_flag"] = "capFix"
    payload["closure"]["if_sub"] = False
    from equilibria.templates.gtap.gtap_contract import build_gtap_contract
    contract = build_gtap_contract(payload)

    # ---- BASELINE ----
    print("\n=== equilibria 9x10 baseline ===")
    base_params = GTAPParameters()
    base_params.load_from_gdx(GDX)

    builder_b = GTAPModelEquations(
        base_params.sets, base_params,
        residual_region=RESIDUAL_REGION,
        closure=contract.closure,
    )
    model_b = builder_b.build_model()
    t0 = time.perf_counter()
    r_b = _run_path_capi_nonlinear_full(
        model_b, base_params,
        enforce_post_checks=False, strict_path_capi=False,
        closure_config=contract.closure, equation_scaling=True,
    )
    sec_b = time.perf_counter() - t0
    print(f"  residual={r_b.get('residual'):.3e}  code={r_b.get('termination_code')}  solve={sec_b:.2f}s")
    base_snap = _collect_key_quantities(model_b, base_params, scale_for_gams=True)
    _augment_snapshot(model_b, base_snap)

    # ---- SHOCK (10% tm_pct on all imptx pairs) ----
    print("\n=== equilibria 9x10 shock (10% tm_pct uniform) ===")
    shock_params = GTAPParameters()
    shock_params.load_from_gdx(GDX)
    # tm_pct: imptx_new = (1+imptx)*1.10 - 1
    for k in list(shock_params.taxes.imptx.keys()):
        cur = float(shock_params.taxes.imptx[k])
        shock_params.taxes.imptx[k] = (1.0 + cur) * 1.10 - 1.0
        rtms = getattr(shock_params.taxes, "rtms", None)
        if isinstance(rtms, dict) and k in rtms:
            rtms[k] = shock_params.taxes.imptx[k]

    builder_s = GTAPModelEquations(
        shock_params.sets, shock_params,
        residual_region=RESIDUAL_REGION,
        closure=contract.closure,
        is_counterfactual=True,
        t0_snapshot=model_b,
    )
    model_s = builder_s.build_model()
    # Warm-start from baseline.
    from pyomo.environ import Var, value
    src_vars = {v.local_name: v for v in model_b.component_objects(Var, active=True)}
    for dv in model_s.component_objects(Var, active=True):
        sv = src_vars.get(dv.local_name)
        if sv is None:
            continue
        for idx in dv:
            try:
                src_val = value(sv[idx])
                if src_val is None or dv[idx].fixed:
                    continue
                dv[idx].set_value(float(src_val))
            except (KeyError, ValueError):
                continue

    t0 = time.perf_counter()
    r_s = _run_path_capi_nonlinear_full(
        model_s, shock_params,
        enforce_post_checks=False, strict_path_capi=False,
        closure_config=contract.closure, equation_scaling=True,
    )
    sec_s = time.perf_counter() - t0
    print(f"  residual={r_s.get('residual'):.3e}  code={r_s.get('termination_code')}  solve={sec_s:.2f}s")
    shock_snap = _collect_key_quantities(model_s, shock_params, scale_for_gams=True)
    _augment_snapshot(model_s, shock_snap)

    # ---- Extract per-region macros ----
    regions = list(base_params.sets.r)
    macros = {}
    for r in regions:
        u_b = base_snap.get("u", {}).get(r, 1.0)
        u_s = shock_snap.get("u", {}).get(r, 1.0)
        gdpmp_b = float(value(model_b.gdpmp[r])) if hasattr(model_b, "gdpmp") else 0.0
        gdpmp_s = float(value(model_s.gdpmp[r])) if hasattr(model_s, "gdpmp") else 0.0
        regy_b = float(value(model_b.regy[r])) if hasattr(model_b, "regy") else 0.0
        regy_s = float(value(model_s.regy[r])) if hasattr(model_s, "regy") else 0.0
        macros[r] = {
            "u_base": u_b, "u_shock": u_s, "u_pct": (u_s - u_b) * 100.0,
            "gdpmp_base": gdpmp_b, "gdpmp_shock": gdpmp_s,
            "gdpmp_pct": (gdpmp_s / gdpmp_b - 1.0) * 100.0 if gdpmp_b else float("nan"),
            "regy_base": regy_b, "regy_shock": regy_s,
            "regy_pct": (regy_s / regy_b - 1.0) * 100.0 if regy_b else float("nan"),
        }

    out_path = OUT_DIR / "equilibria_9x10_tm10.json"
    out_path.write_text(json.dumps({
        "scenario": "9x10 uniform 10% tm_pct shock, capFix closure, NAmerica residual",
        "regions": regions,
        "macros": macros,
    }, indent=2))
    print(f"\nWrote {out_path}")

    print(f"\n{'region':<12}{'u % eq':>10}{'gdpmp % eq':>14}{'regy % eq':>12}")
    for r in regions:
        m = macros[r]
        print(f"{r:<12}{m['u_pct']:>+9.4f}%{m['gdpmp_pct']:>+13.4f}%{m['regy_pct']:>+11.4f}%")

    return 0


if __name__ == "__main__":
    sys.exit(main())
