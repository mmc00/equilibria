"""Diff Python's CALIBRATION INPUTS against GAMS, per region and per tax stream.

THE BLIND SPOT this covers: the four-tool parity cascade
(diff_altertax / triage / validate_reference / nl_compare) all look at the SOLVED
shock, equation residuals at a point, algebra, or closure. None of them isolates a
mis-calibrated benchmark INPUT (a beta share, factY, yTaxInd, or a single ytax tax
stream). A small calibration bias hides three ways:

  1. diff_altertax scores the solved value against tol_rel=1e-3, so a 0.04% betaP
     error shows as "ok" — the comparator never flags it.
  2. validate_reference DOES see the downstream residual (eq_yc/eq_rsav/eq_yg) but
     attributes it to a "corrupt reference", not to a Python input.
  3. nl_compare is blind to parameter VALUES; the .nl gate stays green.

So the bias only surfaces, diluted and amplified, in downstream solved variables
(yc, ytax) where it is indistinguishable from CD-basin noise.

This tool compares the BENCHMARK (period='base') calibration inputs directly:
the regional expenditure shares betaP/betaG/betaS, the income aggregates
factY / yTaxInd / ytaxTot / regY / phi / phiP, and — crucially — each ytax tax
stream (pt, fc, pc, gc, ic, dt, mt, et, ...) BY REGION. At the benchmark there is
no shock and no basin ambiguity, so a 0.04% bias is both visible and attributable
to one stream (e.g. the `mt` stream that used vmsb instead of the CIF value).

Worked precedent (2026-06-14): Python's `ytax[ROW,'mt']` was 0.338282 vs GAMS
0.323560 (imptx·vmsb instead of imptx·VCIF). It biased yTaxInd→regY→betaP by 0.04%,
slipping under diff_altertax's tolerance for ~5 sessions. This tool would have
flagged `ytax[ROW,mt]  py=0.338282  gams=0.323560  rel=4.55%` on the first run.

Usage:
    uv run python scripts/gtap/diff_calibration.py --dataset gtap7_3x3
    uv run python scripts/gtap/diff_calibration.py --dataset gtap7_3x3 \\
        --gdx /path/to/out_altertax_ifsub0.gdx --tol 1e-4

Exit code is non-zero if any calibration input exceeds the tolerance, so this can
gate calibration before a parity comparison is trusted.
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "gtap"))

from _diff_core import gams_levels  # type: ignore

# Reuse validate_reference's model builder so the closure/elasticities match exactly.
import importlib.util as _u
_spec = _u.spec_from_file_location(
    "validate_reference", str(ROOT / "scripts" / "gtap" / "validate_reference.py")
)
_vr = _u.module_from_spec(_spec)
sys.modules["validate_reference"] = _vr
_spec.loader.exec_module(_vr)

DEFAULT_REFS = "/Users/marmol/proyectos2/equilibria_refs"

# GAMS symbol → Python component name (calibration scalars, indexed by region).
_SCALAR_MAP = {
    "betaP": "betap", "betaG": "betag", "betaS": "betas",
    "factY": "facty", "yTaxInd": "ytax_ind", "ytaxTot": "ytaxTot",
    "regY": "regy", "phi": "phi", "phiP": "phip",
}


def _py_region_value(model, comp_name, region):
    from pyomo.environ import value
    comp = getattr(model, comp_name, None)
    if comp is None:
        return None
    try:
        return float(value(comp[region]))
    except Exception:
        return None


def _py_ytax_value(model, region, gy):
    from pyomo.environ import value
    comp = getattr(model, "ytax", None)
    if comp is None:
        return None
    try:
        return float(value(comp[region, gy]))
    except Exception:
        return None


def _rel(py, g, tol_abs):
    d = abs(py - g)
    if abs(g) > 1e-12:
        return d / abs(g)
    return 0.0 if d < tol_abs else float("inf")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="gtap7_3x3")
    ap.add_argument("--gdx", type=Path, default=None,
                    help="GAMS reference GDX (default: the dataset's ifsub1 ref)")
    ap.add_argument("--period", default="base",
                    help="GDX period to read GAMS calibration values from (default base)")
    ap.add_argument("--tol", type=float, default=1e-4,
                    help="Relative tolerance (default 1e-4 = 0.01%%; tighter than the "
                         "solve comparator's 1e-3 so calibration bias is visible)")
    ap.add_argument("--tol-abs", type=float, default=1e-9)
    args = ap.parse_args()

    gdx_path = args.gdx
    if gdx_path is None:
        gdx_path = Path(f"{DEFAULT_REFS}/{args.dataset}_altertax_cd/out_altertax_ifsub1.gdx")
    if not gdx_path.exists():
        print(f"ERROR: reference GDX not found: {gdx_path}")
        sys.exit(2)

    print(f"=== Calibration diff: Python (period='{args.period}') vs {gdx_path.name} ===")
    print(f"    tol_rel={args.tol:g}\n")

    # Build the Python benchmark model. The base period carries the calibrated
    # betas / income aggregates / ytax streams with no shock applied.
    model, _params = _vr._build_model(args.dataset, args.period)

    rows = []   # (symbol, region/stream, py, gams, rel, status)
    n_diff = 0

    # regY is reported income-side by Python (factY + yTaxInd) but the GAMS GDX
    # stores regY FIXED to the expenditure side (yc+yg+rsav, cal.gms:3331). Compare
    # Python's regY against GAMS factY+yTaxInd (income-side, like-for-like) instead
    # of the stored regY, so the income/expenditure SAM imbalance is not a false DIFF.
    g_facty = gams_levels(gdx_path, "factY")
    g_ytaxind = gams_levels(gdx_path, "yTaxInd")

    def _gams_target(gsym, gkey, gval):
        if gsym != "regY":
            return gval
        fy = g_facty.get(gkey)
        yt = g_ytaxind.get(gkey)
        if fy is not None and yt is not None:
            return fy + yt
        return gval

    # --- scalar calibration inputs, per region ---
    for gsym, pyname in _SCALAR_MAP.items():
        gvals = gams_levels(gdx_path, gsym)
        for gkey, gval in gvals.items():
            if not (isinstance(gkey, tuple) and gkey[-1] == args.period):
                continue
            region = gkey[0]
            py = _py_region_value(model, pyname, region)
            if py is None:
                continue
            target = _gams_target(gsym, gkey, gval)
            rel = _rel(py, target, args.tol_abs)
            ok = (abs(py - target) <= args.tol_abs) or (rel <= args.tol)
            n_diff += (not ok)
            rows.append((gsym, region, py, target, rel, "ok" if ok else "DIFF"))

    # --- ytax tax streams, per (region, stream) ---
    gytax = gams_levels(gdx_path, "ytax")
    for gkey, gval in gytax.items():
        if not (isinstance(gkey, tuple) and gkey[-1] == args.period):
            continue
        region, gy = gkey[0], gkey[1]
        py = _py_ytax_value(model, region, gy)
        if py is None:
            continue
        rel = _rel(py, gval, args.tol_abs)
        ok = (abs(py - gval) <= args.tol_abs) or (rel <= args.tol)
        n_diff += (not ok)
        rows.append((f"ytax[{gy}]", region, py, gval, rel, "ok" if ok else "DIFF"))

    # Print: DIFFs first (sorted by rel desc), then a compact ok summary.
    diffs = sorted([r for r in rows if r[5] == "DIFF"], key=lambda r: -r[4])
    print(f"{'symbol':<14}{'region':<10}{'python':>14}{'gams':>14}{'rel':>10}  status")
    print("-" * 74)
    for sym, reg, py, g, rel, st in diffs:
        rel_s = f"{rel*100:.4f}%" if rel != float("inf") else "inf"
        print(f"{sym:<14}{reg:<10}{py:>14.6f}{g:>14.6f}{rel_s:>10}  {st}")
    if not diffs:
        print("  (no calibration inputs exceed tolerance)")

    n_ok = sum(1 for r in rows if r[5] == "ok")
    print("-" * 74)
    print(f"  checked={len(rows)}  ok={n_ok}  DIFF={n_diff}")

    if n_diff:
        print(f"\n⚠️  {n_diff} calibration input(s) diverge from GAMS at the benchmark. "
              f"A bias here propagates into the solved shock and HIDES under the "
              f"solve comparator's tolerance — fix the calibration before trusting parity.")
        sys.exit(1)
    print(f"\n✅ All calibration inputs match GAMS to {args.tol:g}.")


if __name__ == "__main__":
    main()
