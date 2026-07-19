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
factY / yTaxInd / ytaxTot / regY / phi / phiP, each ytax tax stream (pt, fc, pc,
gc, ic, dt, mt, et, ...) BY REGION, and the CDE / demand block eh / bh / xcshr /
zcons BY (region, commodity). At the benchmark there is no shock and no basin
ambiguity, so a 0.04% bias is both visible and attributable to one input (e.g.
the `mt` stream that used vmsb instead of the CIF value).

ALL calibration inputs live here: this is the single source of truth for "does
Python's benchmark match GAMS before the solver runs". If everything here is ok
but a warm-start residual persists (e.g. eq_phip), the gap is in the equation
FORM or the MCP fixed-point — NOT in a calibrated input.

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
from _parity_json import make_violation, run_tool  # noqa: E402 — shared JSON contract

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

# CDE / demand-block calibration inputs indexed by (region, commodity). GAMS keys
# carry the c_ prefix and may add an hhd dim (xcshr/zcons); both are normalised.
# These drive phiP (eq_phip = Σ xcshr·eh) and the household CDE demand — a fine
# mismatch here biases phiP→yc and shows up as a check-period warm-start residual.
_RI_MAP = {
    "eh": "eh", "bh": "bh", "xcshr": "xcshr", "zcons": "zcons",
}

# Factor-block calibration inputs. These drive eq_pfeq/eq_pft/eq_pfact (the factor
# price level). A bias here inflates pf→pft→pfact (gtap7_3x3: pf diverges ~188% vs
# ifSUB=0). GAMS keys carry f_/a_ prefixes and a trailing period; py keys are bare.
# value: (python_component, n_python_index_dims) — the GAMS body is normalised to
# that many leading elements (prefixes stripped).
_FACTOR_MAP = {
    "gf": ("gf_share", 3),   # (r, f, a)
    "aft": ("aft", 2),       # (r, f)
    "xscale": ("xscale", 2), # (r, a)
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


def _strip_c(s):
    """Strip a leading GAMS set-type prefix (c_/a_/f_/r_), case-preserving."""
    if isinstance(s, str) and len(s) > 2 and s[1] == "_" and s[0] in "acfr":
        return s[2:]
    return s


def _py_indexed_value(model, comp_name, key):
    """Read model.<comp>[key] tolerating (r,i) vs (r,i,hhd) shapes."""
    from pyomo.environ import value
    comp = getattr(model, comp_name, None)
    if comp is None:
        return None
    for k in (key, key[:-1] if len(key) > 1 else key):  # try full, then drop last (hhd)
        try:
            return float(value(comp[k] if len(k) > 1 else comp[k[0]]))
        except Exception:
            continue
    return None


def _rel(py, g, tol_abs):
    d = abs(py - g)
    if abs(g) > 1e-12:
        return d / abs(g)
    return 0.0 if d < tol_abs else float("inf")


# ---------------------------------------------------------------------------
# Human-readable formatter — KEPT for debug only; writes to STDERR so it can
# never contaminate the JSON on stdout. Not called in the normal path.
# ---------------------------------------------------------------------------
def _debug_print(gdx_name, period, tol, rows, n_diff):
    print(f"=== Calibration diff: Python (period='{period}') vs {gdx_name} ===",
          file=sys.stderr)
    print(f"    tol_rel={tol:g}\n", file=sys.stderr)
    diffs = sorted([r for r in rows if r[5] == "DIFF"], key=lambda r: -r[4])
    print(f"{'symbol':<14}{'region/cell':<16}{'python':>14}{'gams':>14}"
          f"{'rel':>10}  status", file=sys.stderr)
    for sym, reg, py, g, rel, st in diffs:
        rel_s = f"{rel*100:.4f}%" if rel != float("inf") else "inf"
        print(f"{sym:<14}{reg:<16}{py:>14.6f}{g:>14.6f}{rel_s:>10}  {st}",
              file=sys.stderr)
    n_ok = sum(1 for r in rows if r[5] == "ok")
    print(f"  checked={len(rows)}  ok={n_ok}  DIFF={n_diff}", file=sys.stderr)


def _work(args) -> dict:
    gdx_path = args.gdx
    if gdx_path is None:
        gdx_path = Path(f"{DEFAULT_REFS}/{args.dataset}_altertax_cd/"
                        f"out_altertax_ifsub1.gdx")
    if not gdx_path.exists():
        return dict(status="error", period=args.period,
                    headline=f"reference GDX not found: {gdx_path}",
                    violations=[],
                    meta={"error_kind": "gdx_not_found", "gdx": str(gdx_path)})

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

    # --- CDE / demand-block inputs, per (region, commodity) ---
    for gsym, pyname in _RI_MAP.items():
        gvals = gams_levels(gdx_path, gsym)
        for gkey, gval in gvals.items():
            if not (isinstance(gkey, tuple) and gkey[-1] == args.period):
                continue
            body = gkey[:-1]  # drop the trailing period
            # body is (r, c_i) for eh/bh or (r, c_i, hhd) for xcshr/zcons.
            region = body[0]
            commodity = _strip_c(body[1])
            py = _py_indexed_value(model, pyname, (region, commodity))
            if py is None:
                continue
            rel = _rel(py, gval, args.tol_abs)
            ok = (abs(py - gval) <= args.tol_abs) or (rel <= args.tol)
            n_diff += (not ok)
            rows.append((f"{gsym}", f"{region},{commodity}", py, gval, rel,
                         "ok" if ok else "DIFF"))

    # --- factor-block inputs (gf/aft/xscale), variable arity ---
    for gsym, (pyname, ndim) in _FACTOR_MAP.items():
        gvals = gams_levels(gdx_path, gsym)
        for gkey, gval in gvals.items():
            if not (isinstance(gkey, tuple) and gkey[-1] == args.period):
                continue
            body = tuple(_strip_c(e) for e in gkey[:-1])  # drop period, strip prefixes
            if len(body) != ndim:
                continue
            py = _py_indexed_value(model, pyname, body)
            if py is None:
                continue
            rel = _rel(py, gval, args.tol_abs)
            ok = (abs(py - gval) <= args.tol_abs) or (rel <= args.tol)
            n_diff += (not ok)
            rows.append((f"{gsym}", ",".join(body), py, gval, rel,
                         "ok" if ok else "DIFF"))

    # --- top-Armington shares alphad/alpham: Python vs GAMS, per cell ---
    # The shares come from alphad=(xda/xaa)·(pdp/pa)^sigma (GAMS gms:23750). The CORRECT
    # invariant is NOT "sum to 1" — under CES with benchmark prices != 1 the shares do
    # NOT sum to 1, and that is fine (GAMS has the same). E.g. IND/VegFruit/Grains: both
    # Python AND GAMS sum to 1.1214 (pdp=pmp≈0.10, pa=1) — NOT a bug. The REAL bug
    # (gtap7_15x10 MEX/Rice/gov) is where Python's shares DIVERGE from GAMS's: Python
    # 0.767 vs GAMS 1.0000, because Python's xaa init is inflated (floored ~1e-8) while
    # GAMS calibrates with a consistent xa.l. So compare Python alphad/alpham AGAINST the
    # GAMS-stored values, not against 1. (Earlier "sum to 1" audit was wrong — it false-
    # positived on the IND CES cells where Python matches GAMS. See memory.)
    _share_cache = getattr(model, "_armington_shares_cache", None)
    if _share_cache:
        g_alphad = gams_levels(gdx_path, "alphad")
        g_alpham = gams_levels(gdx_path, "alpham")
        for (r, i, aa), (ad, am) in _share_cache.items():
            if ad + am <= 0.0:
                continue  # cell not active (no demand)
            # GAMS keys are prefixed (c_<comm>, a_<act>) + trailing period.
            gk = (str(r), f"c_{i}", aa if aa in ("hhd", "gov", "inv") else f"a_{aa}", args.period)
            gad = g_alphad.get(gk)
            gam = g_alpham.get(gk)
            if gad is None and gam is None:
                continue  # not in GAMS ref (can't compare)
            gad = gad or 0.0
            gam = gam or 0.0
            for label, py, g in (("alphad", ad, gad), ("alpham", am, gam)):
                rel = _rel(py, g, args.tol_abs)
                ok = (abs(py - g) <= args.tol_abs) or (rel <= args.tol)
                n_diff += (not ok)
                rows.append((label, f"{r},{_strip_c(i)},{aa}", py, g, rel,
                             "ok" if ok else "DIFF"))

    _debug_print(gdx_path.name, args.period, args.tol, rows, n_diff)

    # Each calibration input that exceeds tolerance = one violation. value=rel
    # (the relative bias). An infinite rel (gams≈0 but |py-gams|>tol_abs) keeps
    # value=inf (serialized as "inf" by the shared emitter) and exposes the raw
    # py/gams so the magnitude is still legible; rel_is_infinite flags it.
    diffs = sorted([r for r in rows if r[5] == "DIFF"], key=lambda r: -r[4])
    violations = []
    for sym, reg, py, g, rel, _st in diffs:
        idx = reg.split(",") if isinstance(reg, str) else [str(reg)]
        v = make_violation(sym, idx, "calib_rel", rel)
        v["py_value"] = py
        v["gams_value"] = g
        v["abs_diff"] = abs(py - g)
        v["rel_is_infinite"] = (rel == float("inf"))
        violations.append(v)

    n_ok = sum(1 for r in rows if r[5] == "ok")
    status = "dirty" if violations else "clean"
    if violations:
        worst = violations[0]
        wrel = ("inf" if worst["rel_is_infinite"]
                else f"{worst['value'] * 100:.4f}%")
        headline = (
            f"calibration diff ({args.period}): {n_diff} input(s) diverge from "
            f"GAMS > {args.tol:g}; worst {worst['entity']}{worst['index']} "
            f"py={worst['py_value']:.6f} gams={worst['gams_value']:.6f} "
            f"rel={wrel} — a benchmark bias that hides under the solve tolerance")
    else:
        headline = (
            f"calibration diff ({args.period}): all {len(rows)} checked input(s) "
            f"match GAMS to {args.tol:g} — calibration is consistent; this layer "
            f"does not explain the gap")

    return dict(
        status=status, period=args.period, headline=headline,
        violations=violations,
        meta={"gdx": str(gdx_path), "tol_rel": args.tol, "tol_abs": args.tol_abs,
              "n_checked": len(rows), "n_ok": n_ok, "n_diff": n_diff})


def main() -> int:
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
    ap.add_argument("--mode", default="altertax", choices=["altertax", "gtap"],
                    help="altertax CD (default) or pure-gtap real-CES")
    ap.add_argument("--ifsub", type=int, default=0, choices=[0, 1],
                    help="ifSUB mode (pure-gtap only)")
    args = ap.parse_args()

    # mode=gtap UNSUPPORTED: this tool targets the altertax CD model; the
    # pure-gtap shock is wired in solve_multiperiod (multi-period, in-place
    # rebuilds). Accept the flag (the orchestrator passes it) but emit an
    # honest mode_unsupported rather than diff the wrong model. Use
    # seed_and_solve --mode gtap for the pure-gtap diagnostic.
    if args.mode == "gtap":
        def _unsupported() -> dict:
            return dict(
                status="error", period=getattr(args, "period", None),
                headline=("diff_calibration does not support --mode gtap (altertax-only tool); "
                          "use seed_and_solve --mode gtap for the pure-gtap diagnostic."),
                violations=[],
                meta={"error_kind": "mode_unsupported", "mode": "gtap",
                      "ifsub": args.ifsub})
        return run_tool("diff_calibration", args.dataset, _unsupported,
                        period_hint=getattr(args, "period", None))
    # period here is a GDX-read period (default 'base'); it IS on the base/check/
    # shock axis, so pass it through as the period_hint.
    return run_tool("diff_calibration", args.dataset, lambda: _work(args),
                    period_hint=args.period)


if __name__ == "__main__":
    raise SystemExit(main())
