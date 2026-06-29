"""EXHAUSTIVE pairing/free-row table: every equation in GAMS `model gtap /.../` vs
its Python state, in ONE pass. Answers "which OTHER pfteq-class differences exist?"
without chasing them one per session.

For each GAMS model entry it prints:  gams_eq | gams_pairing | python_eq | python_state | DIFFER?

GAMS pairing  = the .var it is complementary to, or FREE-ROW (listed with no .var).
python_state  = active N/total of the real Python homolog (via the explicit
                GAMS_EQ_TO_PY map + name heuristic), and whether the paired var is FIXED.
DIFFER verdict:
  - match            : GAMS pairs eq.var  AND Python keeps the homolog active (or var fixed)
  - match (free-row) : GAMS free-row      AND Python deactivates / benign-active
  - DIFFER pairing   : GAMS free-row but Python active+paired (pfteq factor-2 class), OR
                       GAMS pairs eq.var but Python fully deactivates with var FREE
  - benign           : GAMS pairs eq.var, Python deactivates BUT var is FIXED (wedge)
  - check            : homolog not found / ambiguous — inspect by hand

Usage:
  uv run python scripts/gtap/pairing_table.py --dataset gtap7_3x3 --period check
  uv run python scripts/gtap/pairing_table.py --dataset gtap7_3x3 --only-differ
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "gtap"))

import diff_mcp_pairing as MP  # reuse the parser + _py_family_state + closure + GAMS_EQ_TO_PY
import validate_reference as _vr


# free-rows that are benign when Python keeps them active (linear single-root /
# numeraire / welfare) — same set diff_mcp_pairing uses to avoid false alarms.
_BENIGN_FREE = MP.BENIGN if hasattr(MP, "BENIGN") else set()


def _classify(gams_eq_for_classify, gams_var, pyname, active, total, var_fixed_frac):
    """Return (verdict, detail) comparing GAMS pairing to the Python state."""
    if pyname is None:
        return ("check", "no Python homolog found")
    if gams_var is None:
        # GAMS FREE-ROW
        if active == 0:
            return ("match (free-row)", "Python deactivates → matches free-row")
        # Active in Python. Reuse diff_mcp_pairing's economic classifier so a benign
        # free-row (linear market-clearing / balance identity / numeraire / welfare
        # report — single-root, safe to pair) is NOT flagged as a real mismatch. Only
        # a multi-valued free-row Python solves (the pfteq factor-2 class) is DIFFER.
        kind, _sev, knote = MP._classify_free_row(gams_eq_for_classify, pyname, active, total)
        if kind == "pairing_mismatch":
            return ("DIFFER pairing", f"GAMS free-row but Python active {active}/{total} "
                    f"(multi-root — pfteq class): {knote}")
        return ("match (free-row, benign)", f"active {active}/{total} but BENIGN: {knote}")
    # GAMS PAIRS eq.var
    var_fixed = (var_fixed_frac is not None and var_fixed_frac[1] > 0
                 and var_fixed_frac[0] == var_fixed_frac[1])
    if active > 0:
        return ("match", f"GAMS pairs .{gams_var}, Python active {active}/{total}")
    # Python fully deactivated
    if var_fixed:
        return ("benign", f"eq deactivated but {gams_var} FIXED "
                f"({var_fixed_frac[0]}/{var_fixed_frac[1]}) — exogenous wedge")
    return ("DIFFER pairing", f"GAMS pairs .{gams_var} but Python deactivates AND "
            f"{gams_var} is FREE → variable unpaired (root-selection candidate)")


def _work(args) -> dict:
    gms_path, gms_is_fallback = MP._resolve_gms_path(args.dataset, args.ifsub)
    if gms_path is None:
        return dict(status="error", period=args.period,
                    headline=f"GAMS source not found for {args.dataset}",
                    violations=[], meta={"error_kind": "gams_source_missing"})
    pairs = MP._parse_gams_model(gms_path, args.model_name)
    if not pairs:
        return dict(status="error", period=args.period,
                    headline="model block not parsed", violations=[],
                    meta={"error_kind": "model_block_unparsed"})

    model, _p = _vr._build_model(args.dataset, args.period)
    if args.apply_closure:
        try:
            MP._apply_solver_closure(model, args.dataset, _p)
        except Exception as e:  # noqa: BLE001
            print(f"WARN: closure failed: {e}", file=sys.stderr)

    table = []
    n_differ = 0
    for gams_eq, gams_var in pairs:
        pyname, active, total, var_frac = MP._py_family_state(model, gams_eq, gams_var)
        verdict, detail = _classify(gams_eq, gams_var, pyname, active, total, var_frac)
        if verdict.startswith("DIFFER"):
            n_differ += 1
        table.append({
            "gams_eq": gams_eq,
            "gams_pairing": gams_var if gams_var else "FREE-ROW",
            "python_eq": pyname or "-",
            "python_active": f"{active}/{total}" if pyname else "-",
            "var_fixed": (f"{var_frac[0]}/{var_frac[1]}"
                          if var_frac and var_frac[0] else "-"),
            "verdict": verdict,
            "detail": detail,
        })

    # human-readable table → stderr (full or only-differ)
    shown = [r for r in table if (not args.only_differ or r["verdict"].startswith("DIFFER"))]
    print(f"=== PAIRING TABLE: {args.dataset} / {args.period} "
          f"({len(table)} GAMS model entries, {n_differ} DIFFER) ===", file=sys.stderr)
    print(f"{'gams_eq':<14}{'gams_pair':<12}{'python_eq':<18}{'active':<9}"
          f"{'varfix':<8}{'verdict':<16}detail", file=sys.stderr)
    for r in shown:
        print(f"{r['gams_eq']:<14}{r['gams_pairing']:<12}{r['python_eq']:<18}"
              f"{r['python_active']:<9}{r['var_fixed']:<8}{r['verdict']:<16}{r['detail']}",
              file=sys.stderr)

    differ = [r for r in table if r["verdict"].startswith("DIFFER")]
    status = "dirty" if differ else "clean"
    headline = (f"pairing table ({args.period}): {n_differ} of {len(table)} equations "
                f"DIFFER between GAMS and Python "
                f"[{', '.join(r['gams_eq'] for r in differ)}]"
                if differ else
                f"pairing table ({args.period}): all {len(table)} equations match")
    return dict(status=status, period=args.period, headline=headline,
                violations=[{"entity": r["gams_eq"], "index": [], "metric": "pairing_differ",
                             "value": 1.0, **r} for r in differ],
                meta={"gms_path": str(gms_path), "gms_is_fallback": gms_is_fallback,
                      "n_entries": len(table), "n_differ": n_differ, "table": table})


def main() -> int:
    from _parity_json import run_tool
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="gtap7_3x3")
    ap.add_argument("--model-name", default="gtap")
    ap.add_argument("--ifsub", type=int, default=0, choices=[0, 1])
    ap.add_argument("--period", default="check", choices=["base", "check", "shock"])
    ap.add_argument("--apply-closure", action="store_true", default=True)
    ap.add_argument("--no-apply-closure", dest="apply_closure", action="store_false")
    ap.add_argument("--only-differ", action="store_true",
                    help="print only the rows where GAMS and Python differ")
    args = ap.parse_args()
    return run_tool("pairing_table", args.dataset, lambda: _work(args),
                    period_hint=args.period)


if __name__ == "__main__":
    raise SystemExit(main())
