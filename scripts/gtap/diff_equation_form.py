"""Show a Python equation's expanded symbolic form next to its GAMS source line.

THE BLIND SPOT this covers: nl_compare (.nl diff) compares Jacobian COEFFICIENTS at
a point — it confirms two equations evaluate the same there, but it does NOT show
the algebraic shape, so a structural difference that happens to match numerically
at the seed (or that lives in an aggregation the .nl linearises) slips through. The
gtap7_3x3 factor/quantity gap is exactly this class: every input and coefficient
matches GAMS, the .nl gate is 5/5, yet Python lands ~7% off in pf/pe/xd. The
suspicion is a per-agent scaling (1/xscale applied to some agents but not others)
in an aggregation equation — invisible to coefficient diffs but visible the moment
you read the two equations side by side.

This tool prints, for a requested Python equation family + cell:
  - the FULLY EXPANDED Pyomo expression (coefficients already substituted, so a
    stray 1/xscale shows up as a literal 10.0 or 100.0 multiplier), and
  - the matching GAMS equation block pulled verbatim from the reference .gms.

It does NOT auto-diff (symbolic equality across two languages is brittle); it puts
both forms in front of you so the structural difference is obvious. Pair it with
nl_compare (coefficients) and diff_calibration (inputs): if those are clean but a
family still diverges, read the forms here.

Usage:
  uv run python scripts/gtap/diff_equation_form.py --dataset gtap7_3x3 \\
      --eq eq_xd_agg --cell ROW,Svces --gams-eq xds
  uv run python scripts/gtap/diff_equation_form.py --dataset gtap7_3x3 \\
      --eq eq_pfteq --cell EU_28,SkLab --gams-eq pfteq --period shock
"""
from __future__ import annotations
import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "gtap"))

import importlib.util as _u
_spec = _u.spec_from_file_location(
    "validate_reference", str(ROOT / "scripts" / "gtap" / "validate_reference.py")
)
_vr = _u.module_from_spec(_spec)
sys.modules["validate_reference"] = _vr
_spec.loader.exec_module(_vr)

DEFAULT_REFS = "/Users/marmol/proyectos2/equilibria_refs"


def _default_gms(dataset: str, ifsub: int) -> Path:
    return Path(f"{DEFAULT_REFS}/{dataset}_altertax_cd/model_altertax_ifsub{ifsub}.gms")


def _extract_gams_eq(gms_path: Path, eq_name: str) -> str:
    """Pull the GAMS equation DEFINITION block `eq_name(...).. <body with =X=> ;`.

    Must skip the `Equations` declaration section (where the name appears followed
    by a quoted description, no `..`), and require a relational operator (=e=/=g=/
    =l=) in the body so a declaration is never mistaken for the definition.
    """
    text = gms_path.read_text()
    # All `name ... .. ... ;` blocks; keep the one whose body has a relational op.
    candidates = re.finditer(
        rf"(?m)^\s*{re.escape(eq_name)}\s*(\([^)]*\))?[^.;]*\.\.(.*?);", text, re.S
    )
    for m in candidates:
        body = m.group(2)
        if re.search(r"=[eEgGlL]=", body):
            block = m.group(0).strip()
            return "\n".join(ln.rstrip() for ln in block.splitlines() if ln.strip())
    return f"(GAMS equation definition '{eq_name}' not found in {gms_path.name})"


def _parse_cell(cell: str):
    if not cell:
        return None
    parts = tuple(p.strip() for p in cell.split(","))
    return parts[0] if len(parts) == 1 else parts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="gtap7_3x3")
    ap.add_argument("--eq", required=True, help="Python constraint family, e.g. eq_xd_agg")
    ap.add_argument("--cell", default=None,
                    help="Index cell, comma-separated, e.g. ROW,Svces (default: first active)")
    ap.add_argument("--gams-eq", default=None,
                    help="GAMS equation name to show alongside (e.g. xds, xdeq, pfteq). "
                         "Default: strip the eq_ prefix and append 'eq'.")
    ap.add_argument("--period", default="shock", choices=["base", "check", "shock"])
    ap.add_argument("--ifsub", type=int, default=0, choices=[0, 1])
    args = ap.parse_args()

    from pyomo.environ import Constraint

    model, _p = _vr._build_model(args.dataset, args.period)
    con_comp = getattr(model, args.eq, None)
    if con_comp is None:
        print(f"ERROR: Python constraint '{args.eq}' not found on the model.")
        sys.exit(2)

    cell = _parse_cell(args.cell)
    if cell is None:
        # first active index
        for idx in con_comp:
            if con_comp[idx].active:
                cell = idx
                break

    try:
        con = con_comp[cell]
    except Exception as e:
        print(f"ERROR: cell {cell!r} not in {args.eq}: {e}")
        sys.exit(2)

    print("=" * 78)
    print(f"PYTHON  {args.eq}[{cell}]  (period={args.period}, expanded — 1/xscale shows as a literal)")
    print("=" * 78)
    print(f"  {con.expr}")

    gams_eq = args.gams_eq or (args.eq[3:] + "eq" if args.eq.startswith("eq_") else args.eq)
    gms_path = _default_gms(args.dataset, args.ifsub)
    print()
    print("=" * 78)
    print(f"GAMS    {gams_eq}   (from {gms_path.name})")
    print("=" * 78)
    print(_extract_gams_eq(gms_path, gams_eq))
    print()
    print("Compare the shapes above. A 1/xscale (literal 10.0/100.0) on some terms")
    print("but not others, an extra/missing factor, or a different sum domain is the")
    print("kind of structural difference the .nl coefficient diff does not surface.")


if __name__ == "__main__":
    main()
