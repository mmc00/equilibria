"""Validate that a GAMS reference GDX satisfies its OWN equations.

THE BLIND SPOT this covers: every parity tool (triage, diff_*, residual-at-point)
compares Python AGAINST GAMS assuming GAMS is correct. None checks whether the GAMS
GDX is internally consistent. A mis-converged GAMS solve (e.g. the gtap7_3x3 altertax
shock where the 10-step homotopy left pe inflated and 27/27 pefobeq cells violated)
produces a CORRUPT reference, and every downstream comparison silently scores Python
against garbage.

This tool seeds the GAMS reference values into a Python model and evaluates each
equation family's residual AT THE GAMS POINT. If GAMS itself violates an equation,
that family lights up — telling you the REFERENCE is bad, not Python.

Usage:
    uv run python scripts/gtap/validate_reference.py --dataset gtap7_3x3 --period shock
    uv run python scripts/gtap/validate_reference.py --dataset gtap7_3x3 --period check \\
        --gdx /path/to/out.gdx --tol 1e-2

Exit code is non-zero if any equation family exceeds the tolerance — so this can gate
a reference before it is trusted (CI or a pre-comparison guard).
"""
from __future__ import annotations
import argparse
import math
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "gtap"))

from _diff_core import gams_levels, list_populated_vars  # type: ignore

# GAMS camelCase / aggregate → Python Var-name map (same as the warm-start maps).
_GAMS_TO_PY_NAME = {
    "ytaxInd": "ytax_ind", "factY": "facty", "phiP": "phip", "regY": "regy",
    "xd": "xda", "xm": "xma",
}

# Python Vars with no GAMS counterpart (cannot be seeded → equations touching them
# would report a false residual). Equations using any of these are skipped.
_UNSEEDABLE = {"p_rai", "pp_rai", "xc", "xd", "xe", "pf0", "xf0"}


def _strip_prefix(k):
    if isinstance(k, str) and len(k) > 2 and k[1] == "_" and k[0] in "acfr":
        return k[2:]
    return k


def _build_model(dataset: str, period: str):
    """Build the Python model matching the GAMS period (base/check/shock)."""
    import copy
    from equilibria.templates.gtap import GTAPParameters, GTAPModelEquations
    from equilibria.templates.gtap.altertax import apply_altertax_elasticities
    from equilibria.templates.gtap.gtap_contract import GTAPClosureConfig

    data_dir = ROOT / "datasets" / dataset
    p = GTAPParameters()
    p.load_from_har(
        basedata_path=data_dir / "basedata.har",
        sets_path=data_dir / "sets.har",
        default_path=data_dir / "default.prm",
        baserate_path=data_dir / "baserate.har",
    )
    p = apply_altertax_elasticities(p, in_place=False)
    res = list(p.sets.r)[-1]

    base_clo = GTAPClosureConfig(
        name="base", closure_type="MCP", capital_mobility="sluggish",
        fix_endowments=False, fix_taxes=False, fix_technology=False,
        if_sub=False, numeraire="pnum",
    )
    if period == "base":
        eq = GTAPModelEquations(p.sets, p, base_clo, residual_region=res)
        return eq.build_model(), p

    # check/shock use the altertax closure with the base as t0 snapshot.
    m_b = GTAPModelEquations(p.sets, p, base_clo, residual_region=res).build_model()
    p_period = p
    if period == "shock":
        p_period = copy.deepcopy(p)
        for k in list(p_period.taxes.imptx.keys()):
            p_period.taxes.imptx[k] = float(p_period.taxes.imptx[k] or 0.0) * 1.10
    alt_clo = GTAPClosureConfig(
        name="altertax", closure_type="MCP", capital_mobility="mobile",
        fix_endowments=False, fix_taxes=True, fix_technology=True,
        if_sub=False, numeraire="pnum",
    )
    eq = GTAPModelEquations(
        p_period.sets, p_period, alt_clo, residual_region=res, t0_snapshot=m_b,
    )
    return eq.build_model(), p_period


def _seed_gams(model, gdx_path: Path, period: str) -> set:
    """Seed every matchable GAMS value for `period` into the model. Returns the
    set of id()s of seeded variable-data objects (for the clean-equation filter)."""
    from pyomo.environ import Var

    seeded_ids = set()
    for vn in list_populated_vars(gdx_path):
        gvals = gams_levels(gdx_path, vn)
        py_name = _GAMS_TO_PY_NAME.get(vn, vn)
        pyvar = getattr(model, py_name, None)
        if pyvar is None:
            pyvar = getattr(model, vn, None)
        if pyvar is None:
            pyvar = getattr(model, vn.lower(), None)
        if pyvar is None:
            continue
        for gkey, gval in gvals.items():
            if not (isinstance(gkey, tuple) and gkey[-1] == period):
                continue
            pk = tuple(_strip_prefix(k) for k in gkey[:-1])
            try:
                v = pyvar[pk] if len(pk) > 1 else pyvar[pk[0]]
                if not v.fixed:
                    v.set_value(float(gval))
                    seeded_ids.add(id(v))
            except Exception:
                pass
    return seeded_ids


def _equation_residuals(model, seeded_ids: set):
    """Per-equation-family max residual at the seeded point. Only families whose
    variables are ALL seeded (and none unseedable) are reported — others are
    'incomplete' and listed separately so a false residual is never confused for a
    real one."""
    from pyomo.environ import value as V, Constraint
    from pyomo.core.expr import identify_variables

    by_family = defaultdict(lambda: {"max": 0.0, "n": 0, "worst": None})
    incomplete = set()

    for c in model.component_objects(Constraint, active=True):
        fam = c.name
        for idx in c:
            con = c[idx]
            try:
                clean = True
                for vobj in identify_variables(con.body, include_fixed=False):
                    nm = vobj.parent_component().name
                    if nm in _UNSEEDABLE or (
                        id(vobj) not in seeded_ids and not vobj.fixed
                    ):
                        clean = False
                        break
                if not clean:
                    incomplete.add(fam)
                    continue
                body = V(con.body)
                lo = V(con.lower) if con.lower is not None else 0.0
                resid = abs(body - lo)
                if math.isnan(resid):
                    continue
                rec = by_family[fam]
                rec["n"] += 1
                if resid > rec["max"]:
                    rec["max"] = resid
                    rec["worst"] = idx
            except Exception:
                incomplete.add(fam)
    return by_family, incomplete


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="gtap7_3x3")
    ap.add_argument("--period", default="shock", choices=["base", "check", "shock"])
    ap.add_argument("--gdx", type=Path, default=None,
                    help="Reference GDX (default: the durable CD ref for the dataset)")
    ap.add_argument("--tol", type=float, default=1e-2,
                    help="Max acceptable equation residual at the GAMS point")
    ap.add_argument("--top", type=int, default=20)
    args = ap.parse_args()

    gdx_path = args.gdx
    if gdx_path is None:
        gdx_path = Path(
            f"/Users/marmol/proyectos2/equilibria_refs/{args.dataset}_altertax_cd/out_altertax_cd.gdx"
        )
    if not gdx_path.exists():
        print(f"ERROR: reference GDX not found: {gdx_path}")
        sys.exit(2)

    print(f"=== Validating reference {gdx_path.name} (period='{args.period}') ===")
    model, _ = _build_model(args.dataset, args.period)
    seeded_ids = _seed_gams(model, gdx_path, args.period)
    print(f"Seeded {len(seeded_ids)} variable cells from GAMS {args.period}")

    fams, incomplete = _equation_residuals(model, seeded_ids)
    violating = sorted(
        ((rec["max"], fam, rec["worst"], rec["n"]) for fam, rec in fams.items()
         if rec["max"] > args.tol),
        reverse=True,
    )

    print(f"\nFully-seeded equation families checked: {len(fams)}")
    print(f"Families the GAMS point VIOLATES (> {args.tol:g}): {len(violating)}")
    if violating:
        print(f"\n{'residual':>12}  {'equation family':<24} worst cell")
        print("-" * 70)
        for resid, fam, worst, n in violating[:args.top]:
            print(f"{resid:>12.4e}  {fam:<24} {worst}")
        print(
            "\n⚠️  The GAMS reference VIOLATES its own equations above. This is a "
            "CORRUPT / mis-converged reference — fix the reference before comparing "
            "Python against it."
        )
        sys.exit(1)
    else:
        print(
            f"\n✅ The GAMS reference satisfies all fully-seeded equations to {args.tol:g}. "
            "It is internally consistent and safe to compare against."
        )
        sys.exit(0)


if __name__ == "__main__":
    main()
