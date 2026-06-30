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
from _parity_json import make_violation, run_tool  # noqa: E402 — shared JSON contract

# GAMS camelCase / aggregate → Python Var-name map (same as the warm-start maps).
# p_rai/pp_rai DO have GAMS counterparts (the make-matrix producer-price Vars p/pp);
# map them so eq_x/eq_peq/eq_pp_rai become checkable (they were silently un-checked
# while eq_x actually violated 0.33 — the hole this map closes).
_GAMS_TO_PY_NAME = {
    "ytaxInd": "ytax_ind", "factY": "facty", "phiP": "phip", "regY": "regy",
    "xd": "xda", "xm": "xma", "p": "p_rai", "pp": "pp_rai",
    # COMPLETE-SEEDING additions: map every GAMS Var to its Python counterpart so the
    # full GAMS point can be seeded and the residual diff shows the TRUE list of bad
    # equations in one pass (not artifacts of unseeded vars in their init state).
    "xa": "xaa",            # GAMS xa(r,i,aa) → Python xaa(r,i,aa)
    "factY".lower(): "facty",
}

# Python Vars with truly no GAMS counterpart (cannot be seeded). p_rai/pp_rai were
# REMOVED here: they map to GAMS p/pp above, so their equations are now checkable.
_UNSEEDABLE = {"xc", "xd", "xe", "pf0", "xf0"}


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
            # Try the key as-is, then a few normalisations so the WHOLE GAMS point
            # gets seeded (not just the trivially-matching vars):
            #   - scalar var: pk==() → index with None / no index
            #   - drop a singleton 'hhd' dim (GAMS uh/ev/cv/xcshr/zcons carry it, Python doesn't)
            candidates = [pk]
            if "hhd" in pk:
                candidates.append(tuple(e for e in pk if e != "hhd"))
            v = None
            for cand in candidates:
                try:
                    if len(cand) == 0:
                        # scalar Var: pyvar itself or pyvar[None]
                        v = pyvar if pyvar.is_indexed() is False else pyvar[None]
                    elif len(cand) == 1:
                        v = pyvar[cand[0]]
                    else:
                        v = pyvar[cand]
                    break
                except Exception:
                    v = None
                    continue
            try:
                if v is not None and not v.fixed:
                    v.set_value(float(gval))
                    seeded_ids.add(id(v))
            except Exception:
                pass
    return seeded_ids


def _equation_residuals(model, seeded_ids: set):
    """Per-equation-family max residual at the seeded point.

    Checks are PER CELL, not per family: a family with some unseedable cells still
    reports the residual of its CLEAN (fully-seeded) cells. Previously a single
    unseedable cell marked the WHOLE family 'incomplete' and dropped it — which
    silently hid real violations (e.g. eq_x violating 0.33 while a sibling cell had
    an unseedable var). `by_family[fam]["n_incomplete"]` counts the skipped cells so
    a partially-checked family is visible, and `fully_incomplete` lists families
    with ZERO checkable cells (cannot be validated at all — must be named, not
    silently passed)."""
    from pyomo.environ import value as V, Constraint
    from pyomo.core.expr import identify_variables

    by_family = defaultdict(lambda: {"max": 0.0, "n": 0, "worst": None, "n_incomplete": 0})

    for c in model.component_objects(Constraint, active=True):
        fam = c.name
        rec = by_family[fam]
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
                    rec["n_incomplete"] += 1
                    continue
                body = V(con.body)
                lo = V(con.lower) if con.lower is not None else 0.0
                resid = abs(body - lo)
                if math.isnan(resid):
                    continue
                rec["n"] += 1
                if resid > rec["max"]:
                    rec["max"] = resid
                    rec["worst"] = idx
            except Exception:
                rec["n_incomplete"] += 1

    # Families with NO checkable cell at all — cannot be validated, must be surfaced.
    fully_incomplete = {fam for fam, rec in by_family.items()
                        if rec["n"] == 0 and rec["n_incomplete"] > 0}
    return by_family, fully_incomplete


def _idx_to_list(idx):
    """Normalise a worst-cell idx (tuple / scalar / None) to a list for the JSON."""
    if idx is None:
        return []
    if isinstance(idx, (list, tuple)):
        return [str(x) for x in idx]
    return [str(idx)]


# ---------------------------------------------------------------------------
# Human-readable formatter — KEPT for debug only; writes to STDERR so it can
# never contaminate the JSON on stdout. Not called in the normal path.
# ---------------------------------------------------------------------------
def _debug_print(gdx_name, period, n_seeded, n_checked, n_partial,
                 fully_incomplete, violating, tol, top):
    print(f"=== Validating reference {gdx_name} (period='{period}') ===",
          file=sys.stderr)
    print(f"Seeded {n_seeded} variable cells from GAMS {period}", file=sys.stderr)
    print(f"Families with >=1 checkable cell: {n_checked} ({n_partial} partial, "
          f"{len(fully_incomplete)} fully un-checkable)", file=sys.stderr)
    print(f"Families VIOLATED (> {tol:g}): {len(violating)}", file=sys.stderr)
    for resid, fam, worst, n, n_inc in violating[:top]:
        skip = f"({n_inc} unchecked)" if n_inc else ""
        print(f"{resid:>12.4e}  {fam:<24} {str(worst):<28} {skip}",
              file=sys.stderr)


def _work(args) -> dict:
    gdx_path = args.gdx
    if gdx_path is None:
        # Default to the ifSUB=1 reference (the NEOS-generated model diff_altertax
        # compares against by default). Pass --gdx .../out_altertax_ifsub0.gdx for
        # the regenerated ifSUB=0 reference.
        gdx_path = Path(
            f"/Users/marmol/proyectos2/equilibria_refs/{args.dataset}_altertax_cd/"
            f"out_altertax_ifsub1.gdx"
        )
    if not gdx_path.exists():
        return dict(status="error", period=args.period,
                    headline=f"reference GDX not found: {gdx_path}",
                    violations=[],
                    meta={"error_kind": "gdx_not_found", "gdx": str(gdx_path)})

    model, _ = _build_model(args.dataset, args.period)
    seeded_ids = _seed_gams(model, gdx_path, args.period)
    fams, fully_incomplete = _equation_residuals(model, seeded_ids)

    violating = sorted(
        ((rec["max"], fam, rec["worst"], rec["n"], rec["n_incomplete"])
         for fam, rec in fams.items() if rec["max"] > args.tol),
        reverse=True,
    )
    n_checked = sum(1 for rec in fams.values() if rec["n"] > 0)
    n_partial = sum(1 for rec in fams.values()
                    if rec["n"] > 0 and rec["n_incomplete"] > 0)

    _debug_print(gdx_path.name, args.period, len(seeded_ids), n_checked,
                 n_partial, fully_incomplete, violating, args.tol, args.top)

    # Each VIOLATING family = one violation: the GAMS reference fails its OWN eq
    # at the seeded point → the reference is mis-converged (or Python's eq/closure
    # differs there). metric=residual; value=max residual on clean cells.
    violations = []
    for resid, fam, worst, n, n_inc in violating:
        v = make_violation(fam, _idx_to_list(worst), "residual", resid)
        v["cells_checked"] = n
        v["cells_unchecked"] = n_inc
        violations.append(v)

    status = "dirty" if violations else "clean"
    fi = sorted(fully_incomplete)
    if violations:
        worst_v = violations[0]
        headline = (
            f"reference {gdx_path.name} ({args.period}): VIOLATES {len(violations)} "
            f"of its OWN equation famil(ies) > {args.tol:g}; worst "
            f"{worst_v['entity']}{worst_v['index']}={worst_v['value']:.3e} — the "
            f"reference is mis-converged or Python's eq/closure differs there")
    else:
        # 'clean' with un-checkable families is NOT a full clean: a violation in
        # those would be invisible. Say so explicitly in the headline.
        if fi:
            headline = (
                f"reference {gdx_path.name} ({args.period}): satisfies all "
                f"{n_checked} CHECKABLE famil(ies) to {args.tol:g}, BUT "
                f"{len(fi)} famil(ies) have no fully-seeded cell — a violation "
                f"there would be invisible")
        else:
            headline = (
                f"reference {gdx_path.name} ({args.period}): satisfies all "
                f"{n_checked} equation famil(ies) to {args.tol:g} — the reference "
                f"is internally consistent")

    return dict(
        status=status, period=args.period, headline=headline,
        violations=violations,
        meta={"gdx": str(gdx_path), "tol": args.tol,
              "cells_seeded": len(seeded_ids),
              "n_families_checked": n_checked,
              "n_families_partial": n_partial,
              "n_families_violating": len(violations),
              "fully_incomplete": fi,
              "n_fully_incomplete": len(fi)})


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="gtap7_3x3")
    ap.add_argument("--period", default="shock", choices=["base", "check", "shock"])
    ap.add_argument("--gdx", type=Path, default=None,
                    help="Reference GDX (default: the durable CD ref for the dataset)")
    ap.add_argument("--tol", type=float, default=1e-2,
                    help="Max acceptable equation residual at the GAMS point")
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--mode", default="altertax", choices=["altertax", "gtap"],
                    help="altertax CD (default) or pure-gtap real-CES")
    ap.add_argument("--ifsub", type=int, default=0, choices=[0, 1],
                    help="ifSUB mode (pure-gtap only)")
    args = ap.parse_args()

    # mode=gtap UNSUPPORTED, honestly: the pure-gtap shock equations are produced by
    # solve_multiperiod(mode="gtap") via an in-place constraint-rebuild SEQUENCE
    # (_rebuild_eq_pmeq_shock / NatRes anchor / _rebuild_eq_ytax_mt_shock + an imptx
    # param shock, applied in a specific order). validate_reference checks residuals
    # on the AS-BUILT equations; without re-running that exact mutation sequence the
    # shock-period equations here would be the UN-shocked ones, so a residual verdict
    # would be measured against the wrong model. Replicating the sequence risks subtle
    # infidelity. seed_and_solve --mode gtap runs the real driver and is the faithful
    # path for the pure-gtap residual-at-GAMS question. Emit mode_unsupported.
    if args.mode == "gtap":
        def _unsupported() -> dict:
            return dict(
                status="error", period=args.period,
                headline=("validate_reference does not support --mode gtap: the "
                          "pure-gtap shock equations are built by solve_multiperiod's "
                          "in-place rebuild sequence, which this static residual check "
                          "cannot reproduce faithfully. Use seed_and_solve --mode gtap."),
                violations=[],
                meta={"error_kind": "mode_unsupported", "mode": "gtap",
                      "ifsub": args.ifsub})
        return run_tool("validate_reference", args.dataset, _unsupported,
                        period_hint=args.period)

    return run_tool("validate_reference", args.dataset, lambda: _work(args),
                    period_hint=args.period)


if __name__ == "__main__":
    raise SystemExit(main())
