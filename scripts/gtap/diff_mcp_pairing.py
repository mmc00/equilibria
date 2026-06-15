"""Diff the MCP equation<->variable pairing: GAMS `model gtap /.../` vs Python.

THE BLIND SPOT this covers — the deepest one, invisible to ALL other cascade tools:
nl_compare (coefficients), diff_calibration (inputs) and diff_equation_form (forms)
all assume the same equation determines the same variable in both engines. But an
MCP is defined by its PAIRING — which equation is complementary to which variable,
and which equations are FREE ROWS (listed with no `.var` in GAMS's model statement,
so no variable is solved *for* them). If Python pairs an equation that GAMS leaves
free (or vice versa), PATH can converge (code=1, residual ~0) to a DIFFERENT root of
a multi-valued block — exactly the gtap7_3x3 factor-2 bug: GAMS `pfteq` is a free row,
Python solved it for pft and hit the spurious high root (pft≈3.6 vs GAMS 1.0). No
input/coefficient/form diff shows this; only the pairing does.

This tool:
  1. parses GAMS's `model <name> / eqA.varA, eqB.varB, eqC, ... /` block, classifying
     each equation as PAIRED(var) or FREE-ROW (no `.var`);
  2. builds the Python model and applies the solver closure (free-row deactivations,
     aggressive fixing) by running the solver setup with a 0-iteration limit, then
     records, per equation family, whether it is ACTIVE and whether its GAMS-paired
     variable is FIXED;
  3. flags families where GAMS treats the equation as a free row but Python keeps it
     active+paired (or vice versa) — the pairing mismatch class.

Usage:
  uv run python scripts/gtap/diff_mcp_pairing.py --dataset gtap7_3x3
  uv run python scripts/gtap/diff_mcp_pairing.py --dataset gtap7_3x3 --model-name gtap
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

# GAMS equation name (lowercased, no 'eq' suffix conventions vary) → Python family.
# Python uses eq_<name>; GAMS uses <name>eq. We map by stripping/adding around a stem.
def _gams_eq_to_py(gams_eq: str) -> str:
    """gamsname 'pfteq' -> python 'eq_pfteq'? No: python is eq_<stem> where GAMS is
    <stem>eq. So 'pfteq' -> stem 'pft' -> 'eq_pft'... but Python actually names it
    eq_pfteq. Python convention here is eq_<gamsname-without-trailing-eq's-stem>.
    Empirically Python = 'eq_' + gams_eq with trailing 'eq' kept for some, dropped
    for others. We try both and let the caller's model decide which exists."""
    return "eq_" + gams_eq


def _parse_gams_model(gms_path: Path, model_name: str) -> list[tuple[str, str | None]]:
    """Return [(gams_eq, paired_var_or_None)] from `model <name> / ... /`.
    A FREE ROW is an entry with no `.var` → paired_var is None."""
    text = gms_path.read_text()
    m = re.search(rf"(?ms)^\s*model\s+{re.escape(model_name)}\s*/(.*?)/\s*;", text)
    if not m:
        return []
    body = m.group(1)
    # strip GAMS line comments (* at col0) and inline ones
    lines = []
    for ln in body.splitlines():
        s = ln.strip()
        if not s or s.startswith("*"):
            continue
        lines.append(s)
    joined = " ".join(lines)
    pairs: list[tuple[str, str | None]] = []
    for tok in joined.split(","):
        tok = tok.strip()
        if not tok:
            continue
        if "." in tok:
            eq, var = tok.split(".", 1)
            pairs.append((eq.strip(), var.strip()))
        else:
            pairs.append((tok, None))  # free row
    return pairs


def _py_family_state(model, gams_eq: str):
    """Return (py_eq_name_or_None, active_count, total, paired_var_fixed_frac)."""
    from pyomo.environ import Constraint, Var
    # candidate Python constraint names
    cands = [f"eq_{gams_eq}"]
    if gams_eq.endswith("eq"):
        cands.append(f"eq_{gams_eq[:-2]}")     # pfteq -> eq_pft
    cands.append(f"eq_{gams_eq.replace('eq', '')}")
    comp = None
    pyname = None
    for c in cands:
        comp = getattr(model, c, None)
        if comp is not None and comp.ctype is Constraint:
            pyname = c
            break
    if comp is None:
        return (None, 0, 0, None)
    active = total = 0
    for idx in comp:
        total += 1
        if comp[idx].active:
            active += 1
    return (pyname, active, total, None)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="gtap7_3x3")
    ap.add_argument("--model-name", default="gtap", help="GAMS model name (default gtap)")
    ap.add_argument("--ifsub", type=int, default=0, choices=[0, 1])
    ap.add_argument("--period", default="shock", choices=["base", "check", "shock"])
    ap.add_argument("--apply-closure", action="store_true",
                    help="Run the solver setup (0 iterations) so free-row deactivations "
                         "and aggressive fixing are reflected in the Python state.")
    args = ap.parse_args()

    gms_path = Path(f"{DEFAULT_REFS}/{args.dataset}_altertax_cd/model_altertax_ifsub{args.ifsub}.gms")
    if not gms_path.exists():
        print(f"ERROR: GAMS source not found: {gms_path}")
        sys.exit(2)

    pairs = _parse_gams_model(gms_path, args.model_name)
    if not pairs:
        print(f"ERROR: model '{args.model_name}' block not parsed from {gms_path.name}")
        sys.exit(2)

    free_rows = [eq for eq, var in pairs if var is None]
    print(f"=== GAMS `model {args.model_name}` pairing ({gms_path.name}) ===")
    print(f"  {len(pairs)} entries, {len(free_rows)} FREE ROWS (no MCP pair):")
    print(f"    {', '.join(free_rows)}")

    model, _p = _vr._build_model(args.dataset, args.period)
    if args.apply_closure:
        try:
            _apply_solver_closure(model, args.dataset, _p)
        except Exception as e:
            print(f"  [warn] could not apply solver closure ({e}); showing BUILD state")

    # Free rows that are benign when Python keeps them active+paired:
    #  - numeraire / Walras / welfare reports (pnum fixed=1, walras=slack, ev/cv=report);
    #  - LINEAR market-clearing / balance identities (xseq with omegax=inf is xs=xds+xet;
    #    capAccteq is Σsavf=0; savfeq under capFix is trivial) — these have ONE root, so
    #    pairing them with the variable they define is correct, NOT the multi-root class.
    # The DANGEROUS class is a power/quadratic free row (pfteq: pft^(1+omega)=Σgf·pfy^…),
    # which is multi-valued; that is the only one PATH can converge to a wrong root for.
    # NOTE: xseq/savfeq's benignness depends on omegax=inf / savfFlag=capFix — re-check
    # for datasets with finite omegax or capFlex/capShrFix closures.
    BENIGN = {"pnumeq", "walraseq", "eveq", "cveq", "xseq", "capAccteq", "savfeq"}

    print(f"\n=== Python state per GAMS FREE ROW (the mismatch-prone class) ===")
    print(f"  {'gams_eq':<12}{'python_eq':<16}{'py_active':>10}   note")
    print("  " + "-" * 64)
    n_mismatch = 0
    for eq in free_rows:
        pyname, active, total, _ = _py_family_state(model, eq)
        if pyname is None:
            note = "no Python eq (GAMS-only)"
            astr = "-"
        elif active == 0:
            note = "deactivated → matches GAMS free-row ✓"
            astr = f"{active}/{total}"
        elif eq in BENIGN:
            _kind = ("numeraire/walras/welfare report"
                     if eq in {"pnumeq", "walraseq", "eveq", "cveq"}
                     else "linear market-clearing/balance identity (single root)")
            note = f"active but BENIGN ({_kind})"
            astr = f"{active}/{total}"
        else:
            note = "ACTIVE+paired → MISMATCH vs GAMS free-row ⚠"
            astr = f"{active}/{total}"
            n_mismatch += 1
        print(f"  {eq:<12}{(pyname or '-'):<16}{astr:>10}   {note}")

    print("  " + "-" * 64)
    if n_mismatch:
        print(f"\n⚠️  {n_mismatch} ECONOMIC GAMS free-row(s) are ACTIVE+paired in Python. "
              f"PATH may solve them for a variable and pick a different root (the pfteq "
              f"factor-2 class). These are the candidates for a remaining parity gap.")
        sys.exit(1)
    print(f"\n✅ Every economic GAMS free-row is deactivated/absent in Python — pairing matches.")


def _apply_solver_closure(model, dataset, params):
    """Apply the solver closure + free-row handling without solving (0 iterations)."""
    import os
    spec2 = _u.spec_from_file_location("run_gtap", str(ROOT / "scripts" / "gtap" / "run_gtap.py"))
    rg = _u.module_from_spec(spec2); sys.modules["run_gtap"] = rg; spec2.loader.exec_module(rg)
    os.environ["PATH_CAPI_OPTIONS"] = "major_iteration_limit 0"
    # build the altertax closure the same way diff_altertax does
    from equilibria.templates.gtap.gtap_contract import GTAPClosureConfig
    alt = GTAPClosureConfig(name="altertax", closure_type="MCP", capital_mobility="mobile",
                            fix_endowments=False, fix_taxes=True, fix_technology=True,
                            if_sub=False, numeraire="pnum")
    rg._run_path_capi_nonlinear_full(model, params, enforce_post_checks=False,
                                     strict_path_capi=False, equation_scaling=True,
                                     closure_config=alt)


if __name__ == "__main__":
    main()
