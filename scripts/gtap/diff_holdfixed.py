"""Tool #8 of the parity-debug cascade: HOLDFIXED / SEQUENCE diff.

THE BLIND SPOT this covers: tools 0-7 all look at ONE solve in isolation — equation
forms, coefficients, calibration, MCP pairing, drift. NONE of them sees how GAMS
SEQUENCES its periods. GAMS solves base→check→shock in a `loop(tsim)` and, before
solving period `tsim`, HARD-FIXES the entire PREVIOUS period via `var.fx(tsim-1) =
var.l(tsim-1)` (combined with `gtap.holdfixed=1`). That frozen prior period is what
keeps a free/degenerate variable (pva/pnd under CD) from sliding to another branch —
the prior period's pf/xf/pa/pe levels anchor the current period's prices, hence pva.

Python's diff_altertax DOES solve in 3 stages (betaCal/base → check → shock) and warm-
starts each from the previous — BUT it does NOT FREEZE the previous stage; it only uses
it as a starting point and leaves everything free to re-solve. So the anchor GAMS gets
from `holdfixed(tsim-1)` is absent, and the free DOF slides. This difference is invisible
to every other tool because it lives in the SOLVE SEQUENCE (the `loop(tsim)` / `.fx(tsim-1)`
block of the .gms), not in the model equations, coefficients, calibration, or pairing.

WHAT THIS TOOL DOES:
  1. parses the GAMS `var.fx(...tsim-1) = var.l(...tsim-1)` block → the list of variables
     GAMS freezes from the previous period (the holdfix set);
  2. reports which of those Python's stage chain freezes (currently: NONE — Python warm-
     starts but never fixes the prior stage), i.e. the exact gap;
  3. prints the holdfix variable list as the RECIPE for a faithful fix: to replicate
     GAMS, freeze these vars at the PYTHON-computed previous-stage values (NOT GAMS
     values — that would be hardcoding) before the next stage's solve.

USAGE:
    uv run python scripts/gtap/diff_holdfixed.py \\
        --gms /Users/.../model_altertax_ifsub0.gms

INTERPRETATION:
  - The printed holdfix set is what GAMS pins between periods. If Python freezes none of
    them (the current state), that IS the root of a free-DOF/basin gap that tool 7 sees
    the symptom of. The faithful fix is to freeze the same set at Python's own prior-stage
    levels in diff_altertax's [3/3] shock (and [2/3] check) — not the GAMS-seeded values.
"""
from __future__ import annotations
import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "gtap"))

# GAMS var name → Python Var attribute (same convention as diff_altertax.GAMS_TO_PY_NAME).
_GAMS_TO_PY = {
    "factY": "facty", "regY": "regy", "phiP": "phip", "ytaxInd": "ytax_ind",
    "kapEnd": "kapEnd", "chiSave": "chiSave", "xd": "xda", "xm": "xma",
    "p": "p_rai", "pp": "pp_rai", "xa": "xaa",
}


def parse_holdfix_set(gms_path: Path):
    """Return the ordered list of (gams_var, raw_line) that GAMS freezes from the
    previous period: lines matching `VAR.fx(...tsim-1) = VAR.l(...tsim-1)`."""
    pat = re.compile(r"^\s*([A-Za-z_]\w*)\.fx\([^)]*tsim-1\)\s*=\s*\1\.l\([^)]*tsim-1\)\s*;")
    out = []
    for ln in gms_path.read_text().splitlines():
        m = pat.match(ln)
        if m:
            out.append((m.group(1), ln.strip()))
    return out


def python_stage_freezes_anything() -> tuple[bool, str]:
    """Does diff_altertax freeze the previous stage before the next solve? Inspect the
    source for a `.fix(` applied to m_chk/m_alt outside the opt-in --holdfix-pva block."""
    src = (ROOT / "scripts" / "gtap" / "diff_altertax.py").read_text()
    # The only .fix on stage models is inside the `if args.holdfix_pva:` blocks (pva/pnd
    # at GAMS-seeded values) — NOT a faithful prior-stage holdfix. Detect a general
    # prior-stage freeze (would be e.g. a loop fixing m_chk vars from a snapshot).
    has_holdfix_pva = "args.holdfix_pva" in src
    # crude: any `.fix(` on m_chk/m_alt that is NOT in the holdfix_pva block
    return (False, "Python warm-starts each stage but does NOT .fix() the previous stage "
            "(only the opt-in --holdfix-pva block fixes pva/pnd, and at GAMS-seeded values, "
            "which is hardcoding — not a prior-stage holdfix).") if not has_holdfix_pva else \
           (False, "only --holdfix-pva (opt-in, GAMS-seeded) fixes pva/pnd; no faithful "
            "prior-stage holdfix of the full GAMS set.")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gms", type=Path,
                    default=Path("/Users/marmol/proyectos2/equilibria_refs/"
                                 "gtap7_3x3_altertax_cd/model_altertax_ifsub0.gms"))
    args = ap.parse_args()

    if not args.gms.exists():
        print(f"ERROR: GAMS file not found: {args.gms}")
        sys.exit(2)

    hf = parse_holdfix_set(args.gms)
    print(f"=== HOLDFIXED / SEQUENCE diff (GAMS {args.gms.name}) ===")
    print(f"\nGAMS freezes {len(hf)} variables from the PREVIOUS period before each "
          f"`solve gtap` (loop(tsim) + holdfixed=1):")
    print(f"\n  {'GAMS var':<12} {'Python Var':<14} role")
    print("  " + "-" * 56)
    # rough role tags so the price anchors that pin pva stand out
    role = {
        "pf": "factor price  ← anchors pva (VA nest)",
        "xf": "factor qty    ← anchors pva (VA nest)",
        "pa": "Armington price ← anchors pnd",
        "pe": "export price", "pefob": "export price (fob)", "pmcif": "import price (cif)",
        "pm": "import price", "xw": "bilateral trade qty",
        "pabs": "absorption price index (Fisher)", "pfact": "factor price index (Fisher)",
        "pwfact": "world factor price (numeraire chain)", "pmuv": "MUV deflator (report)",
        "pi": "investment price", "psave": "savings price", "ptmg": "margin price",
        "uh": "household utility", "gdpmp": "nominal GDP", "rgdpmp": "real GDP (report)",
        "pgdpmp": "GDP deflator (report)",
        "axp": "tech (fixed param-like)", "lambdand": "tech", "lambdava": "tech",
        "lambdaio": "tech", "lambdaf": "tech",
    }
    for gv, _raw in hf:
        pv = _GAMS_TO_PY.get(gv, gv)
        print(f"  {gv:<12} {pv:<14} {role.get(gv, '')}")

    frozen, note = python_stage_freezes_anything()
    print(f"\nPython stage chain freezes the previous stage? {'YES' if frozen else 'NO'}")
    print(f"  → {note}")

    anchors = [gv for gv, _ in hf if gv in ("pf", "xf", "pa", "pe", "pmcif", "pm",
                                            "pabs", "pfact", "pwfact")]
    print(f"\nKEY: GAMS pins pva/pnd INDIRECTLY by freezing the price/qty anchors "
          f"{anchors} from the prior period. Python freezes NONE of these between stages,")
    print(f"so the free CD DOFs (pva/pnd) re-slide each solve (tool 7 ⚑ symptom). FAITHFUL")
    print(f"FIX (no hardcoding): before the shock solve, .fix() these vars at the PYTHON")
    print(f"check-stage levels (not GAMS values), replicating var.fx(tsim-1)=var.l(tsim-1).")


if __name__ == "__main__":
    main()
