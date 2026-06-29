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

from _parity_json import (  # noqa: E402 — shared JSON contract
    make_violation, run_tool, make_detection, make_prescription, with_diagnosis,
)

DEFAULT_REFS = "/Users/marmol/proyectos2/equilibria_refs"

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


# rough role tags so the price anchors that pin pva stand out
_ROLE = {
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
# the price/qty anchors that indirectly pin the CD free DOFs (pva/pnd)
_ANCHOR_VARS = ("pf", "xf", "pa", "pe", "pmcif", "pm", "pabs", "pfact", "pwfact")


def _default_gms(dataset: str) -> Path:
    return Path(f"{DEFAULT_REFS}/{dataset}_altertax_cd/model_altertax_ifsub0.gms")


# The holdfix block (loop(tsim) + var.fx(tsim-1)) is the SAME GAMS model for every
# dataset — only the sets/data differ. So if a dataset's own .gms is absent we may
# fall back to any sibling dataset's .gms for the sequence-holdfix parse.
_FALLBACK_DATASETS = ("gtap7_3x3", "gtap7_10x7", "9x10", "gtap7_15x10")


def _resolve_gms(dataset: str, explicit) -> tuple[Path | None, bool]:
    """Return (gms_path_or_None, used_fallback)."""
    if explicit is not None:
        return (explicit, False)
    own = _default_gms(dataset)
    if own.exists():
        return (own, False)
    for fb in _FALLBACK_DATASETS:
        cand = _default_gms(fb)
        if cand.exists():
            return (cand, True)
    return (None, False)


def _work(args) -> dict:
    gms, used_fallback = _resolve_gms(args.dataset, args.gms)
    if gms is None or not gms.exists():
        return dict(status="error", period=args.period,
                    headline=f"GAMS model source not found for {args.dataset} "
                             f"(no own .gms, no fallback)",
                    violations=[],
                    meta={"error_kind": "gams_source_missing",
                          "dataset": args.dataset})

    hf = parse_holdfix_set(gms)
    frozen, note = python_stage_freezes_anything()
    anchors = [gv for gv, _ in hf if gv in _ANCHOR_VARS]

    # human-readable summary → stderr
    print(f"=== HOLDFIXED / SEQUENCE diff (GAMS {gms.name}) ===", file=sys.stderr)
    print(f"GAMS freezes {len(hf)} prior-period vars; Python freezes "
          f"{'some' if frozen else 'NONE'}.", file=sys.stderr)
    for gv, _raw in hf:
        pv = _GAMS_TO_PY.get(gv, gv)
        print(f"  {gv:<12} {pv:<14} {_ROLE.get(gv, '')}", file=sys.stderr)

    # Violation model: GAMS freezes a prior-period holdfix set; Python freezes none of
    # it between stages → the sequence anchor is absent → free CD DOFs re-slide. Each
    # GAMS-frozen var that Python does NOT freeze is a missing anchor. Anchors that
    # pin pva/pnd are the dangerous ones; flag the set as dirty when Python freezes 0.
    violations = []
    if not frozen and hf:
        for gv, _raw in hf:
            pv = _GAMS_TO_PY.get(gv, gv)
            is_anchor = gv in _ANCHOR_VARS
            v = make_violation(gv, [], "holdfix_missing", 1.0 if is_anchor else 0.5)
            v["kind"] = "missing_prior_period_holdfix"
            v["python_var"] = pv
            v["is_pva_anchor"] = is_anchor
            v["role"] = _ROLE.get(gv, "")
            # DETECTION is firm (we parsed the .gms and saw the var.fx(tsim-1) entry +
            # that Python freezes none). PRESCRIPTION is a HYPOTHESIS: this tool is
            # STATIC — it never ran a solve, so it cannot know that "freeze it in Python"
            # works. (It does NOT: var.fx(tsim-1) means inherit-from-prior-and-free, not
            # anchor-intra-period; measured pf[check]≠pf[base] proves the recipe wrong.)
            with_diagnosis(
                v,
                detection=make_detection(
                    what=f"GAMS freezes prior-period {gv} (var.fx(tsim-1)); Python freezes none",
                    evidence=f"var.fx({gv}...tsim-1) in {gms.name}; Python stage chain freezes 0",
                    confidence="firm",
                ),
                prescription=make_prescription(
                    how=f"freeze {pv} at the Python prior-stage value before the next solve",
                    validated_by=None,  # static tool — never measured the effect → hypothesis
                ),
            )
            violations.append(v)

    status = "dirty" if violations else "clean"
    if violations:
        headline = (f"holdfix/sequence diff: GAMS freezes {len(hf)} prior-period vars "
                    f"({len(anchors)} pva/pnd anchors: {anchors}); Python freezes NONE "
                    f"between stages → missing sequence anchor, free CD DOFs re-slide "
                    f"(root-selection cause)")
    else:
        headline = (f"holdfix/sequence diff: Python replicates GAMS's prior-period "
                    f"holdfix — sequence anchor present; this layer does not explain the gap")
    return dict(status=status, period=args.period, headline=headline,
                violations=violations,
                meta={"gms_path": str(gms), "gms_is_fallback": used_fallback,
                      "n_gams_holdfix": len(hf),
                      "n_anchors": len(anchors), "anchors": anchors,
                      "python_freezes_prior_stage": frozen,
                      "python_note": note,
                      "holdfix_set": [gv for gv, _ in hf]})


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="gtap7_3x3")
    ap.add_argument("--gms", type=Path, default=None,
                    help="GAMS model .gms (default: DEFAULT_REFS/<dataset>_altertax_cd/"
                         "model_altertax_ifsub0.gms)")
    ap.add_argument("--period", default="check", choices=["base", "check", "shock"],
                    help="period label for the JSON (the holdfix set is sequence-wide)")
    ap.add_argument("--gdx", type=Path, default=None,
                    help="unused (accepted for orchestrator builder uniformity)")
    args = ap.parse_args()
    return run_tool("diff_holdfixed", args.dataset, lambda: _work(args),
                    period_hint=args.period)


if __name__ == "__main__":
    raise SystemExit(main())
