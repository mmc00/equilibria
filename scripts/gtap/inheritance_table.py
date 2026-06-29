"""EXHAUSTIVE inter-period INHERITANCE sweep: every variable GAMS seeds from the prior
period (var.l(t)=var.l(t0)) vs what Python seeds, in one pass. The pfteq free-row trail
started here (GAMS inherits pf from the prior period; Python's seed differs).

GAMS has TWO inheritance mechanisms — keep them apart:
  A. var.l(t) = var.l(t0)        — SEED the current period's start FROM the prior period.
                                   Only a warm-start; the solve then moves the var freely.
                                   12 vars (loop t, lines ~2735-2749): px ps pft pfy/pf pa
                                   pg pi pmt pnd pva ptmg.
  B. var.fx(tsim-1) = var.l(tsim-1) — FREEZE the PRIOR (already-solved) period so the
                                   current solve doesn't disturb it. NOT a current-period
                                   anchor (pf[check]≠pf[base] proves it). 25 vars. Class B
                                   is temporal continuity, already analysed — NOT the
                                   inheritance we sweep here.

This tool reports class A: for each GAMS seed-from-prior var, FROM WHICH PERIOD does
Python seed it in the check? GAMS seeds from t0 (the prior period = base/benchmark).
Python's diff_altertax seeds the check via warmstart_from_gams(check) = the GAMS CHECK
values, i.e. from the SAME period, not the prior — a different inheritance source.

Output: var | gams_seed_source | python_seed_source | differ?

Usage: uv run python scripts/gtap/inheritance_table.py --dataset gtap7_3x3
"""
from __future__ import annotations
import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "gtap"))

DEFAULT_REFS = "/Users/marmol/proyectos2/equilibria_refs"


def _gms(dataset: str, ifsub: int = 0) -> Path:
    own = Path(f"{DEFAULT_REFS}/{dataset}_altertax_cd/model_altertax_ifsub{ifsub}.gms")
    if own.exists():
        return own
    return Path(f"{DEFAULT_REFS}/gtap7_3x3_altertax_cd/model_altertax_ifsub{ifsub}.gms")


def parse_seed_from_prior(gms_path: Path):
    """Class A: vars with `X.l(...t) = X.l(...t0)` (seed current from prior period)."""
    pat = re.compile(r"^\s*([A-Za-z_]\w*)\.l\([^)]*\bt\)\s*=\s*\1\.l\([^)]*\bt0\)\s*;")
    out = []
    for ln in gms_path.read_text().splitlines():
        m = pat.match(ln)
        if m and m.group(1) not in out:
            out.append(m.group(1))
    return out


def parse_freeze_prior(gms_path: Path):
    """Class B: vars frozen from the prior period (var.fx(tsim-1))."""
    pat = re.compile(r"^\s*([A-Za-z_]\w*)\.fx\([^)]*tsim-1\)")
    out = []
    for ln in gms_path.read_text().splitlines():
        m = pat.match(ln)
        if m and m.group(1) not in out:
            out.append(m.group(1))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="gtap7_3x3")
    ap.add_argument("--ifsub", type=int, default=0)
    args = ap.parse_args()
    gms = _gms(args.dataset, args.ifsub)

    seed = parse_seed_from_prior(gms)   # class A
    freeze = parse_freeze_prior(gms)    # class B (reference only)

    # Python's check seed source: diff_altertax.warmstart_from_gams(m_chk, gdx, "check")
    # seeds from the GAMS CHECK values (same period), NOT from the prior (base). So for
    # EVERY class-A var, GAMS seeds from t0 (prior) while Python seeds from check (same).
    # The DIFFERENCE is the inheritance source: prior-period vs same-period-GAMS-value.
    import diff_altertax as DA
    py_seeds_from = "GAMS_check_value (warmstart_from_gams, same period)"
    gams_seeds_from = "t0 = prior period (base)"

    print(f"=== INHERITANCE (class A: seed current period from prior) — {args.dataset} ===\n")
    print(f"GAMS class-A seed-from-prior vars ({len(seed)}): {seed}")
    print(f"GAMS class-B freeze-prior vars   ({len(freeze)}): {freeze}\n")
    print(f"{'var':<8}{'gams seed source':<26}{'python seed source':<46}differ?")
    print("-" * 92)
    differ = []
    for v in seed:
        # both seed the var, but from DIFFERENT periods → inheritance source differs
        d = "DIFFER (prior vs same-period)"
        differ.append(v)
        print(f"{v:<8}{gams_seeds_from:<26}{py_seeds_from:<46}{d}")
    print()
    print(f"=> {len(differ)} of {len(seed)} class-A vars have a DIFFERENT inheritance SOURCE.")
    print("   GAMS seeds the check from the PRIOR period (base); Python seeds it from the")
    print("   GAMS CHECK value (warmstart_from_gams). To test the faithful inheritance, seed")
    print("   each from the PYTHON prior-stage (m_b) value instead — one var per bench round.")
    print(f"\n   Bench candidates (one per round): {differ}")


if __name__ == "__main__":
    main()
