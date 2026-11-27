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

This tool reports class A, measured: it runs the case the MCP gate builds on the core
driver (seeded at the data/ reference, as the cascade does; --seed-from-prior as the
gate's own altertax run) and, at the solve of the check, compares each class-A var's
check level with its base level (the prior period) and with the GAMS check level.

Output: var | distance to prior | distance to GAMS check | Python's seed source

Usage: uv run python scripts/gtap/inheritance_table.py --dataset gtap7_3x3
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from equilibria.sources import find_reference

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "gtap"))

TOL = 1e-9  # relative: below it two levels are the same seed


def _gms(dataset: str, ifsub: int = 0, mode: str = "altertax") -> Path | None:
    """The model.gms of the data/ GAMS MCP reference the MCP gate compares against."""
    folder = find_reference(
        dataset, "gams", form="mcp", mode=mode, ifsub=ifsub, fbep="header"
    )
    return None if folder is None else folder / "model.gms"


def seed_source(d_prior: float, d_gams: float, tol: float = TOL) -> str:
    """Where a check level comes from, by its distance to the prior period's level
    and to the GAMS check level."""
    if d_prior <= tol and d_gams <= tol:
        return "both (they agree)"
    if d_prior <= tol:
        return "prior period"
    if d_gams <= tol:
        return "GAMS, same period"
    return "neither"


def _rel(a: float, b: float) -> float:
    return abs(a - b) / max(abs(b), 1e-12)


def _strip(k):
    return (
        k[2:]
        if isinstance(k, str) and len(k) > 2 and k[1] == "_" and k[0] in "acfr"
        else k
    )


def measure(
    dataset: str, gdx: Path, mode: str, ifsub: int, gams_vars, seed_from_prior: bool
) -> dict[str, tuple[float, float] | None]:
    """{GAMS var: (max distance check-to-base, max distance check-to-GAMS-check)} at
    the core solve of the check, before it runs; None for a var Python lacks."""
    from _core_run import solve_on_core
    from _diff_core import gams_levels  # type: ignore
    from diff_mcp_pairing import GAMS_VAR_TO_PY
    from seed_and_solve import _case

    from equilibria.templates.gtap.recipe import _by_backend

    out: dict = {}

    def spy(model):
        staged = model.staged
        if staged is not None and staged.period == "check" and not out:
            for gv in gams_vars:
                py = GAMS_VAR_TO_PY.get(gv, gv)
                if not model.has_var(py):
                    out[gv] = None
                    continue
                chk, base = model.values(py, "check"), model.values(py, "base")
                gams = {
                    tuple(_strip(x) for x in k[:-1]): v
                    for k, v in gams_levels(gdx, gv).items()
                    if isinstance(k, tuple) and k[-1] == "check"
                }
                d_prior = max(
                    (_rel(v, base[c]) for c, v in chk.items() if c in base), default=0.0
                )
                d_gams = max(
                    (_rel(v, gams[c]) for c, v in chk.items() if c in gams),
                    default=float("inf"),
                )
                out[gv] = (d_prior, d_gams)
        return _by_backend(model)

    p, gc, rr = _case(dataset, bool(ifsub), mode)
    solve_on_core(
        p,
        gc,
        rr,
        mode=mode,
        ref_gdx=gdx,
        seed_gdx=gdx,
        seed_from_prior=seed_from_prior,
        solver=spy,
    )
    return out


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
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--dataset", default="gtap7_3x3")
    ap.add_argument("--ifsub", type=int, default=0)
    ap.add_argument("--mode", default="altertax", choices=["altertax", "gtap"])
    ap.add_argument(
        "--seed-from-prior",
        action="store_true",
        help="start the check from Python's base, as the MCP gate's altertax "
        "run does (default: seeded from the reference, as the cascade)",
    )
    args = ap.parse_args()
    gms = _gms(args.dataset, args.ifsub, args.mode)
    folder = None if gms is None else gms.parent
    if gms is None or not gms.exists() or not (folder / "out.gdx").exists():
        print(f"ERROR: no data/ GAMS MCP reference for {args.dataset}", file=sys.stderr)
        raise SystemExit(2)

    seed = parse_seed_from_prior(gms)  # class A
    freeze = parse_freeze_prior(gms)  # class B (reference only)
    dist = measure(
        args.dataset,
        folder / "out.gdx",
        args.mode,
        args.ifsub,
        seed,
        args.seed_from_prior,
    )

    start = "Python's base" if args.seed_from_prior else "the reference GDX"
    print(
        f"=== INHERITANCE (class A: seed current period from prior) — {args.dataset} ==="
    )
    print(
        f"measured at the core solve of check, before it runs; check seeded from {start}\n"
    )
    print(f"GAMS class-A seed-from-prior vars ({len(seed)}): {seed}")
    print(f"GAMS class-B freeze-prior vars   ({len(freeze)}): {freeze}\n")
    print(f"{'var':<8}{'to prior':>12}{'to GAMS check':>16}  python seed source")
    print("-" * 64)
    counts: dict[str, int] = {}
    for v in seed:
        d = dist.get(v)
        if d is None:
            print(f"{v:<8}{'-':>12}{'-':>16}  not a Python variable")
            continue
        src = seed_source(*d)
        counts[src] = counts.get(src, 0) + 1
        print(f"{v:<8}{d[0]:>12.3e}{d[1]:>16.3e}  {src}")
    print()
    print(f"=> GAMS seeds all {len(seed)} from the prior period (t0); Python: {counts}")


if __name__ == "__main__":
    main()
