"""Tool #8 of the parity-debug cascade: HOLDFIXED / SEQUENCE diff.

GAMS solves base -> check -> shock in a `loop(tsim)` and, before solving period
`tsim`, fixes the previous one: `var.fx(tsim-1) = var.l(tsim-1)` (with
`gtap.holdfixed=1`). This tool checks that Python does the same, by measuring it.

WHAT THIS TOOL DOES:
  1. parses the `var.fx(...tsim-1) = var.l(...tsim-1)` lines of the reference's
     model.gms -> the variables GAMS holds fixed from the previous period;
  2. runs the case on the core driver as the MCP gate builds it (seed_and_solve's
     `_case`, seeded from the reference GDX) and, at every solve, counts the cells
     of those variables left free in any period but the one being solved;
  3. is dirty only if a solve of the requested period left one free.

The core's `freeze_inactive` step fixes every period but the solved one, which
covers GAMS's holdfix of `tsim-1` (MAR-219). Base has no previous period.

USAGE:
    uv run python scripts/gtap/diff_holdfixed.py --dataset gtap7_3x3 --period shock
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
from equilibria.sources import find_reference

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


def _reference(dataset: str, mode: str, ifsub: int):
    """The data/ GAMS MCP reference folder the MCP gate compares against."""
    return find_reference(dataset, "gams", form="mcp", mode=mode, ifsub=ifsub,
                          fbep="header")


def _resolve_gms(dataset: str, explicit, mode: str = "altertax",
                 ifsub: int = 0) -> tuple[Path | None, bool]:
    """Return (gms_path_or_None, used_fallback): the reference's own model.gms."""
    if explicit is not None:
        return (explicit, False)
    folder = _reference(dataset, mode, ifsub)
    return (None if folder is None else folder / "model.gms", False)


def unfrozen_outside(model, names) -> tuple[str | None, dict[str, dict[str, int]]]:
    """At a solve: (the solved period, {other period: {var: free cells}}). The solved
    period is the one whose rows are on; names Python lacks are skipped."""
    rows = model.row_names()
    active = [t for t in model.periods
              if any(model.active_rows(r, t) for r in rows)]
    solved = active[0] if len(active) == 1 else None
    free: dict[str, dict[str, int]] = {}
    for t in model.periods:
        if t == solved:
            continue
        for n in names:
            if not model.has_var(n):
                continue
            k = len(model.var_cells(n, t)) - len(model.fixed_cells(n, t))
            if k:
                free.setdefault(t, {})[n] = k
    return solved, free


def measure_holdfix(dataset: str, gdx: Path, mode: str, ifsub: int,
                    gams_names) -> dict:
    """Run the case on the core driver and record ``unfrozen_outside`` at every
    solve: ``{"solves": [(period, free)], "absent": [python names not in the
    model]}``."""
    from _core_run import solve_on_core
    from seed_and_solve import _case
    from equilibria.templates.gtap.recipe import _by_backend

    names = [_GAMS_TO_PY.get(g, g) for g in gams_names]
    solves: list = []
    absent: set[str] = set()

    def spy(model):
        absent.update(n for n in names if not model.has_var(n))
        solves.append(unfrozen_outside(model, names))
        return _by_backend(model)

    p, gc, rr = _case(dataset, bool(ifsub), mode)
    solve_on_core(p, gc, rr, mode=mode, ref_gdx=gdx, seed_gdx=gdx, solver=spy)
    return {"solves": solves, "absent": sorted(absent)}


def _work(args) -> dict:
    gms, used_fallback = _resolve_gms(args.dataset, args.gms, args.mode, args.ifsub)
    if gms is None or not gms.exists():
        return dict(status="error", period=args.period,
                    headline=f"GAMS model source not found for {args.dataset}",
                    violations=[],
                    meta={"error_kind": "gams_source_missing",
                          "dataset": args.dataset})
    hf = [gv for gv, _ in parse_holdfix_set(gms)]
    meta = {"gms_path": str(gms), "gms_is_fallback": used_fallback,
            "n_gams_holdfix": len(hf), "holdfix_set": hf}
    if args.period == "base":
        return dict(status="clean", period="base", violations=[], meta=meta,
                    headline="holdfix/sequence diff: base has no previous period "
                             "to hold fixed")
    gdx = args.gdx
    if gdx is None:
        folder = _reference(args.dataset, args.mode, args.ifsub)
        gdx = None if folder is None else folder / "out.gdx"
    if gdx is None or not Path(gdx).exists():
        return dict(status="error", period=args.period,
                    headline=f"reference GDX not found for {args.dataset}",
                    violations=[], meta={**meta, "error_kind": "gdx_not_found"})

    measured = measure_holdfix(args.dataset, Path(gdx), args.mode, args.ifsub, hf)
    mine = [free for solved, free in measured["solves"] if solved == args.period]
    meta.update(absent_in_python=measured["absent"], n_solves=len(mine))
    if not mine:
        return dict(status="error", period=args.period, violations=[],
                    headline=f"holdfix/sequence diff: no solve of {args.period} "
                             f"was observed, nothing measured",
                    meta={**meta, "error_kind": "solve_not_observed"})

    worst: dict[str, int] = {}
    for free in mine:
        for cells in free.values():
            for n, k in cells.items():
                worst[n] = max(worst.get(n, 0), k)
    py_to_gams = {_GAMS_TO_PY.get(g, g): g for g in hf}
    violations = []
    for n, k in sorted(worst.items()):
        gv = py_to_gams.get(n, n)
        v = make_violation(n, [], "free_cells_outside_solved_period", float(k))
        v["kind"] = "missing_prior_period_holdfix"
        v["gams_var"] = gv
        v["role"] = _ROLE.get(gv, "")
        with_diagnosis(
            v,
            detection=make_detection(
                what=f"{n} has {k} free cell(s) outside {args.period} while "
                     f"{args.period} is solved; GAMS holds {gv}(tsim-1) fixed",
                evidence=f"measured at the solve; var.fx({gv}...tsim-1) in {gms.name}",
                confidence="firm",
            ),
            prescription=make_prescription(
                how=f"freeze {n} in every period but the solved one "
                    f"(freeze_inactive)",
                validated_by=None,
            ),
        )
        violations.append(v)

    meta["python_freezes_prior_stage"] = not violations
    print(f"=== HOLDFIXED / SEQUENCE diff (GAMS {gms.name}) ===", file=sys.stderr)
    print(f"GAMS holds {len(hf)} prior-period vars; {len(mine)} solve(s) of "
          f"{args.period} measured; free outside it: {worst or 'none'}; "
          f"absent in Python: {measured['absent']}", file=sys.stderr)
    if violations:
        headline = (f"holdfix/sequence diff: {len(violations)} of GAMS's {len(hf)} "
                    f"holdfix vars left free outside {args.period} while it is solved")
    else:
        absent = measured["absent"]
        headline = (f"holdfix/sequence diff: Python holds every other period fixed "
                    f"while {args.period} is solved ({len(hf) - len(absent)} of GAMS's "
                    f"{len(hf)} holdfix vars are Python variables; not variables: "
                    f"{absent or 'none'}); this layer does not explain the gap")
    return dict(status="dirty" if violations else "clean", period=args.period,
                headline=headline, violations=violations, meta=meta)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="gtap7_3x3")
    ap.add_argument("--gms", type=Path, default=None,
                    help="GAMS model .gms (default: the data/ reference's model.gms)")
    ap.add_argument("--period", default="check", choices=["base", "check", "shock"],
                    help="the period whose solves are measured")
    ap.add_argument("--gdx", type=Path, default=None,
                    help="reference GDX the case is seeded from (default: the data/ reference)")
    ap.add_argument("--mode", default="altertax", choices=["altertax", "gtap"],
                    help="altertax CD (default) or pure-gtap real-CES")
    ap.add_argument("--ifsub", type=int, default=0, choices=[0, 1],
                    help="ifSUB mode")
    args = ap.parse_args()

    return run_tool("diff_holdfixed", args.dataset, lambda: _work(args),
                    period_hint=args.period)


if __name__ == "__main__":
    raise SystemExit(main())
