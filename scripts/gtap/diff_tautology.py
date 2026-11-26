"""Tool #9 of the parity-debug cascade: TAUTOLOGY / UNANCHORED-VARIABLE detector.

THE BLIND SPOT this covers (the one that capped gtap7_3x3 at 78.69% for many sessions):
the gap was a MISSING equation, not a DIFFERENT one. Under forced-CD (sigmav=sigmap=1)
eq_pvaeq degenerated to the tautology `exp(log(pva))==pva` (i.e. `1==1`) — identical in
Python AND GAMS — so it placed NO constraint on its paired variable pva, leaving pva free
to slide. Every static-compare tool is structurally blind to this:
  - tool 3 (.nl): identical coefficients (tautology==tautology) → 0 diff, gate green;
  - tool 6 (MCP-pairing): identical pairing (both pair pvaeq.pva) → "matches";
  - residual at the GAMS point: ~0 (a tautology is satisfied by ANY value);
they all "pass" because there is no DIFFERENCE to find — the equation is equally vacuous
in both. Comparison tools detect differences; they cannot detect an ABSENCE.

WHAT THIS TOOL DOES (numerical, not symbolic): it builds the case as the MCP gate
does, seeded at the data/ GAMS reference in every period (seed_and_solve.build_seeded),
and for each equation<->variable MCP pair, over the requested period's cells, PERTURBS
each free variable of the paired row by a small step and measures whether the row's
residual responds. An MCP row need not hold its paired variable (a market clearing such
as xds = sum xda is paired with its price pd), so the test is on the row as a whole: a
row that responds to none of its variables constrains nothing, and its paired variable
is UNANCHORED (a tautology / free DOF). This finds the CAUSE (vacuous equation) that
tools 3/4/6 miss and tool 7 only sees the symptom of.

USAGE:
    uv run python scripts/gtap/diff_tautology.py --dataset gtap7_3x3 --period shock

INTERPRETATION:
  - A flagged pair (eq_X ↔ x, sensitivity≈0) = eq_X is a tautology for x → x is a free DOF.
    The fix is to give x a real determinant (the missing economic identity), NOT to tweak a
    coefficient. Precedent: eq_pvaeq under CD → fixed with the VA-value identity
    pva·va=Σ(pfa·xf).
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "gtap"))

from _parity_json import make_violation, run_tool  # noqa: E402 — shared JSON contract
from equilibria.sources import find_reference


# GAMS model-statement MCP pairs (eq → its complementary var). Subset that matters for
# the CD altertax block; extend as needed. Mirrors `model gtap /eq.var/`.
_PAIRS = [
    ("eq_pvaeq", "pva"), ("eq_pxeq", "px"), ("eq_po", "px"), ("eq_pndeq", "pnd"),
    ("eq_pfeq", "pf"), ("eq_pfteq", "pft"), ("eq_xfteq", "xft"), ("eq_va", "va"),
    ("eq_pdeq", "pd"), ("eq_peq", "p_rai"), ("eq_pmteq", "pmt"), ("eq_pwfact", "pwfact"),
    ("eq_pfact", "pfact"), ("eq_pabs", "pabs"), ("eq_yc", "yc"), ("eq_regy", "regy"),
    # eq_paa↔pa added after the gtap7_15x10 MEX collapse: eq_paa is the top-Armington
    # price (pa^(1-s)=alphad·pdp^(1-s)+alpham·pmp^(1-s)). It is NOT tautological per se,
    # but when its calibrated shares don't sum to 1 it becomes UNSATISFIABLE — the perturb
    # test still helps: a near-zero ∂resid/∂pa flags pa as effectively unanchored there.
    # (The summing-to-1 break itself is caught by diff_calibration's alphad+alpham audit.)
    ("eq_paa", "pa"),
]


def _resid(con) -> float:
    from pyomo.environ import value as V

    body = V(con.body)
    lo = V(con.lower) if con.lower is not None else 0.0
    return body - lo


def _in_period(idx, period: str) -> tuple | None:
    """The cell without the period if ``idx`` (period last) is of ``period``."""
    if isinstance(idx, tuple):
        return idx[:-1] if idx and idx[-1] == period else None
    return () if idx == period else None


def _row_response(con, delta: float) -> float:
    """The largest ``|d resid| / step`` of the row over its free variables (step
    ``delta * max(|x|, 1)``): ~0 only for a row that constrains nothing."""
    from pyomo.core.expr.visitor import identify_variables

    r0 = _resid(con)
    best = 0.0
    for x in identify_variables(con.body, include_fixed=False):
        if x.value is None:
            continue
        x0 = float(x.value)
        h = delta * max(abs(x0), 1.0)
        try:
            x.set_value(x0 + h)
            best = max(best, abs(_resid(con) - r0) / h)
        finally:
            x.set_value(x0)
    return best


def scan_pairs(m, period: str, pairs, delta: float):
    """For each eq<->var pair, the least row response (``_row_response``) over the
    free cells of ``period``: [(sensitivity, eq, var, cell without the period, cells
    sampled)]. An MCP row need not hold its paired variable (a market clearing is
    paired with its price), so the row is read against all its variables: the paired
    variable is unanchored when its row responds to none of them."""
    rows = []
    for eqn, vn in pairs:
        eq = getattr(m, eqn, None)
        v = getattr(m, vn, None)
        if eq is None or v is None:
            continue
        worst_sens = None
        worst_cell = None
        n = 0
        for idx in v:
            cell = _in_period(idx, period)
            if cell is None or v[idx].fixed or v[idx].value is None:
                continue
            try:
                con = eq[idx]
            except KeyError:
                continue  # eq indexed differently (e.g. eq (r,a), var (r,f,a))
            if not con.active:
                continue
            sens = _row_response(con, delta)
            n += 1
            if worst_sens is None or sens < worst_sens:  # the LEAST sensitive
                worst_sens, worst_cell = sens, cell
        if n > 0:
            rows.append((worst_sens, eqn, vn, worst_cell, n))
    return rows


def run_tautology_scan(dataset: str, gdx_path: Path, period: str, delta: float,
                       mode: str = "altertax", ifsub: int = 0):
    """``scan_pairs`` on the case the MCP gate builds, seeded at the GAMS point in
    every period (seed_and_solve.build_seeded)."""
    from seed_and_solve import build_seeded

    _p, _gc, m = build_seeded(dataset, gdx_path, period, mode, bool(ifsub))
    return scan_pairs(m, period, _PAIRS, delta)


def _default_gdx(dataset: str, mode: str = "altertax", ifsub: int = 0) -> Path | None:
    """The data/ GAMS MCP reference the MCP gate compares against."""
    folder = find_reference(dataset, "gams", form="mcp", mode=mode, ifsub=ifsub,
                            fbep="header")
    return None if folder is None else folder / "out.gdx"


def _work(args) -> dict:
    gdx = args.gdx or _default_gdx(args.dataset, args.mode, args.ifsub)
    if gdx is None or not gdx.exists():
        return dict(status="error", period=args.period,
                    headline=f"reference GDX not found: {gdx}",
                    violations=[],
                    meta={"error_kind": "gdx_not_found", "gdx": str(gdx)})

    rows = run_tautology_scan(args.dataset, gdx, args.period, args.delta,
                              args.mode, args.ifsub)
    rows.sort()  # lowest sensitivity first = most tautological

    # human-readable table → stderr (never contaminates the JSON stdout line)
    print(f"=== TAUTOLOGY / unanchored-var scan: {args.dataset} period={args.period} ===",
          file=sys.stderr)
    print(f"{'∂resid/∂var':>12}  {'equation':<14} {'var':<8} {'cell':<22} flag",
          file=sys.stderr)
    for sens, eqn, vn, cell, n in rows:
        flag = "  ⚑ TAUTOLOGY → UNANCHORED" if sens < args.tol else ""
        print(f"{sens:>12.3e}  {eqn:<14} {vn:<8} {str(cell):<22}{flag}", file=sys.stderr)

    flagged = [r for r in rows if r[0] < args.tol]
    violations = []
    for sens, eqn, vn, cell, n in flagged:
        v = make_violation(vn, list(cell) if isinstance(cell, tuple) else [cell],
                           "dresid_dvar", float(sens))
        v["kind"] = "unanchored_dof"
        v["equation"] = eqn
        v["cells_sampled"] = n
        v["note"] = (f"{eqn} does NOT constrain {vn} (∂resid/∂var≈0) → {vn} is a FREE DOF "
                     f"PATH places arbitrarily (root-selection cause)")
        violations.append(v)

    status = "dirty" if flagged else "clean"
    if flagged:
        worst = flagged[0]
        headline = (f"unanchored-DOF scan ({args.period}): {len(flagged)} eq↔var pair(s) "
                    f"with ∂resid/∂var<{args.tol:g} ⚑ — vacuous paired row = free DOF "
                    f"(root-selection cause); worst {worst[1]}↔{worst[2]}")
    else:
        headline = (f"unanchored-DOF scan ({args.period}): every paired equation constrains "
                    f"its variable — no free DOF; this layer does not explain the gap")
    return dict(status=status, period=args.period, headline=headline,
                violations=violations,
                meta={"gdx": str(gdx), "n_pairs_checked": len(rows),
                      "n_unanchored": len(flagged), "delta": args.delta, "tol": args.tol})


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="gtap7_3x3")
    ap.add_argument("--gdx", type=Path, default=None,
                    help="reference GDX (default: the data/ GAMS MCP reference)")
    ap.add_argument("--period", default="shock", choices=["check", "shock"])
    ap.add_argument("--delta", type=float, default=1e-4, help="relative perturbation of the var")
    ap.add_argument("--tol", type=float, default=1e-6,
                    help="sensitivity below this = the eq does NOT constrain its var (tautology)")
    ap.add_argument("--mode", default="altertax", choices=["altertax", "gtap"],
                    help="altertax CD (default) or pure-gtap real-CES")
    ap.add_argument("--ifsub", type=int, default=0, choices=[0, 1],
                    help="ifSUB mode")
    args = ap.parse_args()

    return run_tool("diff_tautology", args.dataset, lambda: _work(args),
                    period_hint=args.period)


if __name__ == "__main__":
    raise SystemExit(main())
