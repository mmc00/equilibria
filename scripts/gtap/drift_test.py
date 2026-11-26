"""Tool #7 of the parity-debug cascade: the DRIFT TEST (free-DOF / basin detector).

THE BLIND SPOT this covers: tools 3 (.nl coeff diff), 4 (calibration diff), 5 (equation-
form diff) and 6 (MCP-pairing diff) all compare Python AGAINST GAMS assuming the gap is a
DIFFERENCE between them — a different coefficient, calibration input, equation form, or
eq↔var pairing. They are structurally BLIND to a gap that is NOT a difference: a variable
that is a FREE / DEGENERATE degree of freedom IDENTICALLY in both models (e.g. under
forced-CD sigmav=sigmap=1, eq_pvaeq/eq_pndeq degenerate to the tautology exp(log(pva))==pva
in BOTH Python and GAMS). Such a variable has no restoring force, so PATH places it on a
different (but valid) equilibrium branch than GAMS — yet:
  - the .nl diff sees IDENTICAL coefficients (tautology == tautology) → 0 diffs, gate green;
  - the MCP-pairing diff sees IDENTICAL pairing (both pair pvaeq.pva) → "pairing matches";
  - the residual at the GAMS point is ~0 (a tautology is satisfied by ANY value);
so none of them flags it. This is exactly the gtap7_3x3 78.69%→97.68% gap: pva/pnd are
free under CD; GAMS pins them only via its multi-period holdfix(t-1) sequencing, a solver-
side mechanism that lives OUTSIDE the model equations and is invisible to static compares.

WHAT THIS TOOL DOES: seed the case the MCP gate builds at the GAMS reference point (the
data/ reference by default), solve it on the core driver from that seed, and report which
cells of the period DRIFT most from the seeded GAMS value (the welfare report variables the
gate leaves out are skipped). A large drift on a converged
(code=1) solve = a free/degenerate DOF (or a basin the warm-start can't hold). It also
flags equations whose residual at the GAMS point is ~0 BUT whose paired variable is in the
top drifters — the signature of a tautological/vacuous equation (the smoking gun).

USAGE:
    uv run python scripts/gtap/drift_test.py --dataset gtap7_3x3 --period shock --top 25

INTERPRETATION:
  - Top drifters are PRICES/quantities of one nest (pva/pnd/va/nd) → free CD subsystem;
    the fix is to HOLD them (holdfix: fix + deactivate the tautological paired eq), not to
    chase an equation/coefficient bug that doesn't exist.
  - Top drifters spread across capital/tax (chif/savf/ytaxshr) with the price nest stable →
    a closure or seeding hole (check the seeder completeness first — see warmstart_from_gams).
  - Solve is code!=1 → not a drift problem; fix convergence first (tool 0 check-warmstart).
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

# var → its (likely) paired eq, for the tautology / free-DOF (⚑) flag.
_PAIR = {
    "pva": "eq_pvaeq",
    "pnd": "eq_pndeq",
    "px": "eq_po",
    "pf": "eq_pfeq",
    "pfact": "eq_pfact",
    "pwfact": "eq_pwfact",
}
_PRICE_NEST = {"pva", "pnd", "va", "nd", "px"}
# The welfare report variables the MCP gate leaves out (test_gtap7_mcp_parity.SKIP):
# the driver mutes their rows and fills them after the solve, so they are not a DOF.
_REPORT = {"walras", "ev", "cv", "uh", "u", "ug", "us"}


def _default_gdx(dataset: str, mode: str = "altertax", ifsub: int = 0) -> Path | None:
    """The data/ GAMS MCP reference the MCP gate compares against."""
    folder = find_reference(
        dataset, "gams", form="mcp", mode=mode, ifsub=ifsub, fbep="header"
    )
    return None if folder is None else folder / "out.gdx"


def period_levels(m, period: str) -> dict[tuple[str, tuple], float]:
    """``{(var, cell without the period): level}`` over ``period``'s cells."""
    import pyomo.environ as pyo

    out = {}
    for v in m.component_objects(pyo.Var, active=True):
        for idx in v:
            if isinstance(idx, tuple):
                if not idx or idx[-1] != period:
                    continue
                cell = idx[:-1]
            elif idx == period:
                cell = ()
            else:
                continue
            if v[idx].value is not None:
                out[(v.name, cell)] = float(v[idx].value)
    return out


def drift_rows(seeded: dict, solved: dict, floor: float = 1e-3):
    """[(|solved - seeded| / |seeded|, var, cell, seeded, solved)], largest first.
    Cells seeded below ``floor`` are skipped: a near-zero seed (walras) gives a huge
    relative move that is not economically real; so are the welfare report
    variables (``_REPORT``), as in the MCP gate."""
    rows = []
    for (vname, cell), sv in seeded.items():
        if vname in _REPORT or abs(sv) < floor or (vname, cell) not in solved:
            continue
        now = solved[(vname, cell)]
        rows.append((abs(now - sv) / abs(sv), vname, cell, sv, now))
    rows.sort(reverse=True)
    return rows


def run_drift(
    dataset: str, gdx_path: Path, period: str, mode: str = "altertax", ifsub: int = 0
):
    """Seed the case the MCP gate builds at the GAMS point (seed_and_solve's
    build_seeded plus the period's derived aggregates), then solve it on the core
    driver from the same seed (seed_and_solve._solve_gtap) and rank the drift of
    ``period``'s cells. Returns (code, residual, drifts, eq_resid at the seed)."""
    from pyomo.environ import Constraint
    from pyomo.environ import value as V
    from seed_and_solve import _cascade_derived_seed_mp, _solve_gtap, build_seeded

    _p, _gc, m0 = build_seeded(dataset, gdx_path, period, mode, bool(ifsub))

    def _setv(comp, key, val):
        try:
            comp[key].set_value(val)
            return 1
        except Exception:
            return 0

    _cascade_derived_seed_mp(m0, V, _setv, period)
    seeded = period_levels(m0, period)

    # Equation residuals AT the seed, per family over the period's rows (to flag
    # tautologies: resid~0 but the paired var drifts).
    eq_resid: dict[str, float] = {}
    for c in m0.component_objects(Constraint, active=True):
        mx = 0.0
        for idx in c:
            last = idx[-1] if isinstance(idx, tuple) and idx else idx
            if last != period or not c[idx].active:
                continue
            try:
                body = V(c[idx].body)
                lo = V(c[idx].lower) if c[idx].lower is not None else 0.0
                mx = max(mx, abs(body - lo))
            except Exception:
                pass
        eq_resid[c.name] = mx

    m, res = _solve_gtap(dataset, gdx_path, bool(ifsub), mode)
    code = res.get(period, {}).get("code")
    resid = float(res.get(period, {}).get("residual") or 0.0)
    return code, resid, drift_rows(seeded, period_levels(m, period)), eq_resid


def _is_free_dof(
    vname: str, rel: float, eq_resid: dict, tol: float
) -> tuple[bool, str, float]:
    """Return (is_free_dof, paired_eq, paired_eq_resid) for the ⚑ flag.

    A var is a FREE/TAUTOLOGICAL DOF when it drifts >= tol while its paired
    equation's residual at the seed is ~0 (the eq does not restrain its own var).
    """
    eqn = _PAIR.get(vname)
    if not eqn:
        return False, "", 0.0
    er = eq_resid.get(eqn, 1.0)
    return (er < 1e-6 and rel >= tol), eqn, er


# ---------------------------------------------------------------------------
# Human-readable formatter — KEPT for debug only; writes to STDERR so it can
# never contaminate the JSON on stdout. Not called in the normal path.
# ---------------------------------------------------------------------------
def _debug_print(args, code, resid, drifts, eq_resid):
    print(
        f"=== DRIFT TEST: {args.dataset} period={args.period} ref={args.gdx.name} ===",
        file=sys.stderr,
    )
    print(f"Solve: code={code} resid={resid:.3e}", file=sys.stderr)
    if code != 1:
        print(
            "  ⚠️  Solve did NOT converge (code!=1) — CONVERGENCE problem, not drift.",
            file=sys.stderr,
        )
    n_drift = sum(1 for d in drifts if d[0] >= args.tol)
    print(
        f"\nVariables drifting >= {args.tol:g}: {n_drift}/{len(drifts)}",
        file=sys.stderr,
    )
    for rel, vname, idx, sv, nv in drifts[: args.top]:
        free, eqn, _er = _is_free_dof(vname, rel, eq_resid, args.tol)
        flag = (
            f"  ⚑ {eqn} resid≈0 but var drifts → FREE/TAUTOLOGICAL DOF" if free else ""
        )
        print(
            f"{rel * 100:>7.2f}%  {vname:<12} {str(idx):<28} "
            f"{sv:>12.5f} {nv:>12.5f}{flag}",
            file=sys.stderr,
        )


def _work(args) -> dict:
    if args.gdx is None:
        args.gdx = _default_gdx(args.dataset, args.mode, args.ifsub)
    if args.gdx is None or not args.gdx.exists():
        return dict(
            status="error",
            period=args.period,
            headline=f"reference GDX not found: {args.gdx}",
            violations=[],
            meta={"error_kind": "gdx_not_found", "gdx": str(args.gdx)},
        )
    code, resid, drifts, eq_resid = run_drift(
        args.dataset, args.gdx, args.period, args.mode, args.ifsub
    )
    _debug_print(args, code, resid, drifts, eq_resid)

    # By its own docstring: code != 1 is a CONVERGENCE problem, not a drift one;
    # the drift ranking is unreliable, so report it as an error for the cascade.
    if code != 1:
        return dict(
            status="error",
            period=args.period,
            headline=(
                f"solve did NOT converge (code={code}, resid={resid:.3e}) "
                f"— convergence problem, not drift; run tool 0 "
                f"(check-warmstart) first"
            ),
            violations=[],
            meta={
                "error_kind": "no_convergence",
                "termination_code": code,
                "residual": resid,
                "gdx_ref": str(args.gdx),
                "tol": args.tol,
            },
        )

    # Converged: rank drifters; a drifter whose paired eq resid≈0 is a free DOF.
    viols = []
    for rel, vname, idx, sv, nv in drifts:
        if rel < args.tol:
            continue
        free, eqn, er = _is_free_dof(vname, rel, eq_resid, args.tol)
        v = make_violation(vname, idx, "drift_rel", rel)
        v["free_dof"] = bool(free)
        v["paired_eq"] = eqn or None
        v["paired_eq_resid"] = er
        v["seeded"] = sv
        v["solved"] = nv
        viols.append(v)

    # Tally leading drifters and detect the price-nest free-DOF signature.
    top_vars: dict[str, int] = {}
    for rel, vname, idx, sv, nv in drifts[: max(args.top, 10)]:
        if rel >= args.tol:
            top_vars[vname] = top_vars.get(vname, 0) + 1
    lead = sorted(top_vars.items(), key=lambda kv: -kv[1])
    price_nest_leads = any(v in _PRICE_NEST for v, _ in lead[:4])
    n_free = sum(1 for v in viols if v["free_dof"])

    status = "dirty" if viols else "clean"
    if not viols:
        headline = (
            f"seed-at-GAMS {args.period}: solve converged (code=1), "
            f"no variable drifts >= {args.tol:g} — the GAMS branch is "
            f"reachable; this layer does not explain the gap"
        )
    else:
        worst = viols[0]
        nest_note = ""
        if price_nest_leads:
            nest_note = (
                " — a PRICE/QUANTITY NEST leads; likely a FREE DOF, "
                "fix = HOLD (fix var + deactivate tautological paired eq)"
            )
        headline = (
            f"seed-at-GAMS {args.period}: {len(viols)} var(s) drift >= "
            f"{args.tol:g}, {n_free} flagged FREE-DOF (⚑); worst "
            f"{worst['entity']}{worst['index']}={worst['value'] * 100:.2f}%"
            f"{nest_note}"
        )

    return dict(
        status=status,
        period=args.period,
        headline=headline,
        violations=viols,
        meta={
            "termination_code": code,
            "residual": resid,
            "n_drifters": len(viols),
            "n_free_dof": n_free,
            "price_nest_leads": price_nest_leads,
            "leading_vars": [{"var": v, "cells": n} for v, n in lead[:8]],
            "tol": args.tol,
            "gdx_ref": str(args.gdx),
        },
    )


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--dataset", default="gtap7_3x3")
    ap.add_argument(
        "--gdx",
        type=Path,
        default=None,
        help="GAMS reference GDX to seed from (default: the data/ GAMS MCP reference)",
    )
    ap.add_argument("--period", default="shock", choices=["check", "shock"])
    ap.add_argument("--top", type=int, default=25)
    ap.add_argument(
        "--tol",
        type=float,
        default=1e-3,
        help="rel-drift threshold to count as drifting",
    )
    ap.add_argument(
        "--mode",
        default="altertax",
        choices=["altertax", "gtap"],
        help="altertax CD (default) or pure-gtap real-CES",
    )
    ap.add_argument("--ifsub", type=int, default=0, choices=[0, 1], help="ifSUB mode")
    args = ap.parse_args()

    return run_tool(
        "drift_test", args.dataset, lambda: _work(args), period_hint=args.period
    )


if __name__ == "__main__":
    raise SystemExit(main())
