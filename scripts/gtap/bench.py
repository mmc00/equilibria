"""FAST ITERATION BENCH — edit a convention → run → see the per-variable delta vs the
previous run (which variables moved TOWARD GAMS, which moved AWAY).

Loop:
  1. edit a convention in Python (a pairing, a free-row, an equation form, ...)
  2. uv run python scripts/gtap/bench.py --dataset gtap7_3x3
  3. read: solve codes | match% | variables TOWARD GAMS | AWAY (vs last run)

It measures the case the MCP gate measures, with the gate's own function
(``test_gtap7_mcp_parity._solve_and_measure``: the fbep=header GAMS MCP reference of
the mode, the same build, seed, solve and tolerance), so its match is the gate's
match — this is not a re-implementation. ``--check-only`` instead reports, per
period, the active rows and free columns the solver was handed (the case
``seed_and_solve`` builds, on the core driver): a gap there is a plumbing imbalance.

The snapshot of per-variable divergent cells is stored at
<tmp>/bench_<dataset>_<mode>_ifsub<n>.json; each run diffs against it then overwrites
it, so consecutive runs show the movement your edit caused.

Usage:
  uv run python scripts/gtap/bench.py --dataset gtap7_3x3                  # solve + delta
  uv run python scripts/gtap/bench.py --dataset gtap7_3x3 --mode gtap      # the gtap mode
  uv run python scripts/gtap/bench.py --dataset gtap7_3x3 --check-only     # MCP squareness
  uv run python scripts/gtap/bench.py --dataset gtap7_3x3 --reset          # forget the snapshot
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "gtap"))

PERIODS = ("check", "shock")


def _snap_path(dataset: str, mode: str, ifsub: int) -> Path:
    return Path(tempfile.gettempdir()) / f"bench_{dataset}_{mode}_ifsub{ifsub}.json"


def _gate():
    """The MCP gate module (a test file, loaded by path)."""
    path = ROOT / "tests" / "templates" / "gtap" / "test_gtap7_mcp_parity.py"
    spec = importlib.util.spec_from_file_location("test_gtap7_mcp_parity", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---- MCP squareness, as the solver was handed each period -------------------------
def mcp_squareness(dataset: str, mode: str, ifsub: int, gdx: Path) -> dict:
    """{period: {"constraints", "free_vars", "square", "gap"}}: the active rows and
    free columns of the case on the core driver, counted right after the solver's
    own free rows and fixings were applied to the period."""
    from _core_run import solve_on_core
    from pyomo.environ import Constraint, Var
    from seed_and_solve import _case

    from equilibria.templates.gtap.recipe import _by_backend

    counts: dict = {}

    def spy(model):
        status = _by_backend(model)
        staged = model.staged
        if staged is not None and staged.period not in counts:
            m = model.pyomo
            n_con = sum(1 for _ in m.component_data_objects(Constraint, active=True))
            n_var = sum(
                1 for v in m.component_data_objects(Var, active=True) if not v.fixed
            )
            counts[staged.period] = {
                "constraints": n_con,
                "free_vars": n_var,
                "square": n_con == n_var,
                "gap": n_con - n_var,
            }
        return status

    p, gc, rr = _case(dataset, bool(ifsub), mode)
    solve_on_core(p, gc, rr, mode=mode, ref_gdx=gdx, seed_gdx=gdx, solver=spy)
    return counts


# ---- solve + per-variable match (the gate's own measure) --------------------------
def solve_and_match(dataset: str, mode: str, ifsub: int, gdx: Path) -> dict:
    per = _gate()._solve_and_measure(dataset, ifsub, mode, gdx)
    return {
        p: {
            "match": per[p]["match"],
            "code": per[p]["code"],
            "var_div": per[p]["var_div"],
        }
        for p in PERIODS
    }


def _delta(prev: dict, cur: dict, period: str) -> None:
    pv, cv = prev[period]["var_div"], cur[period]["var_div"]
    toward, away = [], []
    for v in sorted(set(pv) | set(cv)):
        dold, dnow = pv.get(v, 0), cv.get(v, 0)
        if dnow < dold:
            toward.append((v, dold, dnow))  # fewer divergent cells = toward GAMS
        elif dnow > dold:
            away.append((v, dold, dnow))
    toward.sort(key=lambda x: x[1] - x[2], reverse=True)
    away.sort(key=lambda x: x[2] - x[1], reverse=True)
    m0, m1 = prev[period]["match"], cur[period]["match"]
    print(
        f"\n[delta {period}] match {m0:.2f}% → {m1:.2f}% ({m1 - m0:+.2f}pp) | "
        f"vars TOWARD GAMS: {len(toward)} | vars AWAY: {len(away)}"
    )
    for v, o, n in toward[:10]:
        print(f"    ↓ {v:<14} divergent cells {o} → {n}")
    for v, o, n in away[:10]:
        print(f"    ↑ {v:<14} divergent cells {o} → {n}")


def main():
    from cascade_config import resolve_ref_gdx

    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--dataset", default="gtap7_3x3")
    ap.add_argument("--mode", default="altertax", choices=["altertax", "gtap"])
    ap.add_argument("--ifsub", type=int, default=0, choices=[0, 1])
    ap.add_argument(
        "--gdx",
        type=Path,
        default=None,
        help="reference GDX (default: the mode's fbep=header GAMS MCP reference)",
    )
    ap.add_argument(
        "--check-only", action="store_true", help="MCP squareness only, no measure"
    )
    ap.add_argument("--reset", action="store_true", help="forget the snapshot and exit")
    args = ap.parse_args()
    snap = _snap_path(args.dataset, args.mode, args.ifsub)

    if args.reset:
        snap.unlink(missing_ok=True)
        print(f"snapshot {snap} reset")
        return 0

    ref = resolve_ref_gdx(
        args.dataset, mode=args.mode, ifsub=args.ifsub, explicit=args.gdx
    )
    if ref.path is None:
        print(f"no reference: {ref.note}")
        return 2
    print(f"[ref] {ref.note}")

    if args.check_only:
        counts = mcp_squareness(args.dataset, args.mode, args.ifsub, ref.path)
        ok = True
        for per, c in counts.items():
            sq = "SQUARE" if c["square"] else f"UNBALANCED (gap={c['gap']:+d})"
            print(
                f"[MCP {per}] constraints={c['constraints']} "
                f"free_vars={c['free_vars']} → {sq}"
            )
            ok = ok and c["square"]
        return 0 if ok and counts else 1

    cur = solve_and_match(args.dataset, args.mode, args.ifsub, ref.path)
    for per in PERIODS:
        print(f"[solve {per}] code={cur[per]['code']} | match {cur[per]['match']:.2f}%")

    prev = None
    if snap.exists():
        try:
            prev = json.loads(snap.read_text())
        except Exception:
            prev = None
    if prev and all(p in prev for p in PERIODS):
        for per in PERIODS:
            _delta(prev, cur, per)
    else:
        print("\n[delta] no previous run — this is the baseline snapshot.")

    snap.write_text(json.dumps(cur))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
