"""FAST ITERATION BENCH — edit a convention → run → see the per-cell delta vs the
previous run (which cells moved TOWARD GAMS, which moved AWAY), with an MCP-squareness
pre-check BEFORE solving so a plumbing imbalance is caught up front, not after a wasted
solve.

Loop:
  1. edit a convention in Python (a pairing, a free-row, an equation form, ...)
  2. .venv/bin/python scripts/gtap/bench.py --dataset gtap7_3x3
  3. read: MCP square? | match% | cells TOWARD GAMS | cells AWAY (vs last run)

It REUSES diff_altertax's solve pipeline (the one that gives the real 73%) so the
numbers are the same ones we've been trusting — this is not a re-implementation.

Snapshot of per-cell |py-gams| is stored at /tmp/bench_<dataset>.json; each run diffs
against it then overwrites it, so consecutive runs show the movement your edit caused.

Usage:
  uv run python scripts/gtap/bench.py --dataset gtap7_3x3            # solve + delta
  uv run python scripts/gtap/bench.py --dataset gtap7_3x3 --check-only   # MCP square pre-check, NO solve
  uv run python scripts/gtap/bench.py --dataset gtap7_3x3 --reset    # forget the snapshot
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "gtap"))

TOL_REL, TOL_ABS = 1e-3, 1e-6


def _snap_path(dataset: str) -> Path:
    return Path(f"/tmp/bench_{dataset}.json")


# ---- MCP squareness pre-check (BEFORE solving) -----------------------------------
def mcp_precheck(dataset: str, period: str = "shock") -> dict:
    """Build the shock model + apply the FULL solver closure (conditional-fixing,
    aggressive, squareness patches) with a 0-iteration solve budget, then count active
    constraints vs free variables — the same square test run_gtap does at line ~2316,
    but reported UP FRONT so an imbalance is seen before a real solve is wasted."""
    import validate_reference as _vr
    from diff_mcp_pairing import _apply_solver_closure
    from pyomo.environ import Constraint, Var
    model, p = _vr._build_model(dataset, period)
    warn = None
    try:
        _apply_solver_closure(model, dataset, p)  # runs the real closure at 0 iters
    except Exception as e:  # noqa: BLE001
        warn = str(e)
    n_con = sum(1 for c in model.component_data_objects(Constraint, active=True))
    n_var = sum(1 for v in model.component_data_objects(Var, active=True) if not v.fixed)
    return {"constraints": n_con, "free_vars": n_var, "square": n_con == n_var,
            "gap": n_con - n_var, "closure_warn": warn}


# ---- solve + per-variable match (INVOKE diff_altertax, do NOT re-implement) -------
# Hard lesson (this session, 3×): a standalone re-build of the solve does NOT reproduce
# diff_altertax's 73% — it skips betaCal/regy-unfix/phip and lands code=2/3 with 0 cells.
# So the bench INVOKES diff_altertax.py as a subprocess (the real 73% pipeline) and parses
# its --csv (per-variable diverge counts). The delta granularity is per-VARIABLE, which is
# enough to see which block moved toward/away from GAMS.
def solve_and_match(dataset: str, gdx: Path) -> dict:
    import subprocess, csv, tempfile
    csv_path = Path(tempfile.gettempdir()) / f"bench_diff_{dataset}.csv"
    cmd = [str(ROOT / ".venv" / "bin" / "python"),
           str(ROOT / "scripts" / "gtap" / "diff_altertax.py"),
           "--dataset", dataset, "--gdx", str(gdx), "--csv", str(csv_path)]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    # match rate from stdout (diff_altertax prints "Match rate: NN.NN%")
    match_rate = 0.0
    chk = shk = None
    for ln in proc.stdout.splitlines():
        s = ln.strip()
        if s.startswith("Match rate:"):
            try:
                match_rate = float(s.split(":")[1].strip().rstrip("%"))
            except Exception:
                pass
        if "check residual" in s and "code=" in s:
            chk = s.split("code=")[1].split()[0]
        if "shock residual" in s and "code=" in s:
            shk = s.split("code=")[1].split()[0]
    # per-variable abs-error proxy = diverge count (the CSV's granular signal)
    var_div = {}
    if csv_path.exists():
        for r in csv.DictReader(csv_path.open()):
            if r.get("var") in (None, "__SUMMARY__"):
                continue
            try:
                var_div[r["var"]] = int(r["diverge"])
            except Exception:
                pass
    return {"match_rate": match_rate, "var_div": var_div,
            "check_code": chk, "shock_code": shk,
            "stderr_tail": proc.stderr[-300:] if match_rate == 0.0 else ""}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="gtap7_3x3")
    ap.add_argument("--gdx", type=Path,
                    default=Path("/Users/marmol/proyectos2/equilibria_refs/"
                                 "gtap7_3x3_altertax_cd/out_altertax_ifsub0.gdx"))
    ap.add_argument("--check-only", action="store_true", help="MCP square pre-check only, no solve")
    ap.add_argument("--reset", action="store_true", help="forget the snapshot and exit")
    ap.add_argument("--move-eps", type=float, default=1e-4,
                    help="min |Δ abs-error| to count a cell as moved")
    args = ap.parse_args()
    snap = _snap_path(args.dataset)

    if args.reset:
        snap.unlink(missing_ok=True)
        print(f"snapshot {snap} reset")
        return 0

    # 1. MCP squareness pre-check — ALWAYS, before any solve
    pc = mcp_precheck(args.dataset)
    sq = "SQUARE" if pc["square"] else f"UNBALANCED (gap={pc['gap']:+d})"
    print(f"[MCP pre-check] constraints={pc['constraints']} free_vars={pc['free_vars']} → {sq}")
    if pc["closure_warn"]:
        print(f"[MCP pre-check] closure warn: {pc['closure_warn']}")
    if not pc["square"]:
        print("  ⚠ MCP is NOT square — a solve would fail (residual=inf). Fix the "
              "pairing/closure before solving. (This is the plumbing check you wanted "
              "up front.)")
        if not args.check_only:
            print("  (skipping solve — unbalanced)")
        return 1
    if args.check_only:
        print("  ✓ square — safe to solve.")
        return 0

    # 2. solve + per-variable match (via diff_altertax subprocess)
    cur = solve_and_match(args.dataset, args.gdx)
    print(f"[solve] check code={cur['check_code']} shock code={cur['shock_code']} | "
          f"match {cur['match_rate']:.2f}%")
    if cur["match_rate"] == 0.0 and cur.get("stderr_tail"):
        print(f"  ⚠ match 0% — diff_altertax may have failed. stderr tail:\n  {cur['stderr_tail']}")

    # 3. delta vs previous run (per-variable diverge-count movement)
    prev = None
    if snap.exists():
        try:
            prev = json.loads(snap.read_text())
        except Exception:
            prev = None
    if prev and "var_div" in prev:
        pv, cv = prev["var_div"], cur["var_div"]
        toward, away = [], []
        for v, dnow in cv.items():
            dold = pv.get(v)
            if dold is None:
                continue
            if dnow < dold:
                toward.append((v, dold, dnow))   # fewer divergent cells = moved toward GAMS
            elif dnow > dold:
                away.append((v, dold, dnow))
        toward.sort(key=lambda x: x[1] - x[2], reverse=True)
        away.sort(key=lambda x: x[2] - x[1], reverse=True)
        print(f"\n[delta vs previous run] match {prev.get('match_rate', 0):.2f}% → "
              f"{cur['match_rate']:.2f}%  ({cur['match_rate']-prev.get('match_rate',0):+.2f}pp)")
        print(f"  vars TOWARD GAMS: {len(toward)} | vars AWAY: {len(away)}")
        for v, o, n in toward[:10]:
            print(f"    ↓ {v:<14} divergent cells {o} → {n}")
        for v, o, n in away[:10]:
            print(f"    ↑ {v:<14} divergent cells {o} → {n}")
    else:
        print("\n[delta] no previous run — this is the baseline snapshot.")

    snap.write_text(json.dumps({"match_rate": cur["match_rate"],
                                "var_div": cur["var_div"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
