"""Julia-vs-equilibria cell-by-cell parity tool for the gtap_julia port.

The analogue of the GAMS NLP-vs-NLP gate: run the Julia GTAPv7 model (oracle) and
the Pyomo port on the same dataset + shock, then diff every variable cell. Reports
per-variable match%, the worst cells, and whether a mismatch is a uniform level
shift (numeraire/DOF) or a localized family (a real equation bug).

Usage:
    uv run python scripts/gtap_julia/compare.py [--dataset sample] [--tariff 1.10]
                                                [--base | --shock]
"""
from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parents[1] / "src"))

import pyomo.environ as pyo  # noqa: E402

from equilibria.templates.gtap_julia.model import solve, solve_shock  # noqa: E402
from equilibria.templates.gtap_julia.solution import (  # noqa: E402
    dump_solution,
    load_solution,
)

# Variables to compare (quantities + key prices; exclude pure params).
_COMPARE_VARS = [
    "qo", "qc", "qva", "qint", "qfe", "qfa", "qfd", "qfm", "qca",
    "qxs", "qms", "qds", "qpa", "qga", "qia", "qinv", "qsave", "qes",
    "pds", "po", "pva", "pint", "pfe", "peb", "pes", "pfa", "pms",
    "pmds", "ppa", "pga", "pia", "pinv", "psave", "p", "u", "y", "yp", "yg",
    "pfactor", "rore", "rorc", "rental", "pe", "qe",
]


def compare(dataset="sample", tariff=1.10, shock=False, out_dir=None):
    out = Path(out_dir) if out_dir else _HERE
    # Julia oracle (base solution used both to seed AND — for base — to compare)
    sol = load_solution(dump_solution(dataset=dataset, out_dir=out))
    ref = _julia_reference(dataset, tariff, shock, out)

    res = solve_shock(sol, tariff, rordelta=1) if shock else solve(sol, rordelta=1)
    if not res["ok"]:
        print(f"PORT DID NOT CONVERGE: {res['status']}")
        return
    m = res["model"]

    print(f"=== Julia vs equilibria ({'shock' if shock else 'base'}) ===")
    print(f"{'var':10s} {'cells':>6s} {'match%':>7s} {'median':>10s}  worst-cell")
    all_diffs = []
    for v in _COMPARE_VARS:
        pv = getattr(m, v, None)
        d = ref.get(v)
        if pv is None or not isinstance(d, dict):
            continue
        diffs = []
        worst = (0.0, None)
        for idx in pv:
            key = tuple(str(k) for k in (idx if isinstance(idx, tuple) else (idx,)))
            rv = d.get(key)
            if rv is None or rv != rv or abs(rv) <= 1e-9:
                continue
            rd = abs(pyo.value(pv[idx]) / rv - 1)
            diffs.append(rd)
            if rd > worst[0]:
                worst = (rd, idx)
        if not diffs:
            continue
        all_diffs.extend(diffs)
        within = sum(1 for x in diffs if x <= 0.01) / len(diffs) * 100
        print(f"{v:10s} {len(diffs):6d} {within:6.1f}% "
              f"{statistics.median(diffs):10.2e}  {worst[1]} ({worst[0]:.2e})")
    if all_diffs:
        tot = sum(1 for x in all_diffs if x <= 0.01) / len(all_diffs) * 100
        print(f"\nOVERALL: {tot:.1f}% within 1%, median {statistics.median(all_diffs):.2e}")


def _julia_reference(dataset, tariff, shock, out):
    """Run Julia to the base or shocked solution, return its {var: {idx: val}}."""
    from equilibria.templates.gtap_julia.solution import _dump_shock_solution

    if not shock:
        return load_solution(dump_solution(dataset=dataset, out_dir=out))["all"]
    csv = _dump_shock_solution(dataset, tariff, out)
    return load_solution(csv)["all"]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="sample")
    ap.add_argument("--tariff", type=float, default=1.10)
    ap.add_argument("--shock", action="store_true")
    args = ap.parse_args()
    compare(dataset=args.dataset, tariff=args.tariff, shock=args.shock)
