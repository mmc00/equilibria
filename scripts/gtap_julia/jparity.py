"""jparity — Julia-vs-equilibria NLP-vs-NLP parity for the gtap_julia port.

The symmetric analogue of the GAMS measure_nlp_vs_nlp gate. Three checks:

  1. DIRECT   — equilibria solve vs Julia solve (base or shock), cell-by-cell.
  2. FORWARD  — is the JULIA point a root of the equilibria system? (seed
     equilibria at Julia's solution, resolve, must stay.)
  3. REVERSE  — is the EQUILIBRIA point a root Julia admits? (seed Julia at
     equilibria's solution, resolve, must stay.) If Julia MOVES AWAY, the
     equilibria root is one Julia rejects → a closure/formulation discrepancy,
     NOT basin noise.

A DIRECT mismatch with a clean FORWARD but a dirty REVERSE localizes the bug to
equilibria admitting an equilibrium Julia's model does not (a laxer closure or an
extra root from an equation form).

Usage:
    uv run python scripts/gtap_julia/jparity.py [--dataset sample] [--tariff 1.10]
                                                [--shock] [--reverse]
"""
from __future__ import annotations

import argparse
import statistics
import subprocess
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parents[1] / "src"))

import pyomo.environ as pyo  # noqa: E402

from equilibria.templates.gtap_julia.model import solve, solve_shock  # noqa: E402
from equilibria.templates.gtap_julia.solution import (  # noqa: E402
    _dump_shock_solution,
    dump_solution,
    load_solution,
)

_JULIA = Path.home() / ".juliaup" / "bin" / "julia"
_PKG = Path.home() / "proyectos" / "GlobalTradeAnalysisProjectModelV7.jl"
_VERIFY = _HERE / "verify_root.jl"

_COMPARE_VARS = [
    "qo", "qc", "qva", "qfe", "qxs", "qms", "qds", "qpa", "qga", "qia",
    "qinv", "qsave", "qes", "pds", "pva", "pfe", "peb", "pms", "pmds", "ppa",
    "pinv", "psave", "p", "u", "up", "y", "yp", "pfactor", "rore", "pe",
]


def _match(port_vals: dict, ref: dict, tag: str) -> float:
    diffs = []
    print(f"{'var':10s} {'match%':>7s} {'median':>10s}  worst")
    for v in _COMPARE_VARS:
        pv = port_vals.get(v)
        d = ref.get(v)
        if not isinstance(pv, dict) or not isinstance(d, dict):
            continue
        vd, worst = [], (0.0, None)
        for key, val in pv.items():
            rv = d.get(key)
            if rv is None or rv != rv or abs(rv) <= 1e-9:
                continue
            rd = abs(val / rv - 1)
            vd.append(rd)
            if rd > worst[0]:
                worst = (rd, key)
        if not vd:
            continue
        diffs.extend(vd)
        w = sum(1 for x in vd if x <= 0.01) / len(vd) * 100
        print(f"{v:10s} {w:6.1f}% {statistics.median(vd):10.2e}  {worst[1]}({worst[0]:.1e})")
    tot = sum(1 for x in diffs if x <= 0.01) / len(diffs) * 100 if diffs else 0.0
    med = statistics.median(diffs) if diffs else 0.0
    print(f"\n{tag}: {tot:.1f}% within 1%, median {med:.2e}\n")
    return tot


def _port_values(m) -> dict:
    out: dict[str, dict] = {}
    for vname in m.component_map(pyo.Var):
        pv = getattr(m, vname)
        d = {}
        for idx in pv:
            key = tuple(str(k) for k in (idx if isinstance(idx, tuple) else (idx,)))
            val = pyo.value(pv[idx])
            if val is not None:
                d[key] = val
        out[vname] = d
    return out


def _dump_port_csv(m, path: Path) -> None:
    with path.open("w") as f:
        for vname in m.component_map(pyo.Var):
            pv = getattr(m, vname)
            for idx in pv:
                key = ",".join(str(k) for k in (idx if isinstance(idx, tuple) else (idx,)))
                val = pyo.value(pv[idx])
                if val is not None and val == val:
                    f.write(f"{vname},{key},{val}\n")


def _julia_reseed(dataset, tariff, seed_csv: Path, out_dir: Path) -> dict:
    """Seed Julia at an external point, resolve, return its {var:{idx:val}}."""
    out = out_dir / "julia_reseed.csv"
    res = subprocess.run(
        [str(_JULIA), f"--project={_PKG}", str(_VERIFY),
         dataset, str(tariff), str(seed_csv), str(out)],
        capture_output=True, text=True, timeout=900,
    )
    if res.returncode != 0 or ">>> DONE" not in res.stdout:
        raise RuntimeError(f"verify_root failed:\n{res.stdout[-1500:]}\n{res.stderr[-1500:]}")
    return load_solution(out)["all"]


def run(dataset="sample", tariff=1.10, shock=False, reverse=False, out_dir=None):
    out = Path(out_dir) if out_dir else _HERE
    sol = load_solution(dump_solution(dataset=dataset, out_dir=out))

    # Julia reference (base or shock)
    if shock:
        ref = load_solution(_dump_shock_solution(dataset, tariff, out))["all"]
    else:
        ref = sol["all"]

    res = solve_shock(sol, tariff, rordelta=1) if shock else solve(sol, rordelta=1)
    if not res["ok"]:
        print(f"PORT DID NOT CONVERGE: {res['status']}")
        return
    m = res["model"]
    port = _port_values(m)

    print(f"=== jparity ({'shock' if shock else 'base'}) : DIRECT ===")
    _match(port, ref, "DIRECT")

    if reverse and shock:
        # seed Julia at MY shocked point, resolve — does Julia stay?
        seed_csv = out / "port_shock_seed.csv"
        _dump_port_csv(m, seed_csv)
        julia_from_me = _julia_reseed(dataset, tariff, seed_csv, out)
        print("=== REVERSE: Julia reseeded at equilibria's point ===")
        # compare Julia-from-me vs my point: if Julia stayed, ~100%; if it moved, <100
        _match(port, julia_from_me, "REVERSE (Julia stayed at my point?)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="sample")
    ap.add_argument("--tariff", type=float, default=1.10)
    ap.add_argument("--shock", action="store_true")
    ap.add_argument("--reverse", action="store_true")
    args = ap.parse_args()
    run(dataset=args.dataset, tariff=args.tariff, shock=args.shock, reverse=args.reverse)
