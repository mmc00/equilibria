"""Cell-level measurement of the altertax gates, for step 0 of the driver recipe.

Step 0 deletes the altertax per-period recalibrations (``_recalibrate_*``) one at
a time. A deletion is accepted only when no cell that is within the gates'
tolerance in the last accepted run ends up outside it, no stage's match % goes
down, no converged solve stops converging, and the same cells are compared
(rule decided 2026-11-12). The gates report one % per stage; this records every
cell so the rule can be checked.

It drives the gates' own solve helpers, unchanged, over their altertax rows:

- ``mcp``: ``tests/templates/gtap/test_gtap7_mcp_parity.py`` (PATH, seeds from prior)
- ``nlp``: ``tests/templates/gtap/test_gtap7_nlp_parity.py`` (IPOPT)
- ``mp``:  ``tests/templates/gtap/test_altertax_multiperiod_parity.py`` (PATH)

and walks each reference GDX the way the gates do (same skips and aliases).

Usage (local: needs PATH, IPOPT, gdxdump and the datasets)::

    PYTHONPATH=src python scripts/gtap/measure_altertax_cells.py measure --out base.json
    PYTHONPATH=src python scripts/gtap/measure_altertax_cells.py measure \
        --only mcp:gtap7_3x3:ifsub0 --out smoke.json
    PYTHONPATH=src python scripts/gtap/measure_altertax_cells.py compare base.json new.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
GATES = {
    "mcp": ROOT / "tests/templates/gtap/test_gtap7_mcp_parity.py",
    "nlp": ROOT / "tests/templates/gtap/test_gtap7_nlp_parity.py",
    "mp": ROOT / "tests/templates/gtap/test_altertax_multiperiod_parity.py",
}
PERIODS = ("base", "check", "shock")
TOL = 1e-2  # the gates' relative tolerance (1%)
ABS = 1e-6  # and their absolute one


def within(py: float, gams: float, tol: float = TOL) -> bool:
    """The gates' rule: ``|py - gams| <= 1e-6`` or relative gap ``<= tol``."""
    d = abs(py - gams)
    rel = d / abs(gams) if abs(gams) > 1e-12 else (0.0 if d < ABS else 9e9)
    return d <= ABS or rel <= tol


def verdict(base: dict, new: dict) -> list[str]:
    """Why ``new`` is not accepted against ``base``; an empty list accepts it."""
    reasons: list[str] = []
    for row, b in base.items():
        n = new.get(row)
        if n is None:
            reasons.append(f"{row}: missing in the new run (skipped or not measured)")
            continue
        if "error" in b:
            reasons.append(f"{row}: failed in base, not compared: {b['error']}")
            continue
        if "error" in n:
            reasons.append(f"{row}: failed in the new run: {n['error']}")
            continue
        for per, code in b["codes"].items():
            if code == 1 and n["codes"].get(per) != 1:
                reasons.append(f"{row}: {per} code 1 -> {n['codes'].get(per)}")
        for per, pct in b["match"].items():
            got = n["match"].get(per, -1.0)
            if got < pct - 1e-9:
                reasons.append(f"{row}: {per} match {pct:.4f}% -> {got:.4f}%")
        bc, nc = b["cells"], n["cells"]
        if set(bc) != set(nc):
            gone, extra = sorted(set(bc) - set(nc)), sorted(set(nc) - set(bc))
            reasons.append(
                f"{row}: compared cells differ: {len(gone)} only in base "
                f"{gone[:5]}, {len(extra)} only in new {extra[:5]}"
            )
        for key in sorted(set(bc) & set(nc)):
            if bc[key][2] and not nc[key][2]:
                reasons.append(
                    f"{row}: {key} crossed out of tolerance: "
                    f"{bc[key][0]:.10g} -> {nc[key][0]:.10g} (GAMS {nc[key][1]:.10g})"
                )
    return reasons


def _load(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(f"_gate_{name}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _cells(gate: Any, m: Any, gdx: Path) -> dict[str, list]:
    """Every cell the gate compares, in every period: ``[py, gams, within]``."""
    from _diff_core import gams_levels, list_populated_vars, split_t
    from pyomo.environ import value as V

    tol = getattr(gate, "TOL", TOL)
    out: dict[str, list] = {}
    for vn in list_populated_vars(gdx):
        if vn.lower() in gate.SKIP or vn.lower() in gate.RF:
            continue
        try:
            g = gams_levels(gdx, vn)
        except Exception:
            continue
        pv = getattr(m, gate.ALIAS.get(vn, vn), None) or getattr(m, vn.lower(), None)
        if pv is None:
            continue
        for fk, gval in g.items():
            try:
                body, t = split_t(fk)
            except Exception:
                continue
            if t not in PERIODS:
                continue
            st = tuple(gate._strip(x) for x in body)
            idx = (t,) if not st else ((st[0], t) if len(st) == 1 else (*st, t))
            val = None
            for cand in [idx, (*body, t) if body else (t,)]:
                try:
                    val = float(V(pv[cand]))
                    break
                except Exception:
                    pass
            if val is None:
                continue
            out[f"{vn}|{','.join(st)}|{t}"] = [val, gval, within(val, gval, tol)]
    return out


def _rows(gates: dict[str, Any]) -> Iterator[tuple[str, str, int, Path | None]]:
    """``(gate, dataset, ifsub, reference gdx)`` for every altertax row."""
    for r in gates["mcp"]._GAMS_MCP_ROWS:
        if r.mode == "altertax" and r.ci_status != "blocked":
            yield (
                "mcp",
                r.dataset,
                r.ifsub,
                gates["mcp"]._fixture_gdx(r.dataset, r.ifsub, "altertax"),
            )
    for ds, ifsub, mode, _floors, status in gates["nlp"]._NLP_CASES:
        if mode == "altertax" and status != "blocked":
            yield "nlp", ds, ifsub, gates["nlp"]._fixture_gdx(ds, "altertax", ifsub)
    for r in gates["mp"].altertax_rows():
        if r.ci_status != "blocked":
            yield (
                "mp",
                r.dataset,
                r.ifsub,
                gates["mp"]._reference_gdx(r.dataset, bool(r.ifsub)),
            )


def _solve(gate_name: str, gate: Any, ds: str, ifsub: int, gdx: Path) -> dict:
    """Run the gate's own helper; returns ``{"codes", "match"}``."""
    if gate_name == "mp":
        codes, pct, _tot = gate._solve_and_match(ds, bool(ifsub))
        return {"codes": dict(codes), "match": {"shock": pct}}
    per = gate._solve_and_measure(ds, ifsub, "altertax", gdx)
    return {
        "codes": {k: v["code"] for k, v in per.items()},
        "match": {k: v["match"] for k, v in per.items()},
    }


def initial_results(out: Path, resume: bool) -> dict[str, dict]:
    """The rows to start from: none, unless ``resume`` and ``out`` exists.

    Starting from an earlier run would keep its rows for any row this run skips,
    and ``verdict`` would take them as measured.
    """
    return json.loads(out.read_text()) if resume and out.exists() else {}


def measure(out: Path, only: str | None = None, resume: bool = False) -> None:
    """Measure every altertax row (or those whose id contains ``only``) into ``out``."""
    for p in (ROOT / "scripts/gtap", ROOT / "src", ROOT):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    from equilibria.sources import has_dataset
    from equilibria.templates.gtap import gtap_multiperiod_driver as drv

    gates = {k: _load(k, v) for k, v in GATES.items()}
    solve: Callable[..., Any] = drv.solve_multiperiod
    captured: list[Any] = []

    def capture(m: Any, *args: Any, **kwargs: Any) -> Any:
        captured.append(m)
        return solve(m, *args, **kwargs)

    drv.solve_multiperiod = capture  # the helpers import it at call time
    results = initial_results(out, resume)
    try:
        for gate_name, ds, ifsub, gdx in _rows(gates):
            row = f"{gate_name}:{ds}:ifsub{ifsub}"
            if (only and only not in row) or gdx is None or not has_dataset(ds):
                continue
            print(f"=== {row}", flush=True)
            env = os.environ.get("EQUILIBRIA_GTAP_SOLVE_NLP")
            captured.clear()
            t0 = time.perf_counter()
            try:
                res = _solve(gate_name, gates[gate_name], ds, ifsub, gdx)
                res["cells"] = _cells(gates[gate_name], captured[-1], gdx)
            except Exception as exc:  # recorded: verdict rejects a new failure
                res = {"error": f"{type(exc).__name__}: {exc}"}
            finally:
                # the NLP helper sets it and does not restore it
                if env is None:
                    os.environ.pop("EQUILIBRIA_GTAP_SOLVE_NLP", None)
                else:
                    os.environ["EQUILIBRIA_GTAP_SOLVE_NLP"] = env
            res["seconds"] = round(time.perf_counter() - t0, 1)
            results[row] = res
            out.write_text(json.dumps(results, indent=1, sort_keys=True))
            print(
                f"    {res.get('codes')} {res.get('match')} "
                f"{len(res.get('cells', {}))} cells {res['seconds']} s",
                flush=True,
            )
    finally:
        drv.solve_multiperiod = solve


def main() -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("measure", help="solve the altertax rows and record every cell")
    m.add_argument("--out", type=Path, required=True)
    m.add_argument("--only", help="only rows whose id contains this")
    m.add_argument(
        "--resume", action="store_true", help="keep the rows already in --out"
    )
    c = sub.add_parser("compare", help="apply the acceptance rule")
    c.add_argument("base", type=Path)
    c.add_argument("new", type=Path)
    a = ap.parse_args()
    if a.cmd == "measure":
        measure(a.out, a.only, a.resume)
        return 0
    reasons = verdict(json.loads(a.base.read_text()), json.loads(a.new.read_text()))
    for r in reasons:
        print(r)
    print("ACCEPTED" if not reasons else f"REJECTED: {len(reasons)} reason(s)")
    return 1 if reasons else 0


if __name__ == "__main__":
    sys.exit(main())
