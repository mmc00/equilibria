"""Measure the ALTERTAX array route against a RunGTAP AlterTax reference (10 % tms shock).

    python scripts/family/altertax_rungtap_measure.py <dataset> [--trade-balance ratio|nominal]
                                                       [--decomp-steps auto|K|1] [--out DIR]

``<dataset>`` is a folder of ``data/gtap7`` (``nus333``, ``gtap7_10x7``); the reference is its
``gempack_altertax-rungtap_tms-rate10_s8-16-32`` folder (``parameters.prm``, ``tms_rate10.shk``,
``sl4dump.har`` and the four GEMPACK ``.har`` outputs). Solves with the folder's parameters
and shock, then runs every gate of ``family.altertax.gates`` that can be evaluated against
that folder, one by one (a failing or erroring gate does not stop the others), and writes
``<dataset>_<trade_balance>.json`` into DIR (default: the current directory).

The welfare decomposition is the path integral (``solve_altertax(decomp_steps="auto")``:
K sub-steps, K doubled until the EV moves less than TOL_CHANGE / 10); ``--decomp-steps 1``
gives the one-step decomposition, a number K a fixed path.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

REF = "gempack_altertax-rungtap_tms-rate10_s8-16-32"


def _set_env() -> None:
    from tests.family._prep import _newton_mumps

    for k, v in {**_newton_mumps(), "EQUILIBRIA_GTAP_TR_LINSOLVE": "direct"}.items():
        os.environ.setdefault(k, v)
    os.environ.setdefault(
        "EQUILIBRIA_FAMILY_CACHE", tempfile.mkdtemp(prefix="altax_rg_cache_")
    )


def measure(dataset: str, trade_balance: str, decomp_steps: int | str = "auto") -> dict:
    from equilibria.babel.har import read_har
    from equilibria.family.altertax import gates
    from equilibria.family.altertax.closure import (
        ALTAX_RESIDUAL_REGION,
        residual_region_from_exp,
    )
    from equilibria.family.altertax.decomp import decomp_headers
    from equilibria.family.altertax.pct import gempack_changes, pct_changes
    from equilibria.family.altertax.run import load_altertax_params, solve_altertax
    from equilibria.family.altertax.shocks import load_tms_shock
    from equilibria.family.altertax.summary import summary_headers
    from equilibria.family.altertax.updated import updated_headers
    from equilibria.family.altertax.volume import volume_headers

    d = ROOT / "data" / "gtap7" / dataset
    ref = d / REF
    prm = ref / "parameters.prm"
    shock = load_tms_shock(ref / "tms_rate10.shk")
    params = load_altertax_params(d, prm)
    steps = None if decomp_steps == 1 else decomp_steps
    cmf = ref / "run.cmf"  # RunGTAP's closure names the residual region (cgdslack)
    rres = residual_region_from_exp(cmf) if cmf.exists() else ALTAX_RESIDUAL_REGION
    run = solve_altertax(
        d,
        prm=prm,
        shock=shock,
        trade_balance=trade_balance,
        params=params,
        decomp_steps=steps,
        residual_region=rres,
    )
    res = run.result.results["shock"]
    out: dict = {
        "dataset": dataset,
        "residual_region": rres,
        "trade_balance": trade_balance,
        "codes": run.codes,
        "minutes": run.minutes,
        "shock_nit": res.get("nit"),
        "ratio_iterations": res.get("ratio_iterations"),
        "shock_residual": res.get("residual"),
        "decomp_path": run.decomp_path.to_dict() if run.decomp_path else None,
        "outputs": [],
    }
    cmp = gates.compare_all(run, ref)  # sl4dump folder: no waivers, labels off
    out["outputs"] = cmp["outputs"]
    out["waivers"] = cmp["waivers"]
    # every Newton solve and path step; one failed solve fails the run
    out["solves"] = cmp["solves"]
    out["solves_pass"] = cmp["solves_pass"]
    out["pass"] = cmp["pass"]
    sl4 = gates.read_sl4(ref / gates.SL4_DUMP)
    # closure diagnostics straight from the reference's own % changes
    out["gempack_closure"] = {
        k: sl4.changes[k].values.tolist()
        for k in ("del_tbalry", "del_tbal", "ywld", "cgdslack")
        if k in sl4.changes
    }
    from equilibria.family.driver import _world_income

    mp = run.result.mp
    w0, w1 = _world_income(mp, "check"), _world_income(mp, "shock")
    sv = mp.vars["savf"]
    out["ours_ywld_pct"] = 100.0 * (w1 / w0 - 1.0)
    out["ours_savf_over_income_change"] = {
        k[0]: float(sv.value[i]) / w1
        - float(sv.value[sv.keys.index((k[0], "check"))]) / w0
        for i, k in enumerate(sv.keys)
        if k[-1] == "shock" and k[0] != rres
    }
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("--trade-balance", default="ratio", choices=("ratio", "nominal"))
    ap.add_argument("--decomp-steps", default="auto")
    ap.add_argument("--out", default=".")
    a = ap.parse_args()
    steps = a.decomp_steps if a.decomp_steps == "auto" else int(a.decomp_steps)
    _set_env()
    from equilibria.family.altertax.gates import json_safe

    res = measure(a.dataset, a.trade_balance, steps)
    path = Path(a.out) / f"{a.dataset}_{a.trade_balance}.json"
    path.write_text(json.dumps(json_safe(res), indent=1, allow_nan=False, default=str))
    print(path)
    print(
        "codes",
        res["codes"],
        "minutes",
        round(res["minutes"], 2),
        "ratio_it",
        res["ratio_iterations"],
    )
    if res["decomp_path"]:
        dp = res["decomp_path"]
        print(
            "decomp path: steps",
            dp["steps"],
            "solves",
            dp["solves"],
            "minutes",
            round(dp["minutes"], 2),
            "EV change by steps",
            dp["change_by_steps"],
        )
    for o in res["outputs"]:
        print(
            o["output"] if "output" in o else o.get("error"),
            "max",
            o.get("max"),
            "pass",
            o["pass"],
        )
    print_solves(res["solves"])
    print("overall", "PASS" if res["pass"] else "FAIL")
    return 0 if res["pass"] else 1


def print_solves(solves: list[dict]) -> None:
    """One line per Newton solve / path step; the failed ones are marked FAILED. The
    path steps are summed into one line per K (they are many)."""
    path: dict[str, list[dict]] = {}
    for r in solves:
        if r["solve"].startswith("path K="):
            path.setdefault(r["solve"].split(" step")[0], []).append(r)
            continue
        print(
            f"solve {r['solve']}: code {r['code']} nit {r['nit']} residual "
            f"{r['residual']} null pivots {r['null_pivots']} gmin retries "
            f"{r['gmin_retries']} factor failures {r.get('factor_failures', 0)}"
            + ("" if int(r["code"]) == 1 else "  FAILED")
        )
    for k, rs in path.items():
        bad = [r["solve"] for r in rs if int(r["code"]) != 1]
        print(
            f"solve {k}: {len(rs)} steps, codes "
            f"{sorted({int(r['code']) for r in rs})} null pivots "
            f"{sum(int(r['null_pivots']) for r in rs)}"
            + (f"  FAILED {bad}" if bad else "")
        )


def _exit_now(code: int) -> None:
    """End the command without the interpreter's teardown: every output is written
    and closed by now; freeing the run's arrays (~6 GB on 20x41) took 28-51 s. Only
    here, at the entry point; ``main`` returns normally to any other caller."""
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)


if __name__ == "__main__":
    _exit_now(main())
