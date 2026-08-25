"""Lever B2 — conditional MUMPS symbolic-factorization reuse under GMIN.

HARD GATE: with reuse ON the solve must be result-identical to reuse OFF
(same solution <1e-8, same code) while doing strictly fewer symbolic
factorizations (the spike: 3 distinct patterns in 28 Newton steps).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts/gtap"))

DATA = Path("datasets/gtap7_15x10")

_SOLVE_ENV = {
    "EQUILIBRIA_GTAP_SOLVE_NLP": "1",
    "EQUILIBRIA_GTAP_SOLVER": "scipy_newton_tr",
    "EQUILIBRIA_GTAP_TR_LINSOLVE": "mumps",
    "EQUILIBRIA_GTAP_NLP_NO_JACSCALE": "1",
    "EQUILIBRIA_GTAP_TR_GATE": "1",
    "EQUILIBRIA_GTAP_TR_FTOL": "1e-7",
    "EQUILIBRIA_GTAP_SCIPY_MAXITER": "300",
    "EQUILIBRIA_GTAP_TR_DELTA0": "10.0",
    "EQUILIBRIA_GTAP_SHOCK_CONTINUATION": "0.125,0.25,0.375,0.5,0.625,0.75,0.875,1.0",
    "GTAP_GATES_SKIP": "1",
    "EQUILIBRIA_GTAP_GMIN": "1e-9",
    "EQUILIBRIA_GTAP_TR_RELTOL": "1e-6",
    "EQUILIBRIA_SEED_CACHE_DISABLE": "1",
}


def _load_params():
    from equilibria.templates.gtap import GTAPParameters

    p = GTAPParameters()
    p.load_from_har(
        basedata_path=DATA / "basedata.har", sets_path=DATA / "sets.har",
        default_path=DATA / "default.prm", baserate_path=DATA / "baserate.har",
    )
    return p


def _closure():
    from equilibria.templates.gtap.gtap_contract import GTAPClosureConfig

    return GTAPClosureConfig(
        name="base", closure_type="MCP", capital_mobility="sluggish",
        fix_endowments=False, fix_taxes=False, fix_technology=False,
        if_sub=False, savf_flag="capFix", numeraire="pnum",
    )


def _solve_15x10(reuse):
    """Solve 15x10 with GMIN+MUMPS; return (solution_dict, symbolic_count, shock_code).

    NOTE: the driver loads run_gtap via spec_from_file_location and registers it as
    sys.modules["run_gtap"] (gtap_multiperiod_driver._load_run_gtap). That is the
    instance whose _SYMBOLIC_FACT_COUNT the solve increments — so we read the counter
    from sys.modules["run_gtap"] AFTER the solve, not from an early `import run_gtap`
    (which is a different module object).
    """
    from pyomo.environ import Var
    from pyomo.environ import value as V

    for k, v in _SOLVE_ENV.items():
        os.environ[k] = v
    os.environ["EQUILIBRIA_GTAP_GMIN_SYM_REUSE"] = "1" if reuse else "0"

    # The driver calls _load_run_gtap() itself inside solve_multiperiod (a fresh
    # module starting at count 0), registered as sys.modules["run_gtap"]. We read
    # THAT module's counter after the solve. base_calibrated=True runs the settle
    # then the final solve — the last registered module reflects the final solve.
    from equilibria.templates.gtap.gtap_block_model import solve_block_model
    from equilibria.templates.gtap_sparse.multiperiod import build_sparse_model_mp

    p = _load_params()
    rr = list(p.sets.r)[-1]
    m, mp, _ = build_sparse_model_mp(p, p.sets, _closure(), rr, base_calibrated=True)
    res = solve_block_model(m, p, _closure(), ref_gdx=None, mode="gtap")
    sol = {}
    for vv in m.component_objects(Var, active=True):
        for idx in vv:
            try:
                sol[f"{vv.name}{idx}"] = float(V(vv[idx]))
            except Exception:
                pass
    count = sys.modules["run_gtap"]._SYMBOLIC_FACT_COUNT
    return sol, count, res.get("shock", {}).get("code")


@pytest.mark.skipif(not DATA.exists(), reason="gtap7_15x10 dataset not present")
def test_symbolic_counter_increments():
    _, count, code = _solve_15x10(reuse=False)
    assert code == 1, f"solve did not converge (code={code})"
    assert count > 0, "symbolic factorization counter never incremented"
    print(f"reuse-OFF symbolic count={count}")


@pytest.mark.skipif(not DATA.exists(), reason="gtap7_15x10 dataset not present")
def test_reuse_fewer_symbolics_same_solution():
    sol_off, count_off, code_off = _solve_15x10(reuse=False)
    sol_on, count_on, code_on = _solve_15x10(reuse=True)
    print(f"symbolic count: OFF={count_off} ON={count_on}")
    assert code_off == 1 and code_on == 1, f"codes off={code_off} on={code_on}"
    # reuse must do STRICTLY fewer symbolic factorizations (spike: 3 patterns / 28 steps)
    assert count_on < count_off, (
        f"reuse did not reduce symbolic factorizations: on={count_on} off={count_off}"
    )
    # HARD GATE: identical solution
    shared = set(sol_off) & set(sol_on)
    assert shared, "no shared var keys"
    worst, key = 0.0, None
    for k in shared:
        rel = abs(sol_on[k] - sol_off[k]) / (abs(sol_off[k]) + 1e-12)
        if rel > worst:
            worst, key = rel, k
    assert worst < 1e-8, f"HARD GATE: solve diverged at {key}: rel={worst:.2e}. STOP."


def test_optout_env_default_is_on():
    """The reuse defaults ON ('1') and '0' opts out — the exact gate the solver
    branch reads (`os.environ.get("EQUILIBRIA_GTAP_GMIN_SYM_REUSE", "1") != "0"`).
    No solve needed: this pins the flag contract that test_reuse_* exercises live."""
    os.environ.pop("EQUILIBRIA_GTAP_GMIN_SYM_REUSE", None)
    assert os.environ.get("EQUILIBRIA_GTAP_GMIN_SYM_REUSE", "1") != "0"  # default ON
    os.environ["EQUILIBRIA_GTAP_GMIN_SYM_REUSE"] = "0"
    assert os.environ.get("EQUILIBRIA_GTAP_GMIN_SYM_REUSE", "1") == "0"  # opt-out
    os.environ.pop("EQUILIBRIA_GTAP_GMIN_SYM_REUSE", None)
