"""Solve test: the NLP base model, seeded at the benchmark, stays at residual≈0.

The pep2 BASE benchmark reproduces the SAM, so a faithful model solved from the
benchmark point must return a small max residual (the cyipopt solver early-exits there).
A large residual means an equation differs from the reference — a real fidelity bug."""
from __future__ import annotations
from pathlib import Path
import importlib.util
import pytest

ROOT = Path(__file__).resolve().parents[3]
SAM = ROOT / "src/equilibria/templates/reference/pep2/data/SAM-V2_0.gdx"
VALPAR = ROOT / "src/equilibria/templates/reference/pep2/data/VAL_PAR.xlsx"
_HAS_IPOPT = importlib.util.find_spec("pyomo") is not None


@pytest.fixture(scope="module")
def state():
    from equilibria.templates.pep_calibration_unified import PEPModelCalibrator
    return PEPModelCalibrator(sam_file=SAM, val_par_file=VALPAR).calibrate()


@pytest.mark.skipif(not SAM.exists(), reason="pep2 SAM not present")
def test_base_benchmark_residual_small(state):
    from pyomo.environ import SolverFactory
    if not SolverFactory("ipopt").available():
        pytest.skip("ipopt not available")
    from equilibria.templates.pep_pyomo.pep_pyomo_equations import build_pep_model
    from equilibria.templates.pep_pyomo.pep_pyomo_solver import solve_pep, _max_residual
    m = build_pep_model(state, variant="base", form="nlp")
    # residual AT the seed (before solving) — the benchmark should nearly satisfy the system
    seed_resid = _max_residual(m)
    res = solve_pep(m)
    # after solving it must be feasible; report both for diagnosis
    assert res.code == 1, f"did not converge: {res.message} (resid {res.max_residual:.2e})"
    assert res.max_residual < 1e-3, (
        f"max residual {res.max_residual:.2e} too large (seed was {seed_resid:.2e}) — "
        f"an equation likely differs from the reference")
