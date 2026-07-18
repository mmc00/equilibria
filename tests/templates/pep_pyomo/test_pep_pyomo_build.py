"""Build test: the PEP Pyomo model constructs for both variants + forms, with a sane
variable/constraint count and a near-square system (residual-system feasibility)."""
from __future__ import annotations
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[3]
SAM = ROOT / "src/equilibria/templates/reference/pep2/data/SAM-V2_0.gdx"
VALPAR = ROOT / "src/equilibria/templates/reference/pep2/data/VAL_PAR.xlsx"


@pytest.fixture(scope="module")
def state():
    from equilibria.templates.pep_calibration_unified import PEPModelCalibrator
    return PEPModelCalibrator(sam_file=SAM, val_par_file=VALPAR).calibrate()


@pytest.mark.skipif(not SAM.exists(), reason="pep2 SAM not present")
@pytest.mark.parametrize("variant", ["base", "objdef"])
@pytest.mark.parametrize("form", ["nlp", "mcp"])
def test_model_builds(state, variant, form):
    from pyomo.environ import Var, Constraint
    from equilibria.templates.pep_pyomo.pep_pyomo_equations import build_pep_model
    m = build_pep_model(state, variant=variant, form=form)
    n_vars = sum(len(v) for v in m.component_objects(Var, active=True))
    n_cons = sum(len(c) for c in m.component_objects(Constraint, active=True))
    assert n_vars > 200, f"too few vars: {n_vars}"
    assert n_cons > 150, f"too few constraints: {n_cons}"
    # near-square: the residual system has one free slack (LEON) so |vars - cons| is small
    # relative to the system size (numeraire e fixed in mcp removes one dof).
    assert abs(n_vars - n_cons) < 0.5 * n_cons, f"far from square: {n_vars} vars vs {n_cons} cons"
