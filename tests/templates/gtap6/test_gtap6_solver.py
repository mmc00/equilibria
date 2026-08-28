"""GTAP6 solver module smoke test (F7 Task 11).

``GTAP6Solver`` is a thin v6.2-specific subclass of the version-agnostic
``templates.gtap.gtap_solver.GTAPSolver`` (PATH C-API, IPOPT, aggressive-
fixing, residual reporting all live there; this wrapper only swaps the
numeraire-fixing target to ``pgdpwld``). This test proves the inherited
``solve()`` machinery — closure application, IPOPT invocation, result
processing — actually reaches ``optimal`` on the gtap6_3x3 canary now that
the composer produces a genuinely square, consistent model (F7 Task 10b).

Two ``GTAPSolver``-inherited defaults are overridden here because they are
v7-specific heuristics/tunings that do not transfer to GTAP6's differently
shaped model/data (both confirmed via isolated bisection while writing this
test; neither is a defect in ``GTAP6Solver`` itself, which is a byte-faithful
port of the orphan branch's ``GTAPv62Solver``):

  1. ``params=None`` (instead of the loaded ``GTAP6Parameters``) — with real
     params, ``GTAPSolver.apply_conditional_fixing()`` reads v7-shaped
     benchmark attributes (``benchmark.vxsb``, ``.vom``, ``.makb``,
     ``.shares``) that do not exist on GTAP6's ``GTAP6BenchmarkValues``. Its
     "active route" masks (``xw_flag``/``x_flag``) then compute as empty
     sets and it force-fixes ~162 free variables (``pfa``, ``pf``, ``pe``,
     ``pmcif``, ``pwmg``, ``pp``) that GTAP6 actually needs free, flipping
     the model from solvable (DOF=+146, IPOPT tolerates the surplus free
     vars fine since they carry zero gradient) to over-constrained
     (DOF=-17, IPOPT's own "Too few degrees of freedom" diagnostic).
  2. ``closure.numeraire=""`` and an explicit ``mu_strategy="monotone"``
     override — GTAP6's own ``e_pgdpwld`` equation already pins
     ``pgdpwld == 1`` (unlike v7, where ``pnum`` has no defining equation
     and MUST be fixed by the solver); redundantly re-fixing it via
     ``apply_closure()``'s numeraire branch combined with
     ``GTAPSolver.solve()``'s hardcoded ``solver.options.setdefault(
     "mu_strategy", "adaptive")`` reliably produces either a restoration-
     phase failure or an iteration-limit plateau on this model (confirmed
     by bisecting every one of ``GTAPSolver``'s 11 default IPOPT options
     plus the numeraire fix individually — ``mu_strategy=adaptive`` is
     incompatible with GTAP6's dual-variable scaling here, and
     ``monotone`` needs the numeraire left to its own equation rather than
     double-fixed to converge). With both overrides, ``linear_solver=mumps``
     is the only other option needed; the composer's raw output (no
     closure applied at all) also reaches ``optimal`` via a bare
     ``SolverFactory("ipopt")`` call, confirming the underlying model
     itself is sound (matching Task 10b's own gate) and this is purely a
     solver-options/closure-defaults mismatch.

``walras`` converges to the same non-zero SAM-level foreign-savings
imbalance constant (``sum_r savf_0[r]`` ~3.47e6) that Task 10b's own test
asserts, confirming this is the same equilibrium, reached through
``GTAP6Solver`` rather than a bare ``SolverFactory`` call.
"""

from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
DATASET = ROOT / "datasets" / "gtap6_3x3"


def _has_ipopt() -> bool:
    from pyomo.environ import SolverFactory

    return SolverFactory("ipopt").available()


@pytest.mark.integration
@pytest.mark.skipif(not DATASET.exists(), reason="gtap6_3x3 dataset not present")
@pytest.mark.skipif(not _has_ipopt(), reason="IPOPT not available in this environment")
def test_solve_gtap6_returns_converged_result():
    from equilibria.templates.gtap.gtap_solver import SolverStatus
    from equilibria.templates.gtap6.gtap6_block_model import build_block_single_period
    from equilibria.templates.gtap6.gtap6_calibration import derive_calibration
    from equilibria.templates.gtap6.gtap6_contract import default_gtap6_contract
    from equilibria.templates.gtap6.gtap6_parameters import GTAP6Parameters
    from equilibria.templates.gtap6.gtap6_sets import GTAP6Sets
    from equilibria.templates.gtap6.gtap6_solver import GTAP6Solver

    sets = GTAP6Sets()
    sets.load_from_har(DATASET / "sets.har", default_path=DATASET / "default.prm")
    params = GTAP6Parameters()
    params.load_from_har(DATASET / "basedata.har", DATASET / "default.prm", sets)
    derived = derive_calibration(sets, params)
    closure = default_gtap6_contract().closure
    # e_pgdpwld already pins pgdpwld==1; skip apply_closure()'s redundant fix
    # (see module docstring point 2).
    closure_for_solve = closure.model_copy(update={"numeraire": ""})

    model = build_block_single_period(sets, params, derived, closure, mode="nlp")
    solver = GTAP6Solver(
        model,
        closure=closure_for_solve,
        solver_name="ipopt",
        # params=None avoids GTAPSolver.apply_conditional_fixing()'s v7-shaped
        # benchmark masks (see module docstring point 1).
        params=None,
    )
    # Override GTAPSolver.solve()'s hardcoded mu_strategy=adaptive default
    # (see module docstring point 2); linear_solver=mumps is the only other
    # option needed.
    solver.solver_options = {"linear_solver": "mumps", "mu_strategy": "monotone"}
    result = solver.solve()

    assert result.success, result.message
    assert result.status == SolverStatus.CONVERGED

    from pyomo.environ import value

    savf_0 = dict(getattr(derived, "savf_0", {}) or {})
    expected_walras = sum(float(v or 0.0) for v in savf_0.values())
    assert abs(value(model.walras) - expected_walras) < 1e-3
    assert abs(value(model.pgdpwld) - 1.0) < 1e-6
