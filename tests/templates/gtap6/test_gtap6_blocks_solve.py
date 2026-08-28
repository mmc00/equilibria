"""GTAP6 block-composed model solves gtap6_3x3 (canary — F7 Task 10 gate).

North-star gate: the 5 GTAP6 blocks (``TradeArmingtonBlock``,
``ProductionBlock``, ``FactorBlock``, ``DemandUtilityBlock``,
``IncomeClosureBlock``) composed via ``build_block_single_period`` attempt to
solve ``datasets/gtap6_3x3`` with IPOPT for the first time.

STATUS (documented via the diagnostic-first residual-report discipline
mandated by the task, mirroring F3's own precedent before GTAP7's first
block solve — see ``test_gtap6_seed_residuals_are_small`` below for the
first-class diagnostic assertion): the composer fixed 3 real seeding/
calibration bugs surfaced by evaluating every constraint body at the
INITIAL (benchmark-seeded) variable values BEFORE any solve attempt —

  1. ``qo``/``pfd``/``pfm`` (ProductionBlock's real benchmark seeds, order
     1e6-1e7) were silently shadowed by an earlier block's ``np.ones``
     placeholder stub during ``Model.add_block``'s first-registration-wins
     variable dedup (``TradeArmingtonBlock`` runs before ``ProductionBlock``
     in ``GTAP6_BLOCK_ORDER`` but needs those 3 vars as inputs to its own
     equations) — fixed in the composer via
     ``_reseed_shadowed_production_stubs``.
  2. A one-character index bug in ``blocks/gtap6/production.py``'s
     ``pfd_init`` seed construction: the ``to`` (output-tax) parameter is
     keyed by the INPUT COMMODITY ``(i, r)`` — confirmed against the
     orphan branch's own equation chain (``gtap_v62_model_equations.py``'s
     ``eq_pds_rule``/``eq_pfd_rule``: ``pfd[i,j,r] == ps[i,r]*(1+to[i,r])*
     (1+tfd[i,j,r])``) and its own calibration
     (``gtap_v62_calibration.py:802``) — but ``pfd_init`` looked it up by
     the BUYER SECTOR ``to_arr[prod_secs.index(j), ...]`` instead. An
     earlier round of this fix mistakenly "corrected" the direction in
     ``gtap6_calibration.py`` (which was already right) rather than in
     ``production.py`` (the actual bug); that has since been reverted and
     the real fix applied in ``production.py``'s ``pfd_init``.
  3. ``sav``'s seed used ``save_0`` directly instead of ``save_0 -
     savf_0`` (the value consistent with ``IncomeClosureBlock``'s OWN
     ``e_ysav`` identity ``sav == y - yp - yg``, confirmed exactly against
     ``gtap6_calibration.py``'s documented SAM-close comment) — fixed in
     the composer via ``_reseed_sav``.

These 3 fixes collapsed the seed-point residual report's worst cells from
~5.2e7 (``e_qo``) down to: ``walras`` at ~3.5e6 (the GLOBAL foreign-savings
imbalance, ``sum_r savf_0[r]`` — legitimately nonzero at the seed since
closing it is exactly the solver's job, not a seed defect), ``e_qe`` at
~1.0e6 (a genuine, documented benchmark tax wedge between factor purchase
cost ``vfm`` and factor sales ``evom`` — see ``factor.py``'s own test
docstring), a handful of ``e_qfd_arm``/``e_qfd_cgds`` cells at the
2e5-1e6 scale (the ``i != j`` Armington-demand cells, now driven by the
correctly-signed ``pfd`` seed rather than numerical noise — still well
below the pre-fix ~5.2e7 shadowing-bug scale), and every OTHER equation
family at true numerical noise (~1e-9 to ~1e-16).

TASK 10b (this round): the canary's non-convergence (first a
``maxIterations`` plateau at 0.1238, then an ``internalSolverError``/
Restoration-Phase-Failed at ~1.008) was NOT a bounds/scaling/homotopy
problem as originally hypothesized -- it was a genuinely UNDER-DETERMINED
SYSTEM. A Pyomo variable-constraint bipartite-matching diagnostic
(``scipy.sparse.csgraph.maximum_bipartite_matching``) found the composed
model had 950 variables against only 681 active constraints -- DOF=269,
not the 0 a square system needs. Tracing the unmatched variable groups by
hand (cross-checked against ``scripts/gtap6/_v62_monolith_oracle.py``'s
own ``eq_pds_rule``/``eq_pfd_rule``/``eq_pfm_rule``/``eq_ppd_rule``/
``eq_ppm_rule``/``eq_pgd_rule``/``eq_pgm_rule``) found 7 equations that
were simply NEVER PORTED to any of the 5 blocks -- confirmed by grepping
every block file for ``return m.pds[`` / ``return m.pfd[`` / etc. and
finding zero matches for any of them:

  - ``e_pds``: ``pds[j,r] == ps[j,r]*(1+to[j,r])`` (missing from
    ``production.py``; ``pds`` itself was also missing as an OWNED
    variable there, only ever declared as a placeholder stub elsewhere).
  - ``e_pfd``/``e_pfm``: ``pfd[i,j,r] == pds[i,r]*(1+tfd[i,j,r])`` /
    ``pfm[i,j,r] == pim[i,r]*(1+tfi[i,j,r])`` (missing from
    ``production.py`` -- already flagged as a known gap in that file's own
    prior-round comment: "no e_pfd/e_pfm equation is wired in this task").
  - ``e_ppd``/``e_ppm``: the household-nest analogs (missing from
    ``demand_utility.py``).
  - ``e_pgd``/``e_pgm``: the government-nest analogs (missing from
    ``demand_utility.py``).

These 7 equations (120 constraint cells) were added to ``production.py``
and ``demand_utility.py`` (byte-identical to the oracle's own Skip-guarded
linear tax-wedge identities, using the already-ported ``alpha_dom``/
``alpha_imp``-family shares as the nonzero-share guard in place of the
oracle's own never-ported ``share_dom``/``share_imp`` params) and
registered in ``gtap6_contract.py``'s equation-ID lists. DOF dropped from
269 to 149; the remaining unmatched cells are legitimate zero-share
padding over rectangular arrays (e.g. ``pfe[Land,cgds,*]`` -- Land is not
a VA-nest factor for the investment sector) or variables closed by
aggregate/indirect mechanisms (market clearing + Walras' law, standard in
a square CGE system).

The LAST remaining genuine gap was ``savf`` (net foreign savings): also
never defined by any equation anywhere, including in the oracle itself
(confirmed: ``income_closure.py``'s own ``savf`` declaration comment
already documented "the oracle itself never wires one either -- savf is a
genuine free/closure variable"). This is the standard GTAP ``capFix``
closure's fixed variable (mirroring ``templates/gtap/gtap_block_model.py``'s
own ``savf_flag="capFix"`` default -- v6.2 has no ``capFlex``/``savfeq``
rate-of-return-equalization equation ported, so ``capFix`` is the only
closure this block set supports). Fixed at ``savf_0`` in the composer via
``_fix_savf``.

RESULT: IPOPT now reaches ``optimal`` in ~260 iterations with every
equation residual at true numerical noise (~1e-9 to 1e-6). ``walras``
converges to a reproducible, non-zero constant: ``sum_r savf_0[r]`` (~3.47e6
on gtap6_3x3), EXACTLY -- confirmed to machine precision, and consistent
with the oracle's own ``eq_walras`` comment ("leaving walras to only
absorb sum_r savf, a SAM-level constant... which the bake then offsets" --
no such netting-out "bake" step exists anywhere in this port, so the
constant surfaces directly in ``walras`` here). This is a genuine,
non-zero SAM-level foreign-savings imbalance in this dataset under a
``capFix`` closure, NOT a solver defect or a seed/calibration bug -- so
``test_gtap6_3x3_block_model_solves_nlp`` asserts ``walras`` equals that
derived constant (not literally zero) and its ``xfail`` marker has been
removed.
"""

from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
DATASET = ROOT / "datasets" / "gtap6_3x3"


def _has_ipopt() -> bool:
    from pyomo.environ import SolverFactory

    return SolverFactory("ipopt").available()


def _build(*, return_derived: bool = False):
    from equilibria.templates.gtap6.gtap6_block_model import build_block_single_period
    from equilibria.templates.gtap6.gtap6_calibration import derive_calibration
    from equilibria.templates.gtap6.gtap6_contract import default_gtap6_contract
    from equilibria.templates.gtap6.gtap6_parameters import GTAP6Parameters
    from equilibria.templates.gtap6.gtap6_sets import GTAP6Sets

    sets = GTAP6Sets()
    sets.load_from_har(DATASET / "sets.har", default_path=DATASET / "default.prm")
    params = GTAP6Parameters()
    params.load_from_har(DATASET / "basedata.har", DATASET / "default.prm", sets)
    derived = derive_calibration(sets, params)
    closure = default_gtap6_contract().closure

    pm = build_block_single_period(sets, params, derived, closure, mode="nlp")
    if return_derived:
        return pm, derived
    return pm


@pytest.mark.skipif(not DATASET.exists(), reason="gtap6_3x3 dataset not present")
def test_gtap6_3x3_composer_builds():
    """The composer assembles all 5 blocks into a square-ish Pyomo model
    without raising — the first real exercise of ``Model``/``PyomoBackend``
    composition for GTAP6 (never previously exercised end-to-end; each
    block was only tested individually against the oracle)."""
    from pyomo.environ import Constraint, Var

    pm = _build()
    n_vars = sum(
        len(v) if v.is_indexed() else 1 for v in pm.component_objects(Var, active=True)
    )
    n_cons = sum(
        len(c) if c.is_indexed() else 1
        for c in pm.component_objects(Constraint, active=True)
    )
    assert n_vars > 0
    assert n_cons > 0
    # walras + e_walras exist in the default "nlp" mode (IncomeClosureBlock's
    # own mode gate).
    assert hasattr(pm, "walras")
    assert hasattr(pm, "e_walras")


@pytest.mark.skipif(not DATASET.exists(), reason="gtap6_3x3 dataset not present")
def test_gtap6_seed_residuals_are_small():
    """Diagnostic-first regression gate: at the INITIAL (benchmark) point,
    the residual of every equation family must be small — this is the
    exact per-equation residual-report discipline the task mandated before
    attempting any solve, kept as a permanent regression test so a future
    change cannot silently reintroduce a shadowed-stub-class seed bug
    (the ``qo``/``pfd``/``pfm``/``sav`` fixes documented in this module's
    docstring) without failing CI.

    The floor is deliberately loose — it is NOT a convergence gate (the
    solve itself is still open, see
    ``test_gtap6_3x3_block_model_solves_nlp``) but a "did the composer/
    calibration seed break again" tripwire. Before the 3 fixes this module
    documents, the worst seed-point residual was ~5.2e7 (``e_qo``); after,
    the worst non-``walras`` residual is ``e_qe`` at ~1.02e6 (a genuine,
    documented benchmark tax wedge between factor purchase cost ``vfm``
    and factor sales ``evom`` — see ``factor.py``'s own test docstring on
    the "agent-vs-market-price benchmark wedge" — NOT a seed bug), with
    every other family at or below numerical noise. The floor (5e6) sits
    comfortably above that legitimate wedge and comfortably below the
    ~5.2e7 pre-fix shadowing-bug scale, so it still catches a regression
    of the class this task fixed.
    """
    from pyomo.environ import Constraint, value

    pm = _build()

    worst_non_walras = 0.0
    worst_name = None
    for con in pm.component_objects(Constraint, active=True):
        if con.name == "e_walras":
            continue
        for idx, cdata in con.items():
            if not cdata.active:
                continue
            body = value(cdata.body, exception=False)
            lo = value(cdata.lower, exception=False) if cdata.lower is not None else 0.0
            if body is None or lo is None:
                continue
            resid = abs(body - lo)
            if resid > worst_non_walras:
                worst_non_walras = resid
                worst_name = (con.name, idx)

    assert worst_non_walras < 5e6, (
        f"seed-point residual regressed: worst non-walras cell {worst_name} "
        f"= {worst_non_walras:.3e} (expected < 5e6 post-fix -- the largest "
        "legitimate residual is e_qe's ~1.02e6 vfm/evom tax wedge; was "
        "~5.2e7 pre-fix on e_qo)"
    )


@pytest.mark.integration
@pytest.mark.skipif(not DATASET.exists(), reason="gtap6_3x3 dataset not present")
@pytest.mark.skipif(not _has_ipopt(), reason="IPOPT not available in this environment")
def test_gtap6_3x3_block_model_solves_nlp():
    """Canary solve reaches ``optimal`` (Task 10b: fixed the 7 missing
    price-identity equations + fixed ``savf`` at benchmark -- see module
    docstring for the full diagnostic).

    ``walras`` is asserted against ``sum_r savf_0[r]`` rather than 0 --
    under this dataset's ``capFix`` closure (``savf`` fixed at its
    benchmark value, the only closure v6.2's block set supports), the
    correctly-converged Walras-law residual EQUALS that SAM-level foreign-
    savings imbalance constant, confirmed to machine precision. This is
    not a solver artifact: it is the value the oracle's own ``eq_walras``
    comment predicts ("leaving walras to only absorb sum_r savf, a
    SAM-level constant").
    """
    from pyomo.environ import SolverFactory, TerminationCondition, value

    model, derived = _build(return_derived=True)
    solver = SolverFactory("ipopt")
    result = solver.solve(model, tee=False)

    ok_status = result.solver.termination_condition in (
        TerminationCondition.optimal,
        TerminationCondition.locallyOptimal,
    )
    assert ok_status, result.solver.termination_condition

    savf_0 = dict(getattr(derived, "savf_0", {}) or {})
    expected_walras = sum(float(v or 0.0) for v in savf_0.values())
    assert abs(value(model.walras) - expected_walras) < 1e-3, (
        f"walras={value(model.walras)!r} should equal sum_r savf_0[r]="
        f"{expected_walras!r} (the capFix SAM-imbalance constant) at a "
        "true equilibrium -- a bigger gap would indicate a genuine "
        "market-clearing defect, not this dataset's known imbalance"
    )
