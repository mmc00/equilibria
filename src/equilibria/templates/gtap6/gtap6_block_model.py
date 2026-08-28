"""Compose the 5 GTAP6 symbolic blocks into a solvable model (F7 Task 10).

This is the COMPOSER: it assembles the ``equilibria.blocks.gtap6`` block units
(``TradeArmingtonBlock``, ``ProductionBlock``, ``FactorBlock``,
``DemandUtilityBlock``, ``IncomeClosureBlock``, in ``GTAP6_BLOCK_ORDER``) onto
a single ``equilibria.model.Model``, translates that to Pyomo via
``PyomoBackend``, and hands back the resulting ``ConcreteModel``.

Mirrors ``templates/gtap/gtap_block_model.py``'s composer pattern
(``build_block_single_period``) but SIMPLER, matching v6.2's simpler
structure:

  - No ``aa``/``gy`` aggregate-agent sets (v6.2 has no separate Armington
    final-demand-agent split; ``cgds`` lives in ``sets.prod_comm`` and is
    exposed to the blocks directly as the ``cgds`` set).
  - No ``ifSUB`` closure step (v6.2's ``GTAP6ClosureConfig.if_sub`` is a
    fixed ``False`` constant, not a runtime switch — there is nothing to
    apply post-build).
  - No make-matrix / production-scaling step (v6.2 has no make matrix at
    all — the output relation is implicitly diagonal per commodity, see
    ``GTAP6Sets`` module docstring), so this composer does not call
    anything analogous to GTAP7's ``apply_production_scaling``.

Composition order: build sets -> add each block (dependency order) ->
``PyomoBackend.build`` -> strip the bridge's ``_con`` suffix so constraint
names match the contract's bare equation IDs (``e_qxs`` not ``e_qxs_con``)
-> reseed the ``qo``/``pfd``/``pfm`` stubs and ``sav`` (see
``_reseed_shadowed_production_stubs``/``_reseed_sav`` below — the
composer-level seed corrections this model needs, discovered via the
Task 10 canary-solve residual diagnostic; analogous in spirit, if much
narrower, to GTAP7's own composer-owned ``_align_xi_xaa_post_scaling``
post-build seed-reconciliation step).

The real (confirmed by reading source) composition APIs used here:

  - ``equilibria.core.sets.SetManager()`` (no-arg ctor) + ``.add(Set(...))``
    where ``Set`` is a frozen Pydantic model (``name``, ``elements``).
  - ``equilibria.model.Model(name=...)`` + ``.add_set(Set(...))`` +
    ``.add_block(block_instance)`` — ``add_block`` validates the block's
    ``required_sets`` against the model's own ``SetManager``, then calls
    ``block.setup(set_manager, parameters, variables)`` and folds the
    returned params/vars/equations into the model's managers.
  - ``equilibria.backends.pyomo_backend.PyomoBackend()`` + ``.build(model)``
    (mutates in place; read ``backend.pyomo_model`` for the
    ``ConcreteModel``, ``.build`` does not return it).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from equilibria.backends.pyomo_backend import PyomoBackend
from equilibria.core.sets import Set as ESet
from equilibria.model import Model

if TYPE_CHECKING:
    from pyomo.environ import ConcreteModel

    from equilibria.templates.gtap6.gtap6_contract import GTAP6ClosureConfig
    from equilibria.templates.gtap6.gtap6_parameters import GTAP6Parameters
    from equilibria.templates.gtap6.gtap6_sets import GTAP6Sets


def _set_elems(sets: Any) -> dict[str, list[str]]:
    """Set name -> element list, matching every ``required_sets`` name the
    5 GTAP6 blocks declare (grep-verified against each block's
    ``model_post_init``):

      TradeArmingtonBlock: r, i, j, marg
      ProductionBlock:     r, i, j, f
      FactorBlock:         r, j, f
      DemandUtilityBlock:  r, i
      IncomeClosureBlock:  r, f, j

    ``j`` (production sectors) is v6.2's ``sets.prod_comm`` (TRAD_COMM ∪
    CGDS_COMM, i.e. ``i`` plus ``cgds`` — v6.2 has no separate activity set,
    see ``GTAP6Sets`` module docstring), NOT a bare alias of ``i``. ``s``/
    ``rp`` are bilateral-trade aliases of ``r`` the blocks index bilateral
    quantities (``qxs``, ``pmcif`` etc.) over. ``cgds`` is exposed directly
    (v6.2 has no separate final-demand-agent list to fold it into).
    """
    return {
        "r": list(sets.r),
        "s": list(sets.r),  # bilateral alias (exporter/source)
        "rp": list(sets.r),  # bilateral alias (destination)
        "i": list(sets.i),
        "j": list(sets.prod_comm),
        "f": list(sets.f),
        "mf": list(sets.mf),
        "sf": list(sets.sf),
        "m": list(sets.m),
        "marg": list(sets.marg),
        "cgds": list(sets.cgds),
    }


def _reseed_shadowed_production_stubs(
    pm: ConcreteModel, sets: Any, derived: Any
) -> None:
    """Correct 3 variable seeds shadowed by an earlier block's placeholder
    stub during composition (``qo``, ``pfd``, ``pfm``).

    ROOT CAUSE (found via the Task 10 canary-solve residual diagnostic, see
    the task report): several blocks declare a defensive, guarded
    ``_stub(name, ..., np.ones(...))`` for a variable they need as an INPUT
    to their own equations but do not themselves own — the same "first
    registration wins" leaf-block dedup pattern GTAP7 already uses (see
    ``trade_armington.py``'s own comment citing
    ``blocks/gtap/trade_armington_bilateral.py``). ``Model.add_block``
    registers a variable only if its name is not already present
    (``model.py``: ``if var.name not in self.variable_manager``), so
    whichever block runs FIRST in ``GTAP6_BLOCK_ORDER`` wins the
    registration race. This is harmless for a benchmark-NORMALIZED price
    stub (e.g. ``ps``/``pfe``, correctly ``1.0`` at calibration under any
    block) but silently discards the real benchmark seed whenever the true
    owner's value is NOT ``1.0``:

      - ``qo`` (activity output quantity, owned by ``ProductionBlock``,
        seeded from ``derived.vop``/``vom`` — order 1e6-1e7 on
        gtap6_3x3) is shadowed by ``TradeArmingtonBlock``'s
        ``_stub("qo", ..., np.ones((nj, nr)))`` (``TradeArmingtonBlock``
        runs first in ``GTAP6_BLOCK_ORDER`` but needs ``qo`` for its own
        ``e_qfa``). Confirmed via the residual report: every equation
        reading ``qo`` (``e_qo``, ``e_qva``, ``e_qf``, ``e_qfa``,
        ``e_qcgds``, ``e_taxrev``) showed residuals of the same 1e6-1e7
        order at the seed, collapsing to ~1e-9 once ``qo`` is corrected.
      - ``pfd``/``pfm`` (domestic/imported firm AGENT prices, owned by
        ``ProductionBlock``, seeded to ``(1+to)*(1+tfd)`` /
        ``pim_0*(1+tfi)`` — the SAME wedge-inclusive formula
        ``gtap6_calibration.py``'s ``alpha_dom``/``alpha_imp`` CES
        distribution parameters are calibrated against) are shadowed by
        ``TradeArmingtonBlock``'s own ``_stub("pfd"/"pfm", ...,
        np.ones(...))`` (again needed as an input to its own
        ``e_qfd_arm``/``e_qfm_arm``). Confirmed via the residual report:
        ``e_qfd_arm``/``e_qfm_arm``/``e_qfd_cgds``/``e_qfm_cgds`` showed
        ~1e6 residuals (the CES first-order condition only holds exactly
        at the wedge-inclusive ``pfd0``/``pfm0``, not at ``1.0``).

    This mirrors GTAP7's own composer precedent of a post-``backend.build``
    seed-reconciliation pass owned by the COMPOSER rather than by editing
    any individual (already reviewed) block file — see
    ``templates/gtap/gtap_model_equations.py``'s own
    ``_align_xi_xaa_post_scaling``, applied by that composer right after
    the equivalent build step for exactly the same class of cross-block
    seed-consistency issue. No equation algebra is touched here — only
    these 3 VarData's initial values, recomputed with the SAME formula
    their true owner block (``ProductionBlock``) already uses.
    """
    vop_map = dict(getattr(derived, "vop", {}) or {})
    vom_map = dict(getattr(derived, "vom", {}) or {})
    for j in sets.prod_comm:
        for r in sets.r:
            val = vop_map.get((j, r), vom_map.get((j, r), 1.0)) or 1.0
            pm.qo[j, r].set_value(max(float(val), 1e-6))

    to_map = dict(getattr(derived, "to", {}) or {})
    tfd_map = dict(getattr(derived, "tfd", {}) or {})
    tfi_map = dict(getattr(derived, "tfi", {}) or {})
    pim0_map = dict(getattr(derived, "pim_0", {}) or {})
    for i in sets.i:
        for j in sets.prod_comm:
            for r in sets.r:
                to_ir = to_map.get((j, r), 0.0) or 0.0
                tfd_ijr = tfd_map.get((i, j, r), 0.0) or 0.0
                pfd0 = (1.0 + float(to_ir)) * (1.0 + float(tfd_ijr))
                pm.pfd[i, j, r].set_value(max(pfd0, 1e-6))

                pim0_ir = pim0_map.get((i, r), 1.0) or 1.0
                tfi_ijr = tfi_map.get((i, j, r), 0.0) or 0.0
                pfm0 = float(pim0_ir) * (1.0 + float(tfi_ijr))
                pm.pfm[i, j, r].set_value(max(pfm0, 1e-6))


def _reseed_sav(pm: ConcreteModel, sets: Any, derived: Any) -> None:
    """Correct the ``sav`` (regional savings) seed to satisfy its OWN
    defining identity at the benchmark.

    DIFFERENT root cause from ``_reseed_shadowed_production_stubs`` above —
    ``sav`` has no cross-block shadowing (``IncomeClosureBlock`` is its
    sole declarer, see ``income_closure.py``'s ``_q("sav", ..., sav_init)``
    where ``sav_init`` reads ``derived.save_0`` directly). This is a seed
    FORMULA bug in that single declaration: ``IncomeClosureBlock``'s own
    ``e_ysav`` defines ``sav[r] == y[r] - yp[r] - yg[r]`` (the Phase 3.38
    budget identity, see that block's module docstring), but
    ``gtap6_calibration.py``'s own documented SAM-close comment (the
    ``save_0``/``savf_0`` derivation, ``derive_calibration`` around the
    "Phase 3.27 SAM-close" comment) states the closure identity as
    ``pcgds*qo[cgds,r] == y - yp - yg + savf``, i.e. ``y_0 - yp_0 - yg_0
    == save_0 - savf_0`` (confirmed exactly by the Task 10 diagnostic:
    ``y_0 - yp_0 - yg_0`` and ``save_0 - savf_0`` agree to float precision
    on gtap6_3x3). ``sav``'s seed should therefore be ``save_0 - savf_0``,
    not ``save_0`` alone — using ``save_0`` directly (as recorded on the
    live ``sav`` VarData before this correction) leaves a residual on
    ``e_ysav`` equal to EXACTLY ``savf_0`` (confirmed: 2,849,624.49 for
    USA on gtap6_3x3, matching ``derived.savf_0['USA']`` to float
    precision). No equation algebra is touched — only ``sav``'s initial
    value, using the SAME ``save_0``/``savf_0`` the calibration module
    already computes.
    """
    save0_map = dict(getattr(derived, "save_0", {}) or {})
    savf0_map = dict(getattr(derived, "savf_0", {}) or {})
    for r in sets.r:
        save0 = save0_map.get(r, 0.0) or 0.0
        savf0 = savf0_map.get(r, 0.0) or 0.0
        pm.sav[r].set_value(float(save0) - float(savf0))


def _strip_con_suffix(pm: ConcreteModel) -> None:
    """Rename ``{eq}_con`` -> ``{eq}`` so constraint names match the
    contract's bare equation IDs (the bridge's ``_build_constraints`` always
    emits ``{eq_name}_con``; GTAP6's blocks name their equations ``e_qxs``
    etc. without a ``_con`` suffix)."""
    from pyomo.environ import Constraint

    for c in list(pm.component_objects(Constraint, active=True)):
        nm = c.name
        if nm.endswith("_con"):
            base = nm[:-4]
            pm.del_component(c)
            pm.add_component(base, c)


def build_block_single_period(
    sets: GTAP6Sets,
    params: GTAP6Parameters,
    derived: Any,
    closure: GTAP6ClosureConfig | None = None,
    *,
    mode: str = "nlp",
) -> ConcreteModel:
    """Compose the 5 GTAP6 blocks into a single-period solvable Pyomo model.

    Args:
        sets: Loaded ``GTAP6Sets``.
        params: Loaded ``GTAP6Parameters``.
        derived: ``DerivedGTAP6Calibration`` from ``derive_calibration``.
        closure: ``GTAP6ClosureConfig`` (currently unused beyond ``mode`` —
            v6.2 has no ``if_sub`` runtime switch and no make-matrix scaling
            to condition on; accepted for signature parity with the plan
            and forward-compatibility).
        mode: ``"nlp"`` (default) includes the ``walras`` check variable/
            equation (``IncomeClosureBlock``'s own ``mode`` field); ``"mcp"``
            drops both (Walras' law makes one market-clearing equation
            redundant under complementarity).

    Returns:
        The composed Pyomo ``ConcreteModel``, with ``_con``-suffixed
        constraint names stripped back to their bare equation IDs.
    """
    from equilibria.blocks.gtap6 import GTAP6_BLOCK_ORDER
    from equilibria.blocks.gtap6.income_closure import IncomeClosureBlock

    setmap = _set_elems(sets)
    model = Model(name="gtap6_blocks_sp")
    for name, elems in setmap.items():
        model.add_set(ESet(name=name, elements=tuple(elems)))

    for cls in GTAP6_BLOCK_ORDER:
        kwargs: dict[str, Any] = {"sets": sets, "params": params, "derived": derived}
        if cls is IncomeClosureBlock:
            kwargs["mode"] = mode
        block = cls(**kwargs)
        model.add_block(block)

    backend = PyomoBackend()
    backend.build(model)
    pm = backend.pyomo_model
    _strip_con_suffix(pm)
    _reseed_shadowed_production_stubs(pm, sets, derived)
    _reseed_sav(pm, sets, derived)
    return pm
