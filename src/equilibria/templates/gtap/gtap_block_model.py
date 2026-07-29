"""Compose the 7 GTAP symbolic blocks into a solvable model (F3 Task 5).

This is the COMPOSER: it assembles the migrated ``equilibria.blocks.gtap`` block
units onto a single ``equilibria.model.Model``, translates that to Pyomo via the
repaired ``PyomoBackend``, applies the monolith's benchmark scaling (so the
warm-start point matches GAMS ``cal.gms``), and hands the result to the EXISTING
multi-period + PATH/IPOPT solve stack (``GTAPMultiPeriodModel`` /
``solve_multiperiod``) unchanged.

Design (mirrors the monolith's ``build_model`` ordering, gtap_model_equations.py:434):

  compose 7 blocks -> PyomoBackend.build -> strip ``_con`` suffix
  -> apply_production_scaling -> _align_xi_xaa_post_scaling

The single-period block model produced by :func:`build_block_single_period` is
form-identical to ``GTAPModelEquations.build_model()`` (Task 4: form+domain 0-diff,
same 94 constraint families / 1116 cells / 1330 var cells). The scaling functions
are the monolith's OWN methods, run against the block-composed pyomo model via a
lightweight ``GTAPModelEquations`` shim — they mutate the model's VarData by name
(``hasattr``-guarded), so they operate correctly on the block model.

:class:`GTAPBlockMultiPeriodModel` subclasses ``GTAPMultiPeriodModel`` and swaps the
single-period model source from ``GTAPModelEquations().build_model()`` to the
block-composed SP model, so ``build_vars`` / ``build_equations_intra`` /
``build_equations_fisher`` / ``seed_all_periods`` and the whole ``solve_multiperiod``
path are reused verbatim.

FIDELITY: the monolith ``gtap_model_equations.py`` stays the parity ORACLE and is
never edited. The blocks own the equations; the composer owns the post-registration
scaling + the seed.
"""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING, Any

from equilibria.backends.pyomo_backend import PyomoBackend
from equilibria.core.sets import Set as ESet
from equilibria.model import Model
from equilibria.templates.gtap.gtap_model_equations import GTAPModelEquations
from equilibria.templates.gtap.gtap_model_multiperiod import GTAPMultiPeriodModel

if TYPE_CHECKING:
    from pyomo.environ import ConcreteModel

# The aggregate-agent set (activities + the four final-demand agents) and the
# tax-stream set gy — mirror the monolith's set construction.
_AGENTS = ["hhd", "gov", "inv", "tmg"]
_GY = ["pt", "fc", "pc", "gc", "ic", "dt", "mt", "et", "ft", "fs"]


# Dependency order — imported from the blocks package so it stays authoritative.
def _block_classes() -> list[type]:
    import equilibria.blocks.gtap as gtap_blocks

    order = getattr(gtap_blocks, "GTAP_BLOCK_ORDER", None)
    if order is not None:
        return list(order)
    # Fallback to the documented order if the constant is a list of names.
    names = [
        "TradeCETBlock",
        "ProductionSupplyBlock",
        "FactorBlock",
        "ArmingtonBilateralBlock",
        "DemandUtilityBlock",
        "IncomeBlock",
        "ClosureBlock",
    ]
    return [getattr(gtap_blocks, n) for n in names]


def _set_elems(sets: Any) -> dict[str, list[str]]:
    """Set name -> element list, matching the monolith's _add_sets."""
    return {
        "r": list(sets.r),
        "i": list(sets.i),
        "a": list(sets.a),
        "f": list(sets.f),
        "mf": list(sets.mf),
        "sf": list(sets.sf),
        "m": list(sets.m),
        "marg": list(sets.marg),
        "aa": list(sets.a) + _AGENTS,
        "rp": list(sets.r),
        "gy": _GY,
    }


def _mk_unit(
    cls: type, sets: Any, params: Any, residual_region: str, if_sub: bool = False
) -> Any:
    """Instantiate a block, threading residual_region + if_sub for the units that
    declare those fields (others ignore the kwargs)."""
    fields = getattr(cls, "model_fields", {})
    kwargs: dict[str, Any] = {"sets": sets, "params": params}
    if "residual_region" in fields:
        kwargs["residual_region"] = residual_region
    if "if_sub" in fields:
        kwargs["if_sub"] = if_sub
    return cls(**kwargs)


def _strip_con_suffix(pm: ConcreteModel) -> None:
    """Rename ``{eq}_con`` -> ``{eq}`` so the MCP pairing's forced ``_gams_pairs``
    (which name constraints ``eq_pfeq`` etc.) and the reflection machinery match the
    monolith's constraint names. The bridge emits ``{eq_name}_con``; the monolith
    names them ``{eq_name}``.
    """
    from pyomo.environ import Constraint

    for c in list(pm.component_objects(Constraint, active=True)):
        nm = c.name
        if nm.endswith("_con"):
            base = nm[:-4]
            pm.del_component(c)
            pm.add_component(base, c)


# The 9 ifSUB report equations the monolith deactivates under ifSUB (post-block
# gtap_model_equations.py:7970-7983). Their vars (pfa/pfy/pp_rai/pm/pmcif/pefob/
# pwmg/xwmg/xmgm) become report-only — the tariff/margin wedge is substituted INLINE
# via the M_* macros in the real equations (eq_xweq/eq_pmteq etc., which the blocks
# carry under if_sub=True). Deactivating them here — BEFORE this single-period model
# is used as the reflection source — makes ``build_equations_intra`` (which reflects
# only active constraints) skip them, so the multi-period block model is square
# exactly like the monolith MP (which never reflects them either).
_IFSUB_REPORT_EQS = (
    "eq_pp_rai",
    "eq_xwmg",
    "eq_xmgm",
    "eq_pwmg",
    "eq_pefobeq",
    "eq_pmcifeq",
    "eq_pmeq",
    "eq_pfaeq",
    "eq_pfyeq",
)


# The report VARS the 9 deactivated report eqs used to determine. Under ifSUB the
# monolith substitutes their M_* macro INLINE in the consuming eqs (which the blocks
# do too) AND FIXES these vars (post-block 7985-8035) — otherwise they are orphan
# free columns (no defining eq, absent from every consuming eq) that add spurious DOF
# to the .nl. (Confirmed via .nl column diff: without this the block had 117 extra
# pfa/pfy free columns vs the monolith → non-square shock.)
_IFSUB_REPORT_VARS = (
    "pp_rai",
    "xwmg",
    "xmgm",
    "pwmg",
    "pefob",
    "pmcif",
    "pm",
    "pfa",
    "pfy",
)


def _apply_ifsub_closure(pm: ConcreteModel) -> int:
    """The ifSUB closure: deactivate the 9 report equations AND fix their report
    vars (to their current/benchmark value) — mirrors the monolith's post-block.
    Deactivating alone left pfa/pfy as orphan free columns (the macros are inline in
    the consuming eqs, so nothing determines them) → non-square. Fixing them removes
    that spurious DOF. Makes the reflected multi-period block model square like the
    monolith MP."""
    from pyomo.environ import value

    n = 0
    for eq_name in _IFSUB_REPORT_EQS:
        comp = pm.component(eq_name)
        if comp is None:
            continue
        # Deactivate the COMPONENT (not only its data cells): the reflection filter
        # ``component_objects(Constraint, active=True)`` tests the component flag, so
        # a component with all-deactivated cells but still component-active would
        # otherwise be reflected. Deactivating the component makes it skipped.
        n += sum(1 for idx in comp if comp[idx].active)
        comp.deactivate()
    for var_name in _IFSUB_REPORT_VARS:
        v = pm.component(var_name)
        if v is None:
            continue
        for k in v:
            vd = v[k]
            if not vd.fixed:
                with contextlib.suppress(Exception):
                    vd.fix(float(value(vd)))
    return n


def build_block_single_period(
    params: Any,
    sets: Any,
    closure: Any = None,
    residual_region: str | None = None,
    apply_scaling: bool = True,
) -> ConcreteModel:
    """Compose the 7 blocks into a single-period Pyomo model.

    Steps: build an ``equilibria.model.Model`` (sets before blocks), add each block
    in dependency order, translate via ``PyomoBackend``, strip the ``_con`` suffix,
    and (default) apply the monolith's benchmark scaling so the warm-start matches
    GAMS. Returns the Pyomo ``ConcreteModel``.
    """
    if_sub = bool(getattr(closure, "if_sub", False))
    setmap = _set_elems(sets)
    model = Model(name="gtap_blocks_sp")
    for name, elems in setmap.items():
        model.add_set(ESet(name=name, elements=elems))
    for cls in _block_classes():
        model.add_block(
            _mk_unit(cls, sets, params, residual_region or "ROW", if_sub=if_sub)
        )

    backend = PyomoBackend()
    backend.build(model)
    pm = backend.pyomo_model
    _strip_con_suffix(pm)

    if apply_scaling:
        # The monolith's own scaling methods, run against the block model. A shim
        # GTAPModelEquations carries params/sets + the methods; they mutate the
        # passed model's VarData by name (hasattr-guarded), so they work on the
        # block-composed model exactly as on the monolith's build_model().
        shim = GTAPModelEquations(
            sets, params, closure=closure, residual_region=residual_region
        )
        shim.apply_production_scaling(pm)
        shim._align_xi_xaa_post_scaling(pm)

    if if_sub:
        _apply_ifsub_closure(pm)

    pm._residual_region = residual_region
    return pm


class GTAPBlockMultiPeriodModel(GTAPMultiPeriodModel):
    """Multi-period GTAP model sourced from the composed symbolic blocks.

    Identical to ``GTAPMultiPeriodModel`` except the single-period model reflected
    into the multi-period structure is the block-composed one
    (:func:`build_block_single_period`) rather than
    ``GTAPModelEquations().build_model()``. All downstream machinery (var reflection,
    constraint substitution, Fisher rows, seeding, and the ``solve_multiperiod``
    driver) is inherited unchanged — the block SP model is form-identical to the
    monolith SP (Task 4), so the reflection produces an identical multi-period model.
    """

    def _block_sp(self) -> ConcreteModel:
        return build_block_single_period(
            self.params, self.sets, self.closure, self.residual_region
        )

    def build_vars(self, m: ConcreteModel) -> None:
        """Reflect Var families from the block SP model (×3 periods).

        Copy of the parent's reflection with the SP source swapped to the block
        model. Kept in lockstep with ``GTAPMultiPeriodModel.build_vars``.
        """
        from pyomo.environ import NonNegativeReals, Var

        from equilibria.templates.gtap.gtap_model_multiperiod import PERIODS, _astuple

        sp_model = self._block_sp()
        periods = list(PERIODS)

        for v in sp_model.component_objects(Var, active=True):
            name = v.name
            first_key = next(iter(v)) if v.is_indexed() else None
            first_data = v[first_key] if v.is_indexed() else v[None]
            try:
                domain = first_data.domain
            except Exception:
                domain = NonNegativeReals

            if v.is_indexed():
                new_index = [(*_astuple(k), t) for k in v.index_set() for t in periods]
                sp_vals: dict = {}
                for k in v.index_set():
                    try:
                        sp_vals[_astuple(k)] = float(v[k].value)
                    except (TypeError, ValueError):
                        sp_vals[_astuple(k)] = 1.0

                def _mk_init(vals_dict):
                    def _init(_m, *key):
                        *orig, _t = key
                        return vals_dict.get(tuple(orig), 1.0)

                    return _init

                init_fn = _mk_init(sp_vals)
            else:
                new_index = [(t,) for t in periods]
                try:
                    sp_val = float(first_data.value)
                except (TypeError, ValueError):
                    sp_val = 1.0

                def _mk_scalar_init(val):
                    def _init(_m, t):
                        return val

                    return _init

                init_fn = _mk_scalar_init(sp_val)

            doc = v.doc if hasattr(v, "doc") and v.doc else ""
            setattr(m, name, Var(new_index, within=domain, initialize=init_fn, doc=doc))

    def build_equations_intra(self, m: ConcreteModel, period: str) -> None:
        """Reflect Constraint families from the block SP model, indexed by period.

        The parent's reflection builds its own fresh ``GTAPModelEquations`` SP
        via ``build_model()`` internally, so to reuse its ~150-line
        substitution/merge logic verbatim we make that ``build_model()`` return the
        block-composed SP for the duration of this call. Done through ``setattr`` so
        the swap is a scoped, restored override of one method — the block SP is
        form-identical to the monolith SP (Task 4), so the reflection is identical.
        """
        block_sp = self._block_sp()
        orig = GTAPModelEquations.build_model
        GTAPModelEquations.build_model = lambda _self: block_sp
        try:
            super().build_equations_intra(m, period)
        finally:
            GTAPModelEquations.build_model = orig


def build_block_model(
    params: Any,
    sets: Any,
    closure: Any,
    residual_region: str,
) -> tuple[ConcreteModel, GTAPBlockMultiPeriodModel]:
    """Build the full multi-period block-composed GTAP model (unseeded).

    Returns ``(pyomo_model, mp)``. Seed with ``mp.seed_all_periods(m, gdx)`` and
    solve with ``solve_block_model`` (or the monolith's ``solve_multiperiod``).
    """
    mp = GTAPBlockMultiPeriodModel(
        sets, params, closure, residual_region=residual_region
    )
    m = mp.build_sets()
    mp.build_vars(m)
    from equilibria.templates.gtap.gtap_model_multiperiod import PERIODS

    for per in PERIODS:
        mp.build_equations_intra(m, per)
    mp.build_equations_fisher(m)
    m._residual_region = residual_region
    return m, mp


def solve_block_model(
    m: ConcreteModel,
    params: Any,
    closure: Any,
    ref_gdx: Any,
    *,
    mode: str = "gtap",
) -> Any:
    """Seed + solve the composed multi-period block model via the monolith driver.

    Thin wrapper over ``solve_multiperiod`` — the block model is a drop-in for the
    monolith's multi-period model, so the existing NLP/MCP solve path applies.
    """
    from equilibria.templates.gtap.gtap_multiperiod_driver import solve_multiperiod

    return solve_multiperiod(
        m,
        params,
        closure,
        ref_gdx=ref_gdx,
        skip_base_solve=True,
        mute_welfare=True,
        seed_from_prior=False,
        holdfix_cd=False,
        mode=mode,
    )
