"""Compose the 7 GTAP symbolic blocks into a solvable model (F3 Task 5).

This is the COMPOSER: it assembles the migrated ``equilibria.blocks.gtap`` block
units onto a single ``equilibria.model.Model``, translates that to Pyomo via the
repaired ``PyomoBackend``, applies the monolith's benchmark scaling (so the
warm-start point matches GAMS ``cal.gms``), and hands the result to the EXISTING
multi-period + PATH/IPOPT solve stack (``GTAPMultiPeriodModel`` /
``solve_multiperiod``) unchanged.

Design (mirrors the monolith's ``build_model`` ordering, gtap_model_equations.py:434):

  compose 7 blocks -> PyomoBackend.build -> strip ``_con`` suffix
  -> apply_production_scaling -> align_xi_xaa_post_scaling

The single-period block model produced by :func:`build_block_single_period` is
form-identical to ``GTAPModelEquations.build_model()`` (Task 4: form+domain 0-diff,
same 94 constraint families / 1116 cells / 1330 var cells). The scaling functions
live in ``gtap_benchmark_scaling`` and are shared with the monolith, which keeps
them as one-line delegators — they mutate the model's VarData by name
(``hasattr``-guarded), so they operate correctly on the block model.

Este modulo ya NO importa el monolito. La reflexion multi-periodo heredada de
``GTAPMultiPeriodModel`` pide su modelo SP a ``_build_sp()``, que esta clase
sobrescribe; antes habia que hacerle monkey-patch a
``GTAPModelEquations.build_model``. Ver ``docs/architecture/monolito_vs_bloques.md``.

:class:`GTAPBlockMultiPeriodModel` subclasses ``GTAPMultiPeriodModel`` and swaps the
single-period model source to the block-composed SP model by overriding
``_build_sp()``, so ``build_vars`` / ``build_equations_intra`` /
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
from equilibria.blocks.gtap import model_cache as _model_cache
from equilibria.core.sets import Set as ESet
from equilibria.model import Model
from equilibria.templates.gtap.gtap_benchmark_scaling import (
    ScalingContext,
    align_xi_xaa_post_scaling,
    apply_production_scaling,
)
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
    cls: type,
    sets: Any,
    params: Any,
    residual_region: str,
    if_sub: bool = False,
    savf_flag: str = "capFix",
) -> Any:
    """Instantiate a block, threading residual_region + if_sub + savf_flag for the
    units that declare those fields (others ignore the kwargs)."""
    fields = getattr(cls, "model_fields", {})
    kwargs: dict[str, Any] = {"sets": sets, "params": params}
    if "residual_region" in fields:
        kwargs["residual_region"] = residual_region
    if "if_sub" in fields:
        kwargs["if_sub"] = if_sub
    if "savf_flag" in fields:
        kwargs["savf_flag"] = savf_flag
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
    savf_flag = str(getattr(closure, "savf_flag", "capFix"))
    setmap = _set_elems(sets)
    model = Model(name="gtap_blocks_sp")
    for name, elems in setmap.items():
        model.add_set(ESet(name=name, elements=elems))
    for cls in _block_classes():
        model.add_block(
            _mk_unit(
                cls,
                sets,
                params,
                residual_region or "ROW",
                if_sub=if_sub,
                savf_flag=savf_flag,
            )
        )

    backend = PyomoBackend()
    backend.build(model)
    pm = backend.pyomo_model
    _strip_con_suffix(pm)

    if apply_scaling:
        # El escalado de benchmark vive en su propio modulo: muta los VarData del
        # modelo por nombre (con guarda hasattr), asi que corre sobre el modelo
        # compuesto por bloques igual que sobre el build_model() del monolito.
        ctx = ScalingContext(
            params=params,
            sets=sets,
            residual_region=residual_region or "NAmerica",
        )
        apply_production_scaling(pm, ctx)
        align_xi_xaa_post_scaling(pm, ctx)

        # Composer carry (blocks/gtap/__init__.py item 3): el bloque CLOSURE seedea
        # pf0/xf0/mqfact*_bb desde el benchmark SIN escalar; el monolito los
        # snapshotea del modelo YA escalado. Va aqui, con los niveles definitivos.
        apply_fisher_snapshot_overwrite(pm)

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
        """El modelo SP compuesto por bloques.

        Punto de extension de las SUBCLASES: ``GTAPLogLevelsMultiPeriodModel`` lo
        sobrescribe para entregar el modelo log-envuelto. Mantener el nombre es lo
        que hace que esa cadena siga funcionando.
        """
        return build_block_single_period(
            self.params, self.sets, self.closure, self.residual_region
        )

    def _build_sp(self) -> ConcreteModel:
        """El modelo SP que la reflexion multiperiodo del padre va a leer.

        Unico punto de divergencia con ``GTAPMultiPeriodModel``: el padre llama a
        este metodo cada vez que necesita un SP, asi que sobrescribirlo basta para
        que TODA su maquinaria --reflexion de Vars, sustitucion de Constraints,
        filas Fisher-- opere sobre bloques. Antes esto se conseguia haciendole
        monkey-patch a ``GTAPModelEquations.build_model``.

        Delega en ``_block_sp`` para que las subclases sigan teniendo un unico
        punto que sobrescribir.
        """
        return self._block_sp()

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


def build_block_model(
    params: Any,
    sets: Any,
    closure: Any,
    residual_region: str,
    base_calibrated: bool = False,
    ref_gdx: Any = None,
) -> tuple[ConcreteModel, GTAPBlockMultiPeriodModel]:
    """Build the full multi-period block-composed GTAP model (unseeded).

    Returns ``(pyomo_model, mp)``. Seed with ``mp.seed_all_periods(m, gdx)`` and
    solve with ``solve_block_model`` (or the monolith's ``solve_multiperiod``).

    ``base_calibrated`` (F3.5, default ``False`` = faithful-to-GAMS): when ``True``,
    run the settle solve once via ``FactorBlock.calibrate_base`` and stamp the
    settled check-period point on ``m._settled_seed`` (and ``m._base_calibrated``),
    so the driver seeds the base from it and solves ``base→shock`` (no check).
    The default path is byte-unchanged; ``calibrate_base`` builds its own settle
    model with ``base_calibrated=False`` (no recursion).
    """
    # Propagate the value-added subsidy basis from the closure onto params, so
    # the pure derivation recipes (_va_wedge in _derived_params) and the driver's
    # per-period recalibration read it without threading closure through their
    # signatures. Default "gams" leaves the faithful-to-GAMS path byte-unchanged;
    # the gtap7_gempack closure sets "gempack" (EVFP subsidy basis).
    params.va_subsidy_basis = getattr(closure, "va_subsidy_basis", "gams")

    # capFlex needs risk[r] = rorg/rore(r) calibrated from a benchmark (capFix) solve
    # BEFORE the multi-period model folds mutable params to numbers (gtap_model_multiperiod
    # substitutes every mutable ParamData by its value). Stash it on params so the
    # DemandUtilityBlock reads it in setup(). Gated on _capflex_risk to avoid recursion
    # (the capFix twin build must not re-enter this).
    if (
        str(getattr(closure, "savf_flag", "capFix")) == "capFlex"
        and getattr(params, "_capflex_risk", None) is None
    ):
        params._capflex_risk = _calibrate_capflex_risk(
            params, sets, closure, residual_region, ref_gdx=ref_gdx
        )

    mp = GTAPBlockMultiPeriodModel(
        sets, params, closure, residual_region=residual_region
    )

    # Built-model disk cache (OPT-IN, EQUILIBRIA_GTAP_MODEL_CACHE=1). Building the
    # 20x41 costs ~4 min; loading the same model back costs ~1.35 min — measured 3.6x
    # with byte-identical parity from both paths (94.5% within-1pp, 0.1523pp, code=1).
    # Only `m` is cached: `mp` is the builder and is cheap to re-instantiate (the line
    # above), while `m` is the 631k-constraint object that costs the minutes.
    # The key covers the input files AND the source of every module that builds the
    # model, so an edited equation or a regenerated .har can never be served a stale
    # model. See blocks/gtap/model_cache.py.
    _mc = None
    _mc_key = None
    if _model_cache.enabled():
        # cache_key returns None when it cannot cover every input that shapes the
        # model; that means SKIP the cache, never fall back to a partial key.
        _mc_key = _model_cache.cache_key(
            params, closure, residual_region, base_calibrated, ref_gdx=ref_gdx
        )
        if _mc_key is not None:
            _mc = _model_cache
            _cached = _mc.load(_mc_key)
            if _cached is not None:
                return _cached, mp

    m = mp.build_sets()
    mp.build_vars(m)
    mp.build_equations_all_periods(m)
    mp.build_equations_fisher(m)
    m._residual_region = residual_region
    m._base_calibrated = base_calibrated
    m._settled_seed = None
    if base_calibrated:
        from equilibria.blocks.gtap.factor import FactorBlock

        _fb = FactorBlock(sets=sets, params=params)
        m._settled_seed = _fb.calibrate_base(
            params, sets, closure, residual_region, ref_gdx=None
        )
    if _mc is not None:
        _mc.save(_mc_key, m)
    return m, mp


def _capflex_risk_from_gdx(ref_gdx, regions) -> dict:
    """Fast path: read benchmark rore/rorg straight from the GAMS ref GDX (base period) and
    return ``{region: rorg/rore(r)}``. GAMS already solved the benchmark, so we skip the
    266s capFix twin-solve. Returns {} if the GDX lacks rore/rorg (falls back to the twin)."""
    if ref_gdx is None:
        return {}
    try:
        from pathlib import Path as _P

        from _diff_core import gams_levels  # scripts/gtap on sys.path (parity env)

        gp = _P(str(ref_gdx))
        if not gp.exists():
            return {}
        rore = gams_levels(gp, "rore")
        rorg = gams_levels(gp, "rorg")
    except Exception:
        return {}

    def _pick(d, *cands):
        for c in cands:
            if c in d:
                return float(d[c])
        return None

    rorg_b = _pick(rorg, ("base",), "base")
    if rorg_b is None:
        return {}
    out: dict = {}
    for r in regions:
        rore_b = _pick(rore, (r, "base"), (str(r), "base"))
        if rore_b and abs(rore_b) > 1e-12:
            out[r] = rorg_b / rore_b
    return out if len(out) == len(list(regions)) else {}


def _calibrate_capflex_risk(
    params, sets, closure, residual_region, ref_gdx=None
) -> dict:
    """Calibrate capFlex risk[r] = rorg/rore(r) at benchmark (GAMS cal.gms:676).

    The equalization savfeq (risk*rore == rorg) must preserve the benchmark return spread:
    at base, returns DIFFER by region (rore(r) != rorg), and risk[r] freezes that ratio so a
    shock reallocates investment relative to the benchmark, not toward a spurious uniform rore.
    Leaving risk=1 blows up qinv.

    FAST PATH: read the benchmark rore/rorg from the GAMS ref GDX (GAMS already solved them) —
    avoids a full capFix twin-solve (266s on 10x7). Only when no usable GDX is available do we
    fall back to building+solving a capFix twin. Returns ``{}`` on any failure (block then
    falls back to risk=1)."""
    import contextlib
    import io

    from pyomo.environ import value as _V

    from equilibria.templates.gtap.gtap_contract import GTAPClosureConfig

    regions = _set_elems(sets).get("r", [])

    # --- fast path: GAMS ref GDX carries the benchmark rore/rorg ---
    fast = _capflex_risk_from_gdx(ref_gdx, regions)
    if fast:
        return fast

    # --- fallback: capFix twin-solve (slow) ---
    cfix = (
        closure.model_copy(update={"savf_flag": "capFix"})
        if hasattr(closure, "model_copy")
        else GTAPClosureConfig(
            name="base",
            closure_type="MCP",
            capital_mobility="sluggish",
            if_sub=bool(getattr(closure, "if_sub", False)),
            savf_flag="capFix",
        )
    )
    try:
        with (
            contextlib.redirect_stderr(io.StringIO()),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            mt, _mpt = build_block_model(params, sets, cfix, residual_region)
            solve_block_model(mt, params, cfix, None, mode="gtap")
    except Exception:
        return {}

    def _base(v, *keys):
        for k in keys:
            try:
                return float(_V(getattr(mt, v)[k]))
            except Exception:
                continue
        return None

    rorg_b = _base("rorg", "base", ("base",))
    if rorg_b is None:
        return {}
    out: dict = {}
    for r in regions:
        rore_b = _base("rore", (r, "base"))
        if rore_b and abs(rore_b) > 1e-12:
            out[r] = rorg_b / rore_b
    return out


def solve_block_model(
    m: ConcreteModel,
    params: Any,
    closure: Any,
    ref_gdx: Any,
    *,
    mode: str = "gtap",
    settle_only: bool = False,
) -> Any:
    """Seed + solve the composed multi-period block model via the monolith driver.

    Thin wrapper over ``solve_multiperiod`` — the block model is a drop-in for the
    monolith's multi-period model, so the existing NLP/MCP solve path applies.

    ``settle_only`` (lever A) returns after the check phase (no shock), for
    calibrate_base which only needs the settled check point.
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
        settle_only=settle_only,
    )


def apply_fisher_snapshot_overwrite(pm: ConcreteModel) -> dict[str, Any]:
    """Rebuild the factor-price Fisher rows against the POST-SCALING snapshot.

    ``blocks/gtap/__init__.py`` item 3 states this is the composer's job::

        The composer MUST snapshot pf0=pf.l, xf0=xf.l and recompute
        mqfactr_bb=sum_{f,a} pf0*xf0/xscale (and mqfactw_bb globally) from the
        SCALED model, OVERWRITE the CLOSURE block's pf0/xf0/mqfactr_bb/
        mqfactw_bb Params

    Nothing was doing it. The CLOSURE block seeds ``xf0`` from the UN-scaled
    benchmark (``_vfm_init``), while the monolith snapshots ``xf.l`` after
    ``apply_production_scaling``. Measured on gtap7_3x3: the two differ by exactly
    ``xscale`` — 10.0 in 26 of 45 cells — and ``mqfactw_bb`` comes out 3020.22
    against the monolith's 66.75, a factor of 45.2. ``pf0`` already matches.

    The Params are immutable, so their values are already folded into the
    constraint expressions and cannot be reassigned: the rows themselves are
    rebuilt here from post-scaling ``xf``/``pf`` levels.

    Both backends were carrying the un-scaled value, so this bias cancelled in any
    POI-vs-Pyomo comparison and only shows against the monolith or GAMS.

    Returns a summary of what was rebuilt.
    """
    from pyomo.environ import Constraint
    from pyomo.environ import sqrt as _sqrt
    from pyomo.environ import value as _value

    regions = [str(r) for r in pm.r]
    facs = [str(f) for f in pm.f]
    acts = [str(a) for a in pm.a]

    def _lvl(comp_name: str, key: tuple) -> float:
        comp = pm.find_component(comp_name)
        if comp is None:
            return 0.0
        try:
            return float(_value(comp[key]))
        except Exception:  # noqa: BLE001 - an absent cell contributes nothing
            return 0.0

    # The post-scaling snapshot: exactly what the monolith reads off the model.
    pf0 = {
        (r, f, a): _lvl("pf", (r, f, a)) for r in regions for f in facs for a in acts
    }
    xf0 = {
        (r, f, a): _lvl("xf", (r, f, a)) for r in regions for f in facs for a in acts
    }
    xscale = {(r, a): _lvl("xscale", (r, a)) or 1.0 for r in regions for a in acts}

    mqfactr_bb: dict[str, float] = {}
    mqfactw_bb = 0.0
    for r in regions:
        s_reg = 0.0
        for f in facs:
            for a in acts:
                xs = xscale[(r, a)]
                if xs <= 1e-12:
                    continue
                s_reg += pf0[(r, f, a)] * xf0[(r, f, a)] / xs
        mqfactr_bb[r] = s_reg if s_reg > 0.0 else 1.0
        mqfactw_bb += s_reg
    if mqfactw_bb <= 0.0:
        mqfactw_bb = 1.0

    def _agg(kind: str, region: str | None):
        """One aggregate of the Fisher ratio, over the live pf/xf Vars."""
        total = 0.0
        for r in [region] if region else regions:
            for f in facs:
                for a in acts:
                    xs = xscale[(r, a)]
                    if xs <= 1e-12:
                        continue
                    if kind == "bs":
                        total = total + pf0[(r, f, a)] * pm.xf[r, f, a] / xs
                    elif kind == "ss":
                        total = total + pm.pf[r, f, a] * pm.xf[r, f, a] / xs
                    else:  # "sb"
                        if xf0[(r, f, a)] <= 0.0:
                            continue
                        total = total + pm.pf[r, f, a] * xf0[(r, f, a)] / xs
        return total

    rebuilt: list[str] = []

    # The aggregates keep their own rows (the Fase-0 split): wide but linear, so
    # they never reach the symbolic differentiator.
    for tag in ("bs", "sb", "ss"):
        name = f"eq_mfw_{tag}"
        if pm.find_component(name) is not None:
            pm.del_component(name)
            pm.add_component(
                name,
                Constraint(expr=getattr(pm, f"mfw_{tag}") == _agg(tag, None)),
            )
            rebuilt.append(name)

        rname = f"eq_mfr_{tag}"
        if pm.find_component(rname) is not None:
            pm.del_component(rname)
            pm.add_component(
                rname,
                Constraint(
                    pm.r,
                    rule=lambda _m, r, _t=tag: getattr(_m, f"mfr_{_t}")[r]
                    == _agg(_t, r),
                ),
            )
            rebuilt.append(rname)

    # The Fisher indices themselves, now over the recomputed constants.
    if pm.find_component("eq_pwfact") is not None:
        pm.del_component("eq_pwfact")
        pm.add_component(
            "eq_pwfact",
            Constraint(
                expr=pm.pwfact
                == _sqrt((pm.mfw_sb / mqfactw_bb) * (pm.mfw_ss / (pm.mfw_bs + 1e-12)))
            ),
        )
        rebuilt.append("eq_pwfact")

    if pm.find_component("eq_pfact") is not None:
        pm.del_component("eq_pfact")
        pm.add_component(
            "eq_pfact",
            Constraint(
                pm.r,
                rule=lambda _m, r: _m.pfact[r]
                == _sqrt(
                    (_m.mfr_sb[r] / mqfactr_bb[r])
                    * (_m.mfr_ss[r] / (_m.mfr_bs[r] + 1e-12))
                ),
            ),
        )
        rebuilt.append("eq_pfact")

    # Reseedear las Vars auxiliares. Las filas reconstruidas calculan el agregado
    # post-escalado, pero el nivel de mfw_*/mfr_* sigue en el valor con que las
    # seedeo el bloque (sin escalar): dejarlo asi mete un residual de ~2 en
    # eq_pfact, justo donde el benchmark debe empezar en cero.
    for tag in ("bs", "sb", "ss"):
        w = pm.find_component(f"mfw_{tag}")
        if w is not None:
            w.set_value(float(_value(_agg(tag, None))), skip_validation=True)
        rv = pm.find_component(f"mfr_{tag}")
        if rv is not None:
            for r in regions:
                rv[r].set_value(float(_value(_agg(tag, r))), skip_validation=True)

    return {
        "rebuilt": rebuilt,
        "mqfactw_bb": mqfactw_bb,
        "mqfactr_bb": mqfactr_bb,
    }
