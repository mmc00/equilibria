"""Preparacion de cada periodo del driver multiperiodo (GAMS ``loop(tsim)``).

Antes de resolver un periodo, el driver lo prepara: congela los otros periodos,
siembra, recalibra (altertax), fija lo que fija el modelo de un periodo de
referencia, pone cotas, silencia la cola de bienestar, hace holdfix... Esa
secuencia vivia escrita a mano dentro del driver, una copia por periodo, con un
``if _gtap_mode`` en cada paso.

Aqui cada periodo y cada modo tiene su RECETA: la lista ordenada de pasos que le
toca.  El driver solo pregunta ``preparer.prepare(m, periodo, params)`` y recibe
el cierre con el que resolver (o ``None`` cuando el periodo no se resuelve: el
check copiado del base en F3.5).

    preparer = PeriodPreparer(mode="gtap", base_closure=..., alt_closure=...,
                              residual_region="ROW")
    closure = preparer.prepare(m, "base", params)

El orden de cada receta es el de GAMS/driver original y lo fija
``tests/templates/gtap/test_period_prep_snapshot.py`` (foto del modelo antes de
cada solve).  Los pasos llaman a los helpers del driver por atributo de modulo,
asi un ``monkeypatch`` sobre el driver sigue funcionando.
"""

from __future__ import annotations

import contextlib
import logging
import weakref
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from equilibria.templates.gtap.instruments import exogenous_test

_log = logging.getLogger(__name__)

# F3.5: vars de demanda DERIVADA que no se siembran desde el settled seed. Se
# calculan de precios+ingresos; sembrarlas en el valor del check crea una pequena
# inconsistencia. Dejarlas re-derivar sube el match contra GEMPACK en xg/xc de 78%
# a 100% (global 94%->96%). El nucleo (precios, cantidades de factores, produccion
# y comercio) si se siembra.
F35_DERIVED_DEMAND = frozenset(
    {
        "xg",
        "xc",
        "xg_agg",
        "xi",
        "xiagg",
        "yg",
        "yc",
        "zcons",
        "ug",
        "uh",
        "us",
        "u",
        "ev",
        "cv",
    }
)

# Periodo del que se recalibran las participaciones (altertax).  El shock
# recalibra siempre desde el check, aun cuando se siembre desde el base (F3.5).
_RECALIBRATE_FROM = {"check": "base", "shock": "check"}

# Ancla del numerario en el shock (GAMS fija pnum en el periodo del shock).
_SHOCK_PNUM = 1.5


def _drv():
    # Import diferido: el driver importa este modulo.
    from equilibria.templates.gtap import gtap_multiperiod_driver

    return gtap_multiperiod_driver


@dataclass
class _Ctx:
    m: Any
    period: str
    params: Any
    prep: PeriodPreparer
    closure: Any


# ── Pasos ───────────────────────────────────────────────────────────────────


def _seed_from_settled(c: _Ctx) -> None:
    """F3.5: el base arranca en el punto asentado (m._settled_seed), asi el shock
    responde como GEMPACK (~-3%) y no como el camino GAMS contaminado por el check
    (~-18%). Sin base_calibrated no hace nada (fiel a GAMS)."""
    m = c.m
    if not (c.prep.base_calibrated and getattr(m, "_settled_seed", None)):
        return
    for vn, cells in m._settled_seed.items():
        if vn in F35_DERIVED_DEMAND:
            continue
        exo = exogenous_test(m, vn)
        vobj = getattr(m, vn, None)
        if vobj is None:
            continue
        for body, val in cells.items():
            key = (*body, c.period) if isinstance(body, tuple) else (body, c.period)
            if exo(key):
                continue  # exogeno: queda en su valor fijado (benchmark o shock)
            try:
                vobj[key].set_value(float(val))
            except (KeyError, TypeError, ValueError):
                pass


def _refresh_after_settled_seed(c: _Ctx) -> None:
    """F3.5: tras la siembra asentada (y la fijacion del modelo de referencia) los
    agregados Fisher (mq_*) se recalculan de las sumas que definen, como en el
    arranque en caliente (``_seed_period_from_prior``). Son sumas DEFINIDAS, no
    incognitas: con su valor de construccion el check resuelve contra agregados del
    base (``mq_*_bb[base]``) viejos y el shock contra los refrescados (nus333
    ALTERTAX: un shock nulo no volvia al check). Sin siembra no hace nada. Espejo de
    ``family.prep_recipe._refresh_after_settled_seed``."""
    if not (c.prep.base_calibrated and getattr(c.m, "_settled_seed", None)):
        return
    from equilibria.templates.gtap.gtap_model_multiperiod import (
        GTAPMultiPeriodModel,
    )

    GTAPMultiPeriodModel.refresh_fisher_aggregates(c.m)


def _freeze_inactive(c: _Ctx) -> None:
    _drv().freeze_inactive_periods(c.m, c.period)


def _recalibrate_shares(c: _Ctx) -> None:
    """altertax: rebuild some share rows each period from the previous period.

    GAMS itself calibrates every share once, in the cal block (loop t0), and its
    GDX holds them identical in base, check and shock. Two are already held:
    - io/af: re-deriving af from the prior
      pfa overweighted subsidized Land and slid pft[Land]; holding it raised the
      shock gate 98.21% -> 99.93%.
    - alphad/alpham: re-deriving them from the prior xd/xm/xa broke the tiny
      flows (~1e-9, at the solver's tolerance), and eq_paa raises pa to the
      power 1/(1 - 0.95) = 20 under altertax; 15x10 ifSUB=0 shock 99.45% -> 100%.
    - gx/ax: the old rebuild never fired on the block model (no ``xscale``
      component, and sigmas is infinite under altertax), so it was deleted.
    Pure gtap recalibrates nothing.
    """
    d, prior = _drv(), _RECALIBRATE_FROM[c.period]
    d._recalibrate_and_ava(c.m, c.params, c.period, prior)
    d._recalibrate_alphaa_gov_inv(c.m, c.params, c.period, prior)
    d._recalibrate_alphaa_hhd(c.m, c.params, c.period, prior)


def _seed_from_prior_if_asked(c: _Ctx) -> None:
    """Solo con seed_from_prior=True. Por defecto se conserva la semilla GAMS del
    check: sembrar desde el base la pisaba (pd[USA,Mnfcs] 0.983->1.0) y PATH caia
    en una rama que colapsa el nivel de precios de USA (pgdpmp 0.99->0.67)."""
    if c.prep.seed_from_prior:
        _seed_from_prior(c)


def _seed_from_prior(c: _Ctx) -> None:
    """Arranque en caliente desde el periodo previo ya resuelto."""
    _drv()._seed_period_from_prior(c.m, c.prep.seed_prior(c.period), c.period)


def _unfix_regy(c: _Ctx) -> None:
    """regY endogeno en compStat (GAMS regYeq)."""
    m = c.m
    for r in c.params.sets.r:
        try:
            if hasattr(m, "regy") and m.regy[r, c.period].fixed:
                m.regy[r, c.period].unfix()
        except Exception:
            pass


def _fix_shock_numeraire(c: _Ctx) -> None:
    """Fija pnum del periodo en el ancla del shock (GAMS fija el numerario del
    periodo que resuelve; sin ancla el nivel de precios queda libre)."""
    pnum = getattr(c.m, "pnum", None)
    if pnum is None:
        return
    try:
        vd = pnum[c.period]
    except (KeyError, TypeError):
        return
    if not vd.fixed:
        vd.fix(_SHOCK_PNUM)


def _deactivate_redundant_xft(c: _Ctx) -> None:
    """altertax: apaga eq_xft[r,f,t] donde eq_xfteq[r,f,t] esta activa.

    El gate altertax del solver de un periodo usa el indice de 2 (eq_xft[r,f]) y
    en el multiperiodo (r,f,t) da KeyError en silencio: quedaban eq_xft Y eq_xfteq
    activas para la misma xft (sobredeterminado, code=2).  gtap NO lo hace: el
    wrapper recorta eq_xft como en el base sluggish (Hopcroft-Karp deja 6)."""
    m = c.m
    eq_xft = getattr(m, "eq_xft", None)
    eq_xfteq = getattr(m, "eq_xfteq", None)
    if eq_xft is None or eq_xfteq is None:
        return
    n = 0
    for r in m.r:
        for f in m.f:
            try:
                xfteq = eq_xfteq[(r, f, c.period)]
            except KeyError:
                continue
            if not xfteq.active:
                continue
            try:
                xft = eq_xft[(r, f, c.period)]
            except KeyError:
                continue
            if xft.active:
                xft.deactivate()
                n += 1
    if n:
        _log.info(
            "%s period: deactivated eq_xft for %d (r,f) pairs "
            "(eq_xfteq active → eq_xft redundant, multi-period index fix)",
            c.period,
            n,
        )


def _collapse_pft(c: _Ctx) -> None:
    """gtap: cuadra el bloque de precios de factores del periodo (el cuadrado del
    base de un periodo no actua sobre el indice (r,f,t))."""
    n = _drv()._collapse_pft_pfteq(c.m, c.period)
    if n:
        _log.info(
            "%s period: collapsed pft/eq_pfteq for %d (r,f) pairs "
            "(gtap-mode factor-price squaring)",
            c.period,
            n,
        )


def _replicate_sp_reference(c: _Ctx) -> None:
    """Copia lo que fija y las cotas de un modelo de UN periodo con el mismo cierre:
    build_model fija ~500 ceros estructurales (afeall, p_rai, chiSave...) que
    apply_conditional_fixing no cubre; sin esto el matching fija las 639 vars
    equivocadas."""
    _replicate(c, c.prep._reference_model(c.m, c.params, c.closure, c.period))


def _replicate_own_sp_reference(c: _Ctx) -> None:
    """Como _replicate_sp_reference, pero con un modelo de referencia construido
    con los params de ESTE periodo (los del shock) y sin reutilizar el del base."""
    _replicate(c, c.prep._build_reference_model(c.m, c.params, c.closure))


def _replicate(c: _Ctx, snap) -> None:
    d = _drv()
    d._replicate_snapshot_fixing(c.m, snap, c.period)
    d._replicate_snapshot_bounds(c.m, snap, c.period)


def _gams_bounds(c: _Ctx) -> None:
    _drv()._apply_gams_bounds(c.m, c.period)


def _mute_welfare_if_asked(c: _Ctx) -> None:
    """Silencia la cola inerte de bienestar para que PATH certifique code=1 (con el
    base exacto, el check pasa de code=2 res 1.1e-2 a code=1 res 2.9e-11)."""
    if not c.prep.mute_welfare:
        return
    n = _drv()._mute_welfare_tail(
        c.m, c.period, list(c.params.sets.r), gtap_mode=c.prep.mode == "gtap"
    )
    if n:
        _log.info(
            "%s period: muted %d welfare-leaf rows (cv/ev/walras/u/ug/us)",
            c.period,
            n,
        )


def _fnm_pf(c: _Ctx) -> None:
    """pf de factores especificos con etaff=0 fijo al periodo previo (GAMS
    pf.fx(r,fp,a,tsim-1)); si no es un grado de libertad que PATH hace explotar."""
    n = _drv()._holdfix_fnm_pf(c.m, c.params, c.period)
    if n:
        _log.info("%s period: holdfixed %d fnm pf cells (etaff=0)", c.period, n)


def _copy_from_base(c: _Ctx) -> None:
    """F3.5 sin solve_check: el check ES el base asentado."""
    _drv()._copy_base_to_check(c.m)


# ── Recetas ─────────────────────────────────────────────────────────────────

Step = Callable[[_Ctx], None]


@dataclass(frozen=True)
class _Recipe:
    steps: tuple[Step, ...]
    closure: str | None  # "base" | "altertax" | None (= el periodo no se resuelve)


_BASE = _Recipe(
    steps=(
        _seed_from_settled,
        _freeze_inactive,
        _replicate_sp_reference,
        _refresh_after_settled_seed,
    ),
    closure="base",
)

_RECIPES: dict[tuple[str, str], _Recipe] = {
    ("base", "gtap"): _BASE,
    ("base", "altertax"): _BASE,
    ("check", "gtap"): _Recipe(
        steps=(
            _freeze_inactive,
            _seed_from_prior_if_asked,
            _unfix_regy,
            _collapse_pft,
            _replicate_sp_reference,
            _gams_bounds,
            _mute_welfare_if_asked,
            _fnm_pf,
        ),
        closure="base",
    ),
    ("check", "altertax"): _Recipe(
        steps=(
            _freeze_inactive,
            _recalibrate_shares,
            _seed_from_prior_if_asked,
            _unfix_regy,
            _deactivate_redundant_xft,
            _replicate_sp_reference,
            _mute_welfare_if_asked,
            _fnm_pf,
        ),
        closure="altertax",
    ),
    # El shock se prepara ANTES de aplicarlo: freeze_inactive_periods reactivaria
    # filas que los rebuilds del arancel desactivan.
    ("shock", "gtap"): _Recipe(
        steps=(
            _freeze_inactive,
            _seed_from_prior,
            _unfix_regy,
            _fix_shock_numeraire,
            _collapse_pft,
            _replicate_own_sp_reference,
            _gams_bounds,
            _mute_welfare_if_asked,
            _fnm_pf,
        ),
        closure="base",
    ),
    ("shock", "altertax"): _Recipe(
        steps=(
            _freeze_inactive,
            _recalibrate_shares,
            _seed_from_prior,
            _unfix_regy,
            _fix_shock_numeraire,
            _deactivate_redundant_xft,
            _replicate_own_sp_reference,
            _mute_welfare_if_asked,
            _fnm_pf,
        ),
        closure="altertax",
    ),
}

_CHECK_COPIED = _Recipe(steps=(_copy_from_base,), closure=None)


# Foto del modelo de referencia compartida entre las corridas que reciben el mismo
# objeto ``params``: id(params) -> (ref debil a params, clave, SPSnapshot).  Por id
# y no como clave debil porque ``GTAPParameters`` no es hasheable (dataclass con
# igualdad); la ref debil confirma que el id sigue siendo de ese objeto y un
# finalizador borra la entrada cuando ``params`` muere.  Fuera de los atributos de
# ``params`` y de ``m``: el cache del modelo los serializa, y su clave digiere todos
# los atributos de ``params``.
_SHARED_SP_REF: dict[int, tuple[weakref.ref, str, Any]] = {}


def _shared_get(params, key: str):
    entry = _SHARED_SP_REF.get(id(params))
    if entry is None or entry[0]() is not params or entry[1] != key:
        return None
    return entry[2]


def _shared_put(params, key: str, snap) -> None:
    pid = id(params)
    if pid not in _SHARED_SP_REF:
        weakref.finalize(params, _SHARED_SP_REF.pop, pid, None)
    _SHARED_SP_REF[pid] = (weakref.ref(params), key, snap)


class PeriodPreparer:
    """Prepara los periodos de UNA corrida del driver.

    Guarda el modelo de referencia de un periodo entre llamadas (el check gtap
    reutiliza el del base).  ``prepare`` devuelve el cierre con el que resolver
    el periodo, o ``None`` si el periodo no se resuelve.
    """

    def __init__(
        self,
        *,
        mode: str,
        base_closure: Any,
        alt_closure: Any,
        residual_region: str,
        mute_welfare: bool = True,
        seed_from_prior: bool = False,
        base_calibrated: bool = False,
        solve_check: bool = False,
    ) -> None:
        if mode not in ("gtap", "altertax"):
            raise ValueError(f"mode must be 'altertax' or 'gtap', got {mode!r}")
        self.mode = mode
        self.residual_region = residual_region
        self.mute_welfare = mute_welfare
        self.seed_from_prior = seed_from_prior
        self.base_calibrated = base_calibrated
        self.solve_check = solve_check
        self._closures = {"base": base_closure, "altertax": alt_closure}
        self._sp_ref = None
        self._sp_ref_closure = None

    def _build_reference_model(self, m, params, closure):
        """Foto (``SPSnapshot``) del modelo de un periodo con ``closure``; el modelo
        Pyomo se suelta apenas se lee."""
        d = _drv()
        return d._sp_snapshot(
            d._build_sp_reference(
                params.sets, params, closure, self.residual_region, model=m
            )
        )

    def _reference_model(self, m, params, closure, period: str):
        """Foto del modelo de un periodo con ``closure``.  Se guarda la del base y
        se reutiliza mientras el cierre no cambie (el check gtap usa el mismo); una
        construida para otro cierre se usa y se descarta.

        Ademas se comparte entre corridas con el MISMO objeto ``params``:
        ``calibrate_base`` (sobre otro ``m``) y ``solve_multiperiod`` crean cada uno
        su ``PeriodPreparer`` y construian dos veces la misma (~46 s cada una en
        20x41; identicas, medido)."""
        d = _drv()
        if self._sp_ref is not None and d._sp_ref_reusable(
            self._sp_ref_closure, closure
        ):
            _log.info(
                "%s period: reusing the base reference model (identical closure)",
                period,
            )
            return self._sp_ref
        key = self._shared_key(m, params, closure)
        snap = _shared_get(params, key) if key is not None else None
        if snap is not None:
            _log.info(
                "%s period: reusing the reference model of a previous run (same key)",
                period,
            )
        else:
            snap = self._build_reference_model(m, params, closure)
            if key is not None:
                _shared_put(params, key, snap)
        if self._sp_ref is None:
            self._sp_ref, self._sp_ref_closure = snap, closure
        return snap

    def _shared_key(self, m, params, closure) -> str | None:
        """Clave por CONTENIDO de todo lo que entra al modelo de referencia (params,
        cierre, region residual, fuente del modelo y codigo), o ``None`` si no se
        puede compartir.

        De ``m``, ``_build_sp_reference`` solo lee la fuente (``_sp_source``, que
        ``EQUILIBRIA_GTAP_REF_MODEL`` pisa).  Con un hook ``@overwrite`` activo no
        se comparte: cambia el modelo y no esta en la clave (igual que el cache
        del modelo)."""
        import os

        from equilibria.blocks.gtap.fingerprint import fingerprint
        from equilibria.blocks.gtap.overwrite import overwrite

        if overwrite.active():
            return None
        return fingerprint(
            "sp_ref",
            params=params,
            closure=closure,
            residual_region=self.residual_region,
            sp_source=getattr(m, "_sp_source", None),
            ref_model_env=os.environ.get("EQUILIBRIA_GTAP_REF_MODEL"),
        )

    def seed_prior(self, period: str) -> str:
        """Periodo del que se siembra ``period``.  El shock se siembra del check,
        salvo en F3.5 sin solve_check: ahi el check no se resolvio y se usa el base
        asentado."""
        if period == "shock":
            return "base" if self._check_copied() else "check"
        return "base"

    def _check_copied(self) -> bool:
        """F3.5 sin solve_check: el check no se resuelve, se copia del base."""
        return self.base_calibrated and not self.solve_check

    def recipe(self, period: str) -> _Recipe:
        if period == "check" and self._check_copied():
            return _CHECK_COPIED
        try:
            return _RECIPES[(period, self.mode)]
        except KeyError:
            raise ValueError(f"no recipe for period {period!r}") from None

    def prepare(self, m, period: str, params) -> Any:
        from equilibria.family.prep_capture import capture_state

        capture_state(m, period)  # opt-in oracle for the array port; no-op by default
        recipe = self.recipe(period)
        closure = None if recipe.closure is None else self._closures[recipe.closure]
        ctx = _Ctx(m=m, period=period, params=params, prep=self, closure=closure)
        for step in recipe.steps:
            step(ctx)
        capture_state(m, period, tag="A2")
        return closure
