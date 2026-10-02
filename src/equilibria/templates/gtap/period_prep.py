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

import logging
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

# Periodo del que se recalibra / siembra cada periodo activo.
_PRIOR = {"check": "base"}


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


def _freeze_inactive(c: _Ctx) -> None:
    _drv().freeze_inactive_periods(c.m, c.period)


def _recalibrate_shares(c: _Ctx) -> None:
    """altertax: GAMS iterloop recalibra las participaciones cada periodo.

    NO se recalibra io/af (_recalibrate_io_af): GAMS los mantiene constantes entre
    periodos (iterloop.gms no asigna af( ni io(); el GDX los muestra identicos
    base=check=shock). Recalibrar af desde el pfa del periodo previo sobrepesaba la
    Land subsidiada y deslizaba pft[Land]; quitarlo subio el gate de shock
    98.21% -> 99.93%.  gtap puro no recalibra nada: GAMS calibra UNA vez en t0.
    """
    d, prior = _drv(), _PRIOR[c.period]
    d._recalibrate_and_ava(c.m, c.params, c.period, prior)
    d._recalibrate_gx_ax(c.m, c.params, c.period, prior)
    d._recalibrate_alphad_alpham(c.m, c.params, c.period, prior)
    d._recalibrate_alphaa_gov_inv(c.m, c.params, c.period, prior)
    d._recalibrate_alphaa_hhd(c.m, c.params, c.period, prior)


def _seed_from_prior_if_asked(c: _Ctx) -> None:
    """Solo con seed_from_prior=True. Por defecto se conserva la semilla GAMS del
    check: sembrar desde el base la pisaba (pd[USA,Mnfcs] 0.983->1.0) y PATH caia
    en una rama que colapsa el nivel de precios de USA (pgdpmp 0.99->0.67)."""
    if c.prep.seed_from_prior:
        _drv()._seed_period_from_prior(c.m, _PRIOR[c.period], c.period)


def _unfix_regy(c: _Ctx) -> None:
    """regY endogeno en compStat (GAMS regYeq)."""
    m = c.m
    for r in c.params.sets.r:
        try:
            if hasattr(m, "regy") and m.regy[r, c.period].fixed:
                m.regy[r, c.period].unfix()
        except Exception:
            pass


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
    d = _drv()
    sp = c.prep._reference_model(c.m, c.params, c.closure, c.period)
    d._replicate_sp_fixing(c.m, sp, c.period)
    d._replicate_sp_bounds(c.m, sp, c.period)


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


def _derived_seed_if_holdfix(c: _Ctx) -> None:
    """altertax + holdfix_cd: siembra las vars de demanda derivadas (xc/xg/xi/xd/
    xmt/xiagg) desde las primales sembradas, asi el punto GAMS es un punto fijo."""
    if not c.prep.holdfix_cd:
        return
    n = _drv()._complete_derived_seed(c.m, c.period)
    if n:
        _log.info("%s period: seeded %d derived demand-volume cells", c.period, n)


def _cd_nest_if_holdfix(c: _Ctx) -> None:
    """altertax + holdfix_cd: holdfix del nido VA/ND degenerado bajo CD (pva/pnd)
    en los valores sembrados (GAMS holdfixed=1); si no, PATH los desliza."""
    if not c.prep.holdfix_cd:
        return
    n = _drv()._holdfix_cd_nest(c.m, c.period)
    if n:
        _log.info("%s period: holdfixed %d CD-nest cells (pva/pnd)", c.period, n)


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
    steps=(_seed_from_settled, _freeze_inactive, _replicate_sp_reference),
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
            _derived_seed_if_holdfix,
            _cd_nest_if_holdfix,
            _fnm_pf,
        ),
        closure="altertax",
    ),
}

_CHECK_COPIED = _Recipe(steps=(_copy_from_base,), closure=None)


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
        holdfix_cd: bool = True,
        base_calibrated: bool = False,
        solve_check: bool = False,
    ) -> None:
        if mode not in ("gtap", "altertax"):
            raise ValueError(f"mode must be 'altertax' or 'gtap', got {mode!r}")
        self.mode = mode
        self.residual_region = residual_region
        self.mute_welfare = mute_welfare
        self.seed_from_prior = seed_from_prior
        self.holdfix_cd = holdfix_cd
        self.base_calibrated = base_calibrated
        self.solve_check = solve_check
        self._closures = {"base": base_closure, "altertax": alt_closure}
        self._sp_ref = None
        self._sp_ref_closure = None

    def _reference_model(self, m, params, closure, period: str):
        """Modelo de un periodo con ``closure``.  Se guarda el del base y se
        reutiliza mientras el cierre no cambie (el check gtap usa el mismo); uno
        construido para otro cierre se usa y se descarta."""
        d = _drv()
        if self._sp_ref is not None and d._sp_ref_reusable(
            self._sp_ref_closure, closure
        ):
            _log.info(
                "%s period: reusing the base reference model (identical closure)",
                period,
            )
            return self._sp_ref
        sp = d._build_sp_reference(
            params.sets, params, closure, self.residual_region, model=m
        )
        if self._sp_ref is None:
            self._sp_ref, self._sp_ref_closure = sp, closure
        return sp

    def recipe(self, period: str) -> _Recipe:
        if period == "check" and self.base_calibrated and not self.solve_check:
            return _CHECK_COPIED
        try:
            return _RECIPES[(period, self.mode)]
        except KeyError:
            raise ValueError(f"no recipe for period {period!r}") from None

    def prepare(self, m, period: str, params) -> Any:
        recipe = self.recipe(period)
        closure = None if recipe.closure is None else self._closures[recipe.closure]
        ctx = _Ctx(m=m, period=period, params=params, prep=self, closure=closure)
        for step in recipe.steps:
            step(ctx)
        return closure
