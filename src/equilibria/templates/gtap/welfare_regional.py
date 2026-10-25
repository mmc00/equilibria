"""EV del hogar regional con la definicion de GTAP (gtapv7.tab, seccion 11).

GTAP define el EV como el ingreso que, con los precios de antes del shock fijos,
alcanza la utilidad ``u`` de despues, menos el ingreso de antes
(``EV(r) = INCOMEEV(r)/100 * yev(r)``, con el sistema "sombra" del hogar regional:
consumo privado CDE, gobierno y ahorro con sus parametros de distribucion).

En niveles es el mismo bloque del hogar del modelo con otros datos: se toman sus
propias filas del periodo (``eq_yc`` ... ``eq_u``), los precios ``pa``/``pg``/``psave``
pasan a los del periodo de referencia, ``u`` queda en la del periodo, y el ingreso
``regy`` y el reparto del hogar quedan libres. Resolver ese sistema cuadrado da el
ingreso equivalente; el EV es ese ingreso menos ``regy`` de referencia. Usar las filas
del modelo evita copiar a mano parametros que el modelo lleva plegados (betas, CDE).

El modelo queda como estaba (valores y fijaciones).
"""

from __future__ import annotations

from typing import Any

import numpy as np

# El bloque del hogar regional: cuadrado (14 familias de filas y de variables). Con
# utilidad Cobb-Douglas eq_uh usa las cantidades del hogar (xaa[., hhd]): por eso van
# tambien la demanda (eq_xc) y su copia en xaa (eq_xaa_hhd), que en CDE no cambian nada.
_EQS = (
    "eq_yc", "eq_yg", "eq_rsav", "eq_phi", "eq_phip", "eq_uh", "eq_ug",
    "eq_us", "eq_u", "eq_xg_agg", "eq_zcons", "eq_xcshr", "eq_xc", "eq_xaa_hhd",
)  # fmt: skip
_FREE = (
    "regy", "yc", "yg", "rsav", "phi", "phip", "uh",
    "ug", "us", "xg_agg", "zcons", "xcshr", "xc", "xaa",
)  # fmt: skip
_PRICES = ("pa", "pg", "psave")


def _rows(m: Any, r: str, period: str) -> list:
    rows = []
    for name in _EQS:
        comp = getattr(m, name)
        for idx in comp:
            key = idx if isinstance(idx, tuple) else (idx,)
            if key[0] == r and key[-1] == period:
                rows.append(comp[idx])
    return rows


def _solve_shadow(rows: list, free: list, tol: float = 1e-12, maxit: int = 50) -> None:
    from pyomo.core.expr.calculus.derivatives import Modes, differentiate
    from pyomo.environ import value

    def resid() -> np.ndarray:
        return np.array([value(c.body) - value(c.upper) for c in rows])

    f = resid()
    for _ in range(maxit):
        if np.max(np.abs(f)) < tol:
            return
        jac = np.array(
            [
                differentiate(c.body, wrt_list=free, mode=Modes.reverse_numeric)
                for c in rows
            ]
        )
        step = np.linalg.solve(jac, -f)
        x0 = np.array([v.value for v in free])
        t = 1.0
        while True:
            for v, x in zip(free, x0 + t * step, strict=True):
                v.set_value(float(x), skip_validation=True)
            try:
                g = resid()
                ok = np.all(np.isfinite(g)) and np.linalg.norm(g) < np.linalg.norm(f)
            except (ValueError, ArithmeticError):
                ok = False
            if ok or t < 1e-6:
                break
            t /= 2
        f = g
    raise RuntimeError(
        f"sistema sombra del EV sin converger: |F| = {np.max(np.abs(f)):.2e}"
    )


def regional_ev(m: Any, period: str = "shock", ref: str = "check") -> dict[str, float]:
    """EV del hogar regional de ``period`` a precios de ``ref``, en unidades de ``regy``."""
    from pyomo.core.expr.visitor import identify_variables
    from pyomo.environ import value

    out: dict[str, float] = {}
    for r in m.r:
        rows = _rows(m, r, period)
        involved = {id(v): v for c in rows for v in identify_variables(c.body)}.values()
        free = [v for v in involved if v.parent_component().local_name in _FREE]
        prices = [v for v in involved if v.parent_component().local_name in _PRICES]
        if len(free) != len(rows):
            raise ValueError(
                f"{r}: el bloque del hogar no es cuadrado ({len(rows)} filas, {len(free)} vars)"
            )
        saved = {id(v): (v, v.value, v.fixed) for v in [*free, *prices, m.u[r, period]]}
        try:
            for v in prices:
                idx = v.index()
                v.fix(value(v.parent_component()[(*idx[:-1], ref)]))
            m.u[r, period].fix()
            for v in free:
                v.unfix()
            _solve_shadow(rows, free)
            out[r] = value(m.regy[r, period]) - value(m.regy[r, ref])
        finally:
            for v, val, fixed in saved.values():
                v.set_value(val, skip_validation=True)
                if fixed:
                    v.fix()
                else:
                    v.unfix()
    return out
