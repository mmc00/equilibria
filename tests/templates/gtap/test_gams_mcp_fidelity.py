"""El MCP de un periodo resuelto (check/shock) se arma como el de GAMS.

Tres diferencias medidas en ME9B (nus333) que mandaban a PATH a otra raiz aunque
el sistema de ecuaciones fuera el mismo:

1. ``walras`` es libre en GAMS (``walraseq`` completa el sistema); equilibria la
   fijaba en 0 y el cuadrador desactivaba una fila real (``eq_xseq``).
2. Cotas: GAMS solo acota los precios y utilidades de iterloop.gms:61-88 (``pft``
   esta comentada en :79); todo lo demas es libre (model.gms:35 declara todas las
   variables como ``Variables``). equilibria dejaba ``lb=0`` por el dominio
   ``NonNegativeReals`` aunque se hiciera ``setlb(None)``.
3. Parejas: ``pfeq.pf``, ``pfyeq.pfy`` y ``pdeq.pd`` son parejas declaradas en el
   ``model`` de GAMS; Hopcroft-Karp las reasignaba por caminos de aumento.
"""

from __future__ import annotations

import importlib
from typing import Any, cast

import pytest
from tests.templates.gtap._nus333 import closure, nus333_params


def _mod(name: str) -> Any:
    return cast(Any, importlib.import_module(name))


DRV = "equilibria.templates.gtap.gtap_multiperiod_driver"
PC = "equilibria.solver.path_capi"


def _nus333_model():
    from equilibria.templates.gtap.gtap_block_model import build_block_model

    p = nus333_params()
    gc = closure()
    m, _ = build_block_model(p, p.sets, gc, "ROW", base_calibrated=False, ref_gdx=None)
    return cast(Any, m)


@pytest.mark.integration
def test_walras_libre_y_eq_walras_activa_en_modo_gtap():
    from equilibria.templates.gtap.gtap_multiperiod_driver import _mute_welfare_tail

    m = _nus333_model()
    _mute_welfare_tail(m, "check", list(m.r), gtap_mode=True)

    def _per(k):
        return k[-1] if isinstance(k, tuple) else k

    w = [m.walras[k] for k in m.walras if _per(k) == "check"]
    assert w, "sin walras en check"
    assert not any(v.fixed for v in w)
    assert all(m.eq_walras[k].active for k in m.eq_walras if _per(k) == "check")


@pytest.mark.integration
def test_cotas_de_gams_en_el_periodo_resuelto():
    drv = _mod(DRV)
    GAMS_BOUNDED_VARS, _apply_gams_bounds = (
        drv.GAMS_BOUNDED_VARS,
        drv._apply_gams_bounds,
    )

    m = _nus333_model()
    pf_lb = {k: m.pf[k].lb for k in m.pf if k[-1] == "check"}
    base_lb = {k: m.xf[k].lb for k in m.xf if k[-1] == "base"}
    _apply_gams_bounds(m, "check")
    # Libres en GAMS: sin cota, ni siquiera la del dominio.
    for v in (
        m.xft["USA", "LAND", "check"],
        m.pft["USA", "LAND", "check"],
        m.yc["USA", "check"],
        m.xf["USA", "LAND", "AGR", "check"],
        m.pp_rai["USA", "AGR", "AGR", "check"],
    ):
        assert v.lb is None and v.ub is None, v.name
    # Acotadas en GAMS: no se tocan.
    assert {k: m.pf[k].lb for k in pf_lb} == pf_lb
    # Otro periodo: no se toca.
    assert {k: m.xf[k].lb for k in base_lb} == base_lb
    assert "pft" not in GAMS_BOUNDED_VARS and "pf" in GAMS_BOUNDED_VARS


def test_pisos_relativos_del_solver_en_modo_gtap_son_los_de_gams():
    _relative_floor_vars = _mod(PC)._relative_floor_vars

    prices, levels = _relative_floor_vars(gtap_mode=True)
    assert "pft" not in prices  # iterloop.gms:79 comentada
    assert not {"yc", "phi", "phip", "regy"} & set(levels)  # cal.gms:646-650
    # Fuera del modo gtap, sin cambios.
    prices_alt, levels_alt = _relative_floor_vars(gtap_mode=False)
    assert "pft" in prices_alt and "yc" in levels_alt


def test_pareja_dura_no_se_reasigna():
    """c1 tiene x e y; c2 solo x. Con (c1, x) dura, Hopcroft-Karp no puede
    mover c1 a y para darle x a c2."""
    from pyomo.environ import ConcreteModel, Constraint, Var

    from equilibria.solver._closure_patches import structural_matching

    m = cast(Any, ConcreteModel())
    m.x = Var(initialize=1.0)
    m.y = Var(initialize=1.0)
    m.c1 = Constraint(expr=m.x + m.y == 2)
    m.c2 = Constraint(expr=m.x == 1)
    perm = structural_matching(
        [m.c1, m.c2], [m.x, m.y], forced_pairs=[("c1", "x", True)]
    )
    assert perm[0] is m.x


def test_parejas_duras_de_gams_en_modo_gtap():
    _GAMS_HARD_PAIR_EQS = _mod(PC)._GAMS_HARD_PAIR_EQS

    assert {"eq_pfeq", "eq_pfyeq", "eq_pdeq"} <= set(_GAMS_HARD_PAIR_EQS)
