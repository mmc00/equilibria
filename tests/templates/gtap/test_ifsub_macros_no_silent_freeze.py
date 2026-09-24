"""`_apply_ifsub_closure` no puede degradar al init en silencio.

Los 9 vars de reporte de ifSUB se DESACTIVAN y se FIJAN: sin ecuacion que los
determine, el valor fijado ES el definitivo.  Congelar "el valor actual" hace
DEFINITIVA la diferencia de init entre el monolito (re-valua tras escalar) y
bloques (siembra del benchmark) — medido `pwmg` en 0.001 vs 1.0, factor 1000,
que dejaba el gate MCP de gtap7_15x10 pure ifSUB=1 en 87,00%.

Por eso hay dos contratos que estos tests fijan:

1. cada var de reporte con macro toma el valor del MACRO, no el del init;
2. si un macro falla, se LEVANTA — nunca se cae al init sin avisar.

El (2) se prueba sustituyendo un macro por uno que revienta: sin el guard, el
build termina limpio y el fallo solo se ve 11.118 celdas mas tarde, al resolver
el shock.
"""

from __future__ import annotations

import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
DATASETS = ROOT / "datasets"


def _sp_ifsub(dataset: str = "gtap7_3x3"):
    from equilibria.templates.gtap import GTAPParameters
    from equilibria.templates.gtap.gtap_block_model import build_block_single_period
    from equilibria.templates.gtap.gtap_contract import GTAPClosureConfig

    d = DATASETS / dataset
    p = GTAPParameters()
    p.load_from_har(
        basedata_path=d / "basedata.har",
        sets_path=d / "sets.har",
        default_path=d / "default.prm",
        baserate_path=d / "baserate.har",
    )
    gc = GTAPClosureConfig(
        name="base",
        closure_type="MCP",
        capital_mobility="sluggish",
        fix_endowments=False,
        fix_taxes=False,
        fix_technology=False,
        if_sub=True,
        numeraire="pnum",
    )
    return build_block_single_period(p, p.sets, gc, list(p.sets.r)[-1]), p


def test_xmgm_se_fija_desde_su_macro_no_desde_el_init():
    """`xmgm` toma CUATRO indices; el guard `len(k) == 3` lo dejaba fuera.

    Su macro existe (`_ifsub_macros.m_xmgm`), asi que quedaba congelado al init
    por un detalle de aridad — el mismo fallo que el resto del PR corrige.
    """
    from pyomo.environ import value

    from equilibria.blocks.gtap import _ifsub_macros as mac

    sp, p = _sp_ifsub()
    xm = sp.component("xmgm")
    assert xm is not None, "xmgm no esta declarado en el SP de bloques"

    celdas = list(xm)
    assert celdas, "xmgm no tiene celdas"
    assert all(xm[k].fixed for k in celdas), "ifSUB debe FIJAR los vars de reporte"

    distintas = []
    for k in celdas:
        via_macro = float(value(mac.m_xmgm(sp, p, *k)))
        if abs(float(value(xm[k])) - via_macro) > 1e-12:
            distintas.append((k, float(value(xm[k])), via_macro))

    assert not distintas, (
        f"{len(distintas)} celdas de xmgm NO tienen el valor de su macro "
        f"(quedaron en el init). Ejemplo: {distintas[0]}"
    )


def test_un_macro_que_falla_LEVANTA_en_vez_de_congelar_el_init(monkeypatch):
    """Sin este guard el build termina limpio y el bug viaja al solve."""
    from equilibria.templates.gtap import gtap_block_model as M

    def _macro_roto(model, p, e, c, imp):
        raise ValueError("macro roto a proposito")

    real = M._apply_ifsub_closure

    def _con_macro_roto(pm, params=None):
        from equilibria.blocks.gtap import _ifsub_macros as mac

        monkeypatch.setattr(mac, "m_pwmg", _macro_roto, raising=True)
        return real(pm, params)

    monkeypatch.setattr(M, "_apply_ifsub_closure", _con_macro_roto, raising=True)

    with pytest.raises(RuntimeError, match="no pudo fijar"):
        _sp_ifsub()
