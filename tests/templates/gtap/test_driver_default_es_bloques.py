"""El monolito hay que PEDIRLO: sin marcador, el driver va a BLOQUES.

Antes de F3 un modelo sin `_sp_source` caia al monolito.  Eso era lo
conservador mientras las dos clases convivian sin declararse, pero convierte el
monolito —que F3 retira— en el default que se hereda por olvido: basta una
clase nueva que no marque nada para volver a construir 3,4M de celdas en 20x41.

Ahora las DOS clases declaran su fuente (`GTAPMultiPeriodModel` -> "monolith",
`GTAPBlockMultiPeriodModel` -> "blocks"), asi que el implicito ya no protege a
nadie: solo esconde olvidos.  Estos tests fijan el contrato en las tres
direcciones.
"""

from __future__ import annotations

import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
DATASETS = ROOT / "datasets"


def _params():
    from equilibria.templates.gtap import GTAPParameters

    d = DATASETS / "gtap7_3x3"
    p = GTAPParameters()
    p.load_from_har(
        basedata_path=d / "basedata.har",
        sets_path=d / "sets.har",
        default_path=d / "default.prm",
        baserate_path=d / "baserate.har",
    )
    return p


def _closure():
    from equilibria.templates.gtap.gtap_contract import GTAPClosureConfig

    return GTAPClosureConfig(
        name="base",
        closure_type="MCP",
        capital_mobility="sluggish",
        fix_endowments=False,
        fix_taxes=False,
        fix_technology=False,
        if_sub=False,
        numeraire="pnum",
    )


def _cuenta_monolitos(model) -> int:
    """Construye el SP de referencia y cuenta cuantos monolitos se instancian."""
    import equilibria.templates.gtap as T
    from equilibria.templates.gtap.gtap_multiperiod_driver import _build_sp_reference

    p, gc = _params(), _closure()
    rr = list(p.sets.r)[-1]

    n = {"v": 0}
    original = T.GTAPModelEquations

    class Espia(original):  # type: ignore[misc, valid-type]
        def __init__(self, *a, **k):
            n["v"] += 1
            super().__init__(*a, **k)

    T.GTAPModelEquations = Espia
    try:
        _build_sp_reference(p.sets, p, gc, rr, model=model)
    finally:
        T.GTAPModelEquations = original
    return n["v"]


def test_sin_marcador_va_a_bloques():
    """El caso que importa: una clase que no declara nada NO revive el monolito."""

    class ModeloSinMarcador:
        pass

    assert _cuenta_monolitos(ModeloSinMarcador()) == 0


def test_marcado_monolith_construye_el_monolito():
    """Pedirlo explicitamente sigue funcionando — los 7 scripts de medicion
    contra GAMS dependen de ello."""

    class ModeloMonolito:
        _sp_source = "monolith"

    assert _cuenta_monolitos(ModeloMonolito()) == 1


def test_las_dos_clases_declaran_su_fuente():
    """Ninguna de las dos se apoya en el implicito."""
    from equilibria.templates.gtap.gtap_block_model import GTAPBlockMultiPeriodModel
    from equilibria.templates.gtap.gtap_model_multiperiod import GTAPMultiPeriodModel

    p = _params()
    rr = list(p.sets.r)[-1]
    esperado = {
        GTAPMultiPeriodModel: "monolith",
        GTAPBlockMultiPeriodModel: "blocks",
    }
    for cls, fuente in esperado.items():
        m = cls(p.sets, p, None, residual_region=rr).build_sets()
        assert getattr(m, "_sp_source", None) == fuente, (
            f"{cls.__name__} deberia declarar _sp_source={fuente!r}"
        )


def test_la_env_var_sigue_forzando_el_monolito(monkeypatch):
    """La via de escape documentada no se pierde."""
    monkeypatch.setenv("EQUILIBRIA_GTAP_REF_MODEL", "monolith")

    class ModeloSinMarcador:
        pass

    assert _cuenta_monolitos(ModeloSinMarcador()) == 1
