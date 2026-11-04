"""El monolito hay que PEDIRLO: sin marcador, el driver va a BLOQUES.

Antes de F3 un modelo sin `_sp_source` caia al monolito.  Eso era lo
conservador mientras las dos clases convivian sin declararse, pero convierte el
monolito —que F3 retira— en el default que se hereda por olvido: basta una
clase nueva que no marque nada para volver a construir 3,4M de celdas en 20x41.

Ahora las DOS clases declaran su fuente (`GTAPMultiPeriodModel` -> "monolith",
`GTAPBlockMultiPeriodModel` -> "blocks"), asi que el implicito ya no protege a
nadie: solo esconde olvidos.  Estos tests fijan el contrato en las cuatro
direcciones, incluida la validacion: un valor no reconocido LEVANTA en vez de
caer al default en silencio.
"""

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
import pytest
from test_multiperiod_sets import _load_3x3_params


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


def _cuenta_monolitos(model, monkeypatch) -> int:
    """Construye el SP de referencia y cuenta cuantos monolitos se instancian.

    Cuenta con `monkeypatch.setattr` sobre `__init__`, como los dos vecinos que
    miden lo mismo (`test_driver_ref_model_source`, `test_mp_builder_uses_blocks`).
    """
    from equilibria.templates.gtap import gtap_model_equations as ME
    from equilibria.templates.gtap.gtap_multiperiod_driver import _build_sp_reference

    p, gc = _load_3x3_params(), _closure()
    rr = list(p.sets.r)[-1]

    n = {"v": 0}
    original = ME.GTAPModelEquations.__init__

    def contando(self, *a, **k):
        n["v"] += 1
        original(self, *a, **k)

    monkeypatch.setattr(ME.GTAPModelEquations, "__init__", contando)
    _build_sp_reference(p.sets, p, gc, rr, model=model)
    return n["v"]


def test_sin_marcador_va_a_bloques(monkeypatch):
    """El caso que importa: una clase que no declara nada NO revive el monolito."""

    class ModeloSinMarcador:
        pass

    assert _cuenta_monolitos(ModeloSinMarcador(), monkeypatch) == 0


def test_marcado_monolith_construye_el_monolito(monkeypatch):
    """Pedirlo explicitamente sigue funcionando — los 7 scripts de medicion
    contra GAMS dependen de ello."""

    class ModeloMonolito:
        _sp_source = "monolith"

    assert _cuenta_monolitos(ModeloMonolito(), monkeypatch) == 1


def test_las_dos_clases_declaran_su_fuente():
    """Ninguna de las dos se apoya en el implicito."""
    from equilibria.templates.gtap.gtap_block_model import GTAPBlockMultiPeriodModel
    from equilibria.templates.gtap.gtap_model_multiperiod import GTAPMultiPeriodModel

    p = _load_3x3_params()
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

    assert _cuenta_monolitos(ModeloSinMarcador(), monkeypatch) == 1


def test_un_valor_no_reconocido_LEVANTA(monkeypatch):
    """Un typo no puede caer al default en silencio.

    Antes el driver solo comparaba `== "monolith"`, asi que `"monolit"` —o el
    `"blocks"` que un test creia que pedia bloques— se ignoraba sin avisar.
    Medido: con `"BASURA_XYZ"` en la env var el test pasaba igual.
    """

    class ModeloConTypo:
        _sp_source = "monolit"  # falta la 'h'

    with pytest.raises(ValueError, match="no reconocido"):
        _cuenta_monolitos(ModeloConTypo(), monkeypatch)

    monkeypatch.setenv("EQUILIBRIA_GTAP_REF_MODEL", "BASURA_XYZ")

    class ModeloSinMarcador:
        pass

    with pytest.raises(ValueError, match="no reconocido"):
        _cuenta_monolitos(ModeloSinMarcador(), monkeypatch)
