"""El seed de ``rsav`` en bloques debe valer lo que GAMS, tambien donde es negativo.

``rsav`` (ahorro regional) se sembraba en CERO en el modelo de bloques
(``blocks/gtap/income.py``), mientras el monolito lo deriva del benchmark
(``get_rsav_init``: usa ``save`` cuando es positivo y, si no, lo recalcula como
``regY - yc - yg``).

Para casi todas las regiones da igual: el solve lleva ``rsav`` a su valor de todos
modos. Pero una region DISSAVER --``EGY`` en ``gtap7_3x4``, con ``save`` negativo--
arranca desde un punto sesgado, y en ``altertax`` (donde los impuestos se mueven y
``regY`` se recalcula) el solve cae en otra rama: 15 celdas de ``EGY``
--``regY``/``rore``/``rsav``/``chif``/``yc``/``yg``/``ytax``/``xa``/``xd``/``xm``--
divergen ~1% de GAMS, y el gate NLP baja de 99.8% a 98.3%.

Este test mide el SEED contra GAMS directamente, sin resolver: es donde nace el
sesgo, y asi falla en segundos en vez de tras un solve completo.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
for _p in (ROOT / "src", ROOT / "scripts" / "gtap"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

DATASET = "gtap7_3x4"
# EGY es la unica region dissaver del dataset (save < 0); las otras tres pasaban
# ya antes del arreglo, y estan aqui para que una regresion que las rompa se vea.
GAMS_RSAV_BASE = {
    "EGY": -0.01233539844,
    "EU_28": 2.13391125,
    "ROW": 8.488486,
    "USA": 1.580785625,
}
RTOL = 1e-3


@pytest.fixture(scope="module")
def _block_model():
    d = ROOT / "datasets" / DATASET
    if not (d / "basedata.har").exists():
        pytest.skip(f"dataset HAR ausente: {DATASET}")

    from equilibria.templates.gtap import GTAPParameters
    from equilibria.templates.gtap.gtap_block_model import build_block_single_period

    p = GTAPParameters()
    p.load_from_har(
        basedata_path=d / "basedata.har",
        sets_path=d / "sets.har",
        default_path=d / "default.prm",
        baserate_path=d / "baserate.har",
    )
    rres = list(p.sets.r)[-1]
    return build_block_single_period(p, p.sets, None, rres)


@pytest.mark.parametrize("region", sorted(GAMS_RSAV_BASE))
def test_rsav_seed_matches_gams(_block_model, region):
    """El seed de ``rsav`` coincide con el nivel base de GAMS en cada region."""
    from pyomo.environ import value

    esperado = GAMS_RSAV_BASE[region]
    obtenido = float(value(_block_model.rsav[region]))
    denom = max(abs(esperado), 1e-12)
    assert abs(obtenido - esperado) / denom <= RTOL, (
        f"rsav[{region}] sembrado en {obtenido!r}, GAMS da {esperado!r} "
        f"(error relativo {abs(obtenido - esperado) / denom:.3%})"
    )


def test_rsav_seed_is_negative_for_the_dissaver(_block_model):
    """La region dissaver conserva el SIGNO.

    Un seed en cero, o clampeado a positivo, pasa desapercibido en las otras
    regiones; este aserto fija lo que distingue el caso.
    """
    from pyomo.environ import value

    assert float(value(_block_model.rsav["EGY"])) < 0.0, (
        "rsav[EGY] debe ser negativo: EGY es dissaver (save = -0.0123 en el "
        "benchmark). Un cero o un valor positivo aqui manda la region a otra "
        "rama del solve en altertax."
    )
