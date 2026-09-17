"""El seed de ``ytax`` en bloques debe satisfacer ``eq_ytax`` en el benchmark.

El modelo de bloques siembra las diez corrientes de ``ytax`` en CERO
(``blocks/gtap/income.py``) y confia en que el solve las lleve a su valor. El
monolito, en cambio, las inicializa una por una desde el benchmark
(``get_ytax_stream_init``, monolito 3721).

Para seis corrientes da igual: ``pt``/``fc``/``pc``/``gc``/``ic``/``dt`` acaban
donde deben. Pero ``mt`` (aranceles), ``et`` (impuestos a la exportacion) y
``fs`` (subsidios a factores) quedan sembradas en cero, y su ecuacion NO se
cumple en el punto de arranque: ``eq_ytax[ROW,mt]`` arranca con residual 0.32.

Eso sesga ``ytax_ind = ytaxTot - ytax[dt]`` y, por la identidad ``eq_regy``,
sesga ``regy``: 2.1% en EGY, 0.5-1.2% en el resto.

MEDIDO: arreglarlo NO cambia el match contra GAMS --el barrido de 14 filas da 0
cambios en 42 mediciones--. El solve converge al mismo punto desde ambos seeds.
Este arreglo es de FIDELIDAD, no de match: el benchmark ES un equilibrio y sus
ecuaciones deben cumplirse ANTES de resolver. Un seed que no resuelve su propia
ecuacion funciona hasta que un dataset lo empuja a otra cuenca, que es
exactamente como mordio el bug de ``rsav`` (ver
``test_gtap_blocks_rsav_seed.py``, donde el seed SI movia el resultado).

El test mide sin resolver, que es donde nace el sesgo.
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
# Residual maximo admisible de eq_ytax en el benchmark. El monolito se queda en
# ~5e-8 (ruido de punto flotante al acumular miles de terminos); 1e-6 deja ese
# ruido pasar y atrapa cualquier corriente sembrada en un valor que no resuelve
# su ecuacion (las que faltaban arrancaban entre 5e-7 y 3.2e-1).
ATOL = 1e-6


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


def _residual(con) -> float:
    from pyomo.environ import value

    target = con.lower if con.lower is not None else con.upper
    return abs(float(value(con.body)) - float(value(target)))


def test_eq_ytax_holds_at_benchmark(_block_model):
    """Cada fila de ``eq_ytax`` se cumple en el punto de arranque."""
    eq = _block_model.find_component("eq_ytax")
    assert eq is not None, "eq_ytax ausente del modelo de bloques"

    malas = []
    for idx in eq:
        con = eq[idx]
        if not con.active:
            continue
        resid = _residual(con)
        if resid > ATOL:
            malas.append((resid, idx))

    malas.sort(reverse=True)
    assert not malas, "eq_ytax violada en el benchmark:\n" + "\n".join(
        f"  {str(i):24} residual={r:.6e}" for r, i in malas[:12]
    )


def test_regy_seed_matches_the_income_identity(_block_model):
    """``regy == facty + ytax_ind`` en el seed (la identidad de ``eq_regy``).

    Es la via por la que el sesgo de ``ytax`` llega a ``regy``: si las
    corrientes arrancan bajas, ``regy`` arranca bajo aunque ``facty`` este bien.
    """
    from pyomo.environ import value

    m = _block_model
    for r in m.r:
        regy = float(value(m.regy[r]))
        esperado = float(value(m.facty[r])) + float(value(m.ytax_ind[r]))
        denom = max(abs(esperado), 1e-12)
        assert abs(regy - esperado) / denom <= 1e-9, (
            f"regy[{r}] = {regy!r} pero facty + ytax_ind = {esperado!r}"
        )
