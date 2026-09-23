"""Las celdas sin demanda (`tmg`) deben quedar FIJADAS tambien en bloques.

El monolito las fija; bloques no.  Medido en gtap7_3x3 (closure altertax):
de las 482 variables que el monolito fija, 470 son familias que bloques NO
DECLARA (los 30 shifters muertos: dtxshft, afeall, lambdaf... que no aparecen
en ninguna ecuacion).  Quedan 12 que SI existen en ambos y solo el monolito
fija: `pa` y `xaa` en el sector `tmg`, con el MISMO valor en los dos (1 y 0).
Lo que falta en bloques es el flag.

Importa porque `gtap_multiperiod_driver` construye un GTAPModelEquations
entero — 3,4M de celdas en 20x41 — en tres sitios (3021/3232/3786) SOLO para
leerle `.fixed`/`lb`/`ub` y replicarlo.  Mientras bloques no reproduzca ese
fixing, el monolito sigue siendo dependencia de runtime y F3 no cierra.

Dos comentarios del repo afirman que "the composer applies the .fix"
(blocks/gtap/__init__.py:33 y trade_armington_bilateral.py:155).  MEDIDO:
no lo hace — `build_block_model` deja 1 sola celda fijada en todo el modelo
(`pwfactbase`, el numerario) y 0 de las 12.
"""

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
import pytest

DATASETS = ROOT / "datasets"


def _params(dataset: str):
    from equilibria.templates.gtap import GTAPParameters

    d = DATASETS / dataset
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
        name="altertax",
        closure_type="MCP",
        capital_mobility="mobile",
        fix_endowments=False,
        fix_taxes=True,
        fix_technology=True,
        if_sub=False,
        numeraire="pnum",
    )


@pytest.mark.parametrize("dataset", ["gtap7_3x3", "gtap7_5x5"])
def test_blocks_fix_the_same_cells_the_monolith_fixes(dataset):
    """Toda celda que el monolito fija y que bloques TAMBIEN declara debe
    quedar fijada en bloques, al mismo valor.

    Las que bloques no declara quedan fuera a proposito: son variables
    muertas (0 apariciones en las 3.390 restricciones), no hay nada que fijar.
    """
    from pyomo.environ import Var
    from pyomo.environ import value as V

    from equilibria.templates.gtap import GTAPModelEquations
    from equilibria.templates.gtap.gtap_block_model import build_block_single_period

    p = _params(dataset)
    rr = list(p.sets.r)[-1]
    gc = _closure()

    mono = GTAPModelEquations(p.sets, p, gc, residual_region=rr).build_model()
    blk = build_block_single_period(p, p.sets, gc, rr)

    bcells = {
        (v.name, k): v[k] for v in blk.component_objects(Var, active=True) for k in v
    }

    faltan = []
    for v in mono.component_objects(Var, active=True):
        for k, vd in v.items():
            if not vd.fixed:
                continue
            bv = bcells.get((v.name, k))
            if bv is None:
                continue  # bloques no la declara: variable muerta
            if not bv.fixed:
                faltan.append((v.name, k, float(V(vd)), float(V(bv))))

    assert not faltan, (
        f"[{dataset}] {len(faltan)} celdas fijadas en el monolito y LIBRES en "
        f"bloques; ej {faltan[:4]}"
    )
