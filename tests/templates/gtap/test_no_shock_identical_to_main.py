"""Criterio 1 del ShockBlock: sin shock, el modelo es el MISMO sistema que antes.

Evalua el residuo de cada fila activa de los 3 periodos (gtap7_3x3, cierre puro)
en un punto perturbado de forma determinista, y lo compara con el mismo calculo
hecho sobre ``main`` 3f3fe5f, anterior al ShockBlock (fixture JSON). Con los
instrumentos en su benchmark (lambdava=1, aft=aft0) cada residuo tiene que dar
igual: una ecuacion reescrita, un coeficiente movido o una fila de mas/de menos
lo rompen.

El punto se perturba (no se evalua en el benchmark) porque ahi casi todo residuo
es ~0 y el test no distinguiria dos formulas distintas.

Regenerar SOLO si se cambia el modelo a proposito:
    python tests/templates/gtap/test_no_shock_identical_to_main.py <repo_root> <out>
"""

from __future__ import annotations

import gzip
import json
import math
import pathlib
import sys
import zlib

ROOT = pathlib.Path(__file__).resolve().parents[3]
GOLDEN = ROOT / "tests/fixtures/gtap7_3x3_residuos_sin_shock_3f3fe5f.json.gz"
REL_TOL = 1e-9
_PINNED = ("dintx", "mintx")


def residuos() -> dict[str, float]:
    """{fila: body - cota} en el punto perturbado; solo filas activas."""
    from pyomo.environ import Constraint, Var, value

    from equilibria.templates.gtap import GTAPParameters
    from equilibria.templates.gtap.gtap_block_model import build_block_model
    from equilibria.templates.gtap.gtap_contract import GTAPClosureConfig

    d = ROOT / "datasets/gtap7_3x3"
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
        if_sub=False,
        numeraire="pnum",
    )
    m, _ = build_block_model(p, p.sets, gc, list(p.sets.r)[-1])
    for v in m.component_data_objects(Var):
        # dintx/mintx estan clavadas a su objetivo de benchmark por su propia fila
        # (eq_dintxeq/eq_mintxeq): fuera de ese objetivo la recaudacion (eq_ytax,
        # que las lee vivas como GAMS) no significa nada. Se dejan en benchmark.
        if v.fixed or v.value is None or v.parent_component().name in _PINNED:
            continue
        # +-1% segun el nombre: determinista y distinto por celda.
        v.set_value(
            v.value * (1.0 + 0.01 * ((zlib.crc32(v.name.encode()) % 7) - 3) / 3)
        )
    out = {}
    for c in m.component_data_objects(Constraint, active=True):
        bound = c.upper if c.upper is not None else c.lower
        out[c.name] = float(value(c.body)) - (
            float(value(bound)) if bound is not None else 0.0
        )
    return out


def test_sin_shock_cada_fila_da_lo_mismo_que_antes_del_shockblock():
    want = json.loads(gzip.decompress(GOLDEN.read_bytes()))
    got = residuos()
    assert set(got) == set(want), (
        f"filas solo ahora: {sorted(set(got) - set(want))[:5]}; "
        f"solo antes: {sorted(set(want) - set(got))[:5]}"
    )
    assert len(got) > 3000, f"comparacion vacua: {len(got)} filas"
    distintas = [
        (k, want[k], got[k])
        for k in want
        if not math.isclose(got[k], want[k], rel_tol=REL_TOL, abs_tol=1e-12)
    ]
    assert not distintas, f"{len(distintas)} filas cambiaron; ej {distintas[:3]}"


if __name__ == "__main__":
    # Generador del fixture: correr con el `src` del checkout de referencia primero.
    ref = pathlib.Path(sys.argv[1]).resolve()
    sys.path.insert(0, str(ref / "src"))
    ROOT = ref
    import equilibria

    assert str(ref / "src") in equilibria.__file__, equilibria.__file__
    pathlib.Path(sys.argv[2]).write_bytes(
        gzip.compress(json.dumps(residuos(), sort_keys=True).encode())
    )
    print("filas:", len(residuos()))
