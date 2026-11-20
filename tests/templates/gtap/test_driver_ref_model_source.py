"""F3, el corte central: un run no debe construir el MONOLITO.

El driver viejo construia un `GTAPModelEquations(...).build_model()` en
tres sitios (base / check / shock) y solo le leia `.fixed`, `lb` y `ub` para
replicar el fijado y las cotas del single-period.

En 20x41 eso son 3,4M de celdas construidas para leer tres atributos, y
mientras ocurra el monolito es dependencia de RUNTIME.

Medido (closure altertax) tras d9e6d49, comparando el SP del monolito contra
`build_block_single_period`:

    gtap7_3x3    comunes 1.330   .fixed distinto 0
    gtap7_10x7   comunes 20.145  .fixed distinto 0

Las celdas que solo existen en el monolito (470 y 5.063) son los 30 shifters
muertos, que no aparecen en ninguna ecuacion: no hay nada que replicar.

EL DEFAULT YA ES BLOQUES.  Estuvo bloqueado por dos cosas, las dos medidas y
arregladas:

- 33 filas Fisher duplicadas que solo sobrevivian en la ruta del SP
  (cf2c1b0).
- Los 9 vars de reporte ifSUB se congelaban a "su valor actual", volviendo
  DEFINITIVA la diferencia de init entre las rutas — `pwmg` en 0.001 vs 1.0,
  factor 1000 — con el gate MCP de 15x10 pure ifSUB=1 en 87% (f654f63).
"""

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
import pytest
from test_multiperiod_sets import _load_3x3_params


@pytest.mark.needs_path
def test_core_run_does_not_build_the_monolith(monkeypatch):
    """El contrato: cero `GTAPModelEquations` durante un run completo del core
    (altertax, base, check y shock)."""
    from tests.templates.gtap._routes import solve

    from equilibria.templates.gtap import gtap_model_equations as ME
    from equilibria.templates.gtap.gtap_contract import GTAPClosureConfig

    p = _load_3x3_params()
    rr = list(p.sets.r)[-1]
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

    built: list[int] = []
    orig = ME.GTAPModelEquations.__init__

    def counting(self, *a, **k):
        built.append(1)
        return orig(self, *a, **k)

    monkeypatch.setattr(ME.GTAPModelEquations, "__init__", counting)
    res = solve("core", p, gc, rr, mode="altertax", skip_base_solve=True)

    assert set(res.codes) == {"base", "check", "shock"}
    assert built == [], (
        f"el run construyo {len(built)} monolito(s); el modelo de referencia "
        "sale de bloques"
    )
