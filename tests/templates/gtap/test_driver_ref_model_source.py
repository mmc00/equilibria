"""F3, el corte central: el driver no debe construir el MONOLITO.

`solve_multiperiod` construye un `GTAPModelEquations(...).build_model()` en
tres sitios (gtap_multiperiod_driver 3021 base / 3232 check / 3786 shock) y
solo le lee `.fixed`, `lb` y `ub` via `_replicate_sp_fixing` y
`_replicate_sp_bounds`.

En 20x41 eso son 3,4M de celdas construidas para leer tres atributos, y
mientras ocurra el monolito es dependencia de RUNTIME.

Medido (closure altertax) tras d9e6d49, comparando el SP del monolito contra
`build_block_single_period`:

    gtap7_3x3    comunes 1.330   .fixed distinto 0
    gtap7_10x7   comunes 20.145  .fixed distinto 0

Las celdas que solo existen en el monolito (470 y 5.063) son los 30 shifters
muertos, que no aparecen en ninguna ecuacion: no hay nada que replicar.

EL DEFAULT SIGUE SIENDO EL MONOLITO y por eso este test pide bloques
explicitamente con `EQUILIBRIA_GTAP_REF_MODEL=blocks`: bloques emite 33 filas
de mas (eq_mfr_* x30 + eq_mfw_* x3, duplicados de mq_factr_*/mq_factw_*) que
sobredeterminan el sistema, tumban `eq_xseq[USA,VegFruit]` via
_closure_patches.py:439 y bajan el gate MCP de 15x10 pure ifSUB=1 a 87%
(piso 99%).  Esas filas son PREEXISTENTES (blocks/gtap/closure.py en 1fd4492).

Lo que este test fija HOY es que el camino de bloques existe y no toca el
monolito.  Cuando se quiten las filas duplicadas, el default cambia y basta
con borrar el monkeypatch de la variable.
"""

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
import pytest
from test_multiperiod_sets import _load_3x3_params


@pytest.mark.needs_path
def test_solve_multiperiod_does_not_build_the_monolith(monkeypatch):
    """El contrato: cero `GTAPModelEquations` durante un solve completo."""
    from equilibria.templates.gtap import gtap_model_equations as ME
    from equilibria.templates.gtap.altertax import apply_altertax_elasticities
    from equilibria.templates.gtap.gtap_block_model import GTAPBlockMultiPeriodModel
    from equilibria.templates.gtap.gtap_contract import GTAPClosureConfig
    from equilibria.templates.gtap.gtap_multiperiod_driver import solve_multiperiod

    p_alt = apply_altertax_elasticities(_load_3x3_params(), in_place=False)
    rr = list(p_alt.sets.r)[-1]
    gc = GTAPClosureConfig(
        name="altertax",
        closure_type="MCP",
        capital_mobility="mobile",
        fix_endowments=False,
        fix_taxes=True,
        fix_technology=True,
        if_sub=False,
        numeraire="pnum",
    )

    mp = GTAPBlockMultiPeriodModel(p_alt.sets, p_alt, gc, residual_region=rr)
    m = mp.build_sets()
    mp.build_vars(m)
    for t in ("base", "check", "shock"):
        mp.build_equations_intra(m, t)
    mp.build_equations_fisher(m)
    m._residual_region = rr

    built: list[int] = []
    orig = ME.GTAPModelEquations.__init__

    def counting(self, *a, **k):
        built.append(1)
        return orig(self, *a, **k)

    monkeypatch.setenv("EQUILIBRIA_GTAP_REF_MODEL", "blocks")
    monkeypatch.setattr(ME.GTAPModelEquations, "__init__", counting)
    solve_multiperiod(m, p_alt, gc, mode="altertax")

    assert built == [], (
        f"solve_multiperiod construyo {len(built)} monolito(s) con "
        "EQUILIBRIA_GTAP_REF_MODEL=blocks; el modelo de referencia debe salir "
        "de bloques"
    )
