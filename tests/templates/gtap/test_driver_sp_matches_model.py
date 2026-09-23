"""El SP de referencia debe venir de la MISMA fuente que construyo el modelo.

F3 (bd4d2fe) cambio el default de `_build_sp_reference` a bloques.  Un modelo
construido con `GTAPMultiPeriodModel` (el monolito) y resuelto con ese driver
queda con el modelo y el fixing DESALINEADOS.

Medido en gtap7_3x3 pure ifSUB=1: codes {base:1, check:0, shock:0} — check y
shock no convergen.  Y en test_ifsub_primary_block_consistent daba frac_agree
0.00% en los 4 datasets, con el marcador SOLVE code=0.

Siete scripts de `scripts/gtap/` construyen con el monolito y resuelven con el
driver (measure_gate_tols, measure_gtap_pure_tols, measure_nlp_vs_nlp,
measure_nlp_vs_nlp_altertax, diag_mp_3x3, bench_nlp_timing,
gen_linearization_study).  En vez de parchear cada uno, el driver detecta de
que clase salio el modelo y alinea la fuente del SP.
"""

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
import pytest
from test_multiperiod_sets import _load_3x3_params


def _closure(if_sub: bool = True):
    from equilibria.templates.gtap.gtap_contract import GTAPClosureConfig

    return GTAPClosureConfig(
        name="base",
        closure_type="MCP",
        capital_mobility="sluggish",
        fix_endowments=False,
        fix_taxes=False,
        fix_technology=False,
        if_sub=if_sub,
        numeraire="pnum",
    )


@pytest.mark.needs_path
def test_monolith_built_model_solves_with_a_monolith_reference():
    """Modelo del monolito + driver: los tres periodos deben converger.

    Antes del alineado automatico daba {base:1, check:0, shock:0}.
    """
    import contextlib
    import io

    from equilibria.templates.gtap.gtap_model_multiperiod import (
        PERIODS,
        GTAPMultiPeriodModel,
    )
    from equilibria.templates.gtap.gtap_multiperiod_driver import solve_multiperiod

    p = _load_3x3_params()
    rr = list(p.sets.r)[-1]
    gc = _closure()
    ref = ROOT / "tests/fixtures/gtap7/gtap7_3x3/out_gtap_shock_ifsub1.gdx"
    if not ref.exists():
        pytest.skip(f"fixture ausente: {ref}")

    mp = GTAPMultiPeriodModel(p.sets, p, gc, residual_region=rr)
    m = mp.build_sets()
    mp.build_vars(m)
    for per in PERIODS:
        mp.build_equations_intra(m, per)
    mp.build_equations_fisher(m)
    m._residual_region = rr
    mp.seed_all_periods(m, ref)

    with (
        contextlib.redirect_stdout(io.StringIO()),
        contextlib.redirect_stderr(io.StringIO()),
    ):
        res = solve_multiperiod(
            m,
            p,
            gc,
            ref_gdx=ref,
            skip_base_solve=True,
            mute_welfare=True,
            seed_from_prior=False,
            holdfix_cd=True,
            mode="gtap",
        )

    codes = {k: v.get("code") for k, v in res.items()}
    malos = {k: c for k, c in codes.items() if c not in (1, 2)}
    assert not malos, (
        f"periodos sin converger {malos} (todos: {codes}); el SP de referencia "
        "no esta alineado con la clase que construyo el modelo"
    )
