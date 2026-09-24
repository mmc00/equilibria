"""El SP de referencia no debe traer los agregados auxiliares duplicados.

`build_block_single_period` declara eq_mfr_*/eq_mfw_* (30+3 filas en
gtap7_15x10): los agregados con que el bloque CLOSURE parte la suma ancha de
su eq_pfact/eq_pwfact intra-periodo (closure.py:216-341).  El monolito
inlinea esas sumas y no los declara.

La maquinaria MULTIPERIODO ya los borra —gtap_model_multiperiod
build_equations_fisher, 733-758, con el diagnostico completo escrito ahi— y
por eso el modelo multiperiodo de bloques esta bien.  Pero el driver pide un
SP (`_build_sp_reference`), que nunca pasa por ese borrado.

Por que importa: el driver solo le lee `.fixed`/`lb`/`ub`, pero esas 33 filas
sobredeterminan el sistema, `_closure_patches.py:439` lo cuadra desactivando
una ecuacion REAL —medido: `eq_xseq[USA,VegFruit]`, el balance fisico
xs == xds + xet— y el gate MCP de 15x10 pure ifSUB=1 cae de >=99% a 87,00%.
"""

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
import pytest

DATASETS = ROOT / "datasets"
# Esta lista se mantiene APARTE a proposito, sin importar FISHER_AUX_EQS de
# blocks/gtap/closure.py: si el test importara la misma constante que usa el
# codigo, comprobaria que el codigo es coherente consigo mismo (tautologia) y
# un renombrado en bloque pasaria verde.  Escrita a mano, un renombrado rompe
# el test, que es lo que se quiere.
_DUP_EQS = (
    "eq_mfr_bs",
    "eq_mfr_sb",
    "eq_mfr_ss",
    "eq_mfw_bs",
    "eq_mfw_sb",
    "eq_mfw_ss",
)
_DUP_VARS = ("mfr_bs", "mfr_sb", "mfr_ss", "mfw_bs", "mfw_sb", "mfw_ss")


@pytest.mark.parametrize("dataset", ["gtap7_3x3", "gtap7_15x10"])
@pytest.mark.parametrize("if_sub", [False, True])
def test_sp_reference_has_no_duplicate_fisher_aggregates(dataset, if_sub):
    from equilibria.templates.gtap import GTAPParameters
    from equilibria.templates.gtap.gtap_contract import GTAPClosureConfig
    from equilibria.templates.gtap.gtap_multiperiod_driver import _build_sp_reference

    d = DATASETS / dataset
    p = GTAPParameters()
    p.load_from_har(
        basedata_path=d / "basedata.har",
        sets_path=d / "sets.har",
        default_path=d / "default.prm",
        baserate_path=d / "baserate.har",
    )
    rr = list(p.sets.r)[-1]
    gc = GTAPClosureConfig(
        name="base",
        closure_type="MCP",
        capital_mobility="sluggish",
        fix_endowments=False,
        fix_taxes=False,
        fix_technology=False,
        if_sub=if_sub,
        numeraire="pnum",
    )

    import os

    os.environ["EQUILIBRIA_GTAP_REF_MODEL"] = "blocks"
    try:
        sp = _build_sp_reference(p.sets, p, gc, rr)
    finally:
        os.environ.pop("EQUILIBRIA_GTAP_REF_MODEL", None)

    sobran = [n for n in _DUP_EQS if getattr(sp, n, None) is not None]
    assert not sobran, (
        f"[{dataset}/ifSUB={int(if_sub)}] el SP de referencia trae los agregados "
        f"duplicados {sobran}; sobredeterminan el sistema y el squaring sacrifica "
        "una ecuacion real (eq_xseq)"
    )

    # Borrar la fila sin borrar la Var deja una COLUMNA HUERFANA, que cambia el
    # sistema igual que la fila (mismo razonamiento que multiperiodo 752-758).
    huerfanas = [n for n in _DUP_VARS if getattr(sp, n, None) is not None]
    assert not huerfanas, (
        f"[{dataset}/ifSUB={int(if_sub)}] quedaron columnas huerfanas {huerfanas}: "
        "hay que borrar la Var, no solo la ecuacion"
    )
