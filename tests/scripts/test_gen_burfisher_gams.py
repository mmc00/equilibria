"""gen_burfisher_gams.py: el oraculo GAMS de los ejercicios de Burfisher, sin GAMS.

Lo que se puede probar sin correr GAMS: que cada ejercicio produce su .inc, que los
parches a las fuentes de referencia (getData.gms, model.gms) siguen aplicando sobre las
fuentes del repo y que son idempotentes. Si alguien cambia model.gms o getData.gms y un
parche deja de aplicar, falla aca y no en medio de una corrida de GAMS.
"""

from __future__ import annotations

import importlib.util
import shutil
import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "gtap" / "gen_burfisher_gams.py"
GAMS_SRC = ROOT / "src" / "equilibria" / "templates" / "reference" / "gtap" / "scripts"


def _load_module() -> Any:
    spec = importlib.util.spec_from_file_location("gen_burfisher_gams", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # el dataclass Run lo busca ahi
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def gen() -> Any:
    return _load_module()


def test_cubre_los_41_ejercicios(gen: Any) -> None:
    """Los 37 de run_burfisher.EXERCISES y los 4 de cierre propio (desempleo, QCA, PIB)."""
    todos = gen.all_exercises()
    assert len(todos) == 41
    assert len(set(todos)) == 41
    for exp in ("TBL65B", "ME3C", "TBL94", "ME9A"):
        assert exp in todos
    # ME9D arranca desde el shock de ME9C: tiene que ir despues.
    assert todos.index("ME9C") < todos.index("ME9D")


def test_cada_ejercicio_produce_su_inc(gen: Any, tmp_path: Path) -> None:
    (tmp_path / "ME9C_capFlex.gdx").touch()  # el arranque de ME9D
    for exp in gen.all_exercises():
        prm, texto = gen.shock_inc(exp, tmp_path, defl="tornq")
        assert prm.endswith(".prm"), exp
        assert texto.startswith(f"* {exp} ({prm})"), exp
        assert texto.count("\n") >= 2, f"{exp}: sin sentencias de shock"


def test_imptx_es_un_shock_a_la_potencia(gen: Any) -> None:
    linea = gen.gams_line("imptx", ("ROW", "MFG", "USA"), "power", 10.0)
    assert linea == (
        "imptx.fx('ROW','c_MFG','USA',tsim) = "
        "(1 + imptx.l('ROW','c_MFG','USA',tsim))*1.1 - 1 ;"
    )


def test_instrumento_desconocido_falla(gen: Any) -> None:
    with pytest.raises(ValueError):
        gen.gams_line("noexiste", ("USA",), "pct", 1.0)


def test_me9d_sin_me9c_falla(gen: Any, tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="ME9C"):
        gen.shock_inc("ME9D", tmp_path, defl="tornq")


def test_parches_aplican_sobre_las_fuentes_del_repo(gen: Any, tmp_path: Path) -> None:
    shutil.copytree(GAMS_SRC, tmp_path, dirs_exist_ok=True)
    gen.patch_gams_sources(tmp_path, defl="tornq")
    getdata = (tmp_path / "getData.gms").read_text()
    model = (tmp_path / "model.gms").read_text()
    assert "[F-val] factor con ETRE != 0" in getdata
    assert "wrFlag(r,fm)" in model
    assert "qcaeq.prdtx, gdpeq.axpreg" in model

    # Idempotente: correr dos veces en el mismo directorio no duplica nada.
    gen.patch_gams_sources(tmp_path, defl="tornq")
    assert (tmp_path / "getData.gms").read_text() == getdata
    assert (tmp_path / "model.gms").read_text() == model


def test_deflactor_desconocido_falla(gen: Any, tmp_path: Path) -> None:
    with pytest.raises(KeyError):
        gen.shock_inc("TBL65B", tmp_path, defl="noexiste")


@pytest.mark.parametrize("kind", ["pct", "power", "power_kappa", "power_fct"])
def test_formula_gams_es_la_de_equilibria(gen: Any, kind: str) -> None:
    """El texto GAMS evaluado da lo mismo que run_burfisher.shocked con numeros."""
    f, chk, fs = 1.1, 0.13, 0.02
    texto = str(gen.shocked(kind, gen.GamsExpr("C"), f, gen.GamsExpr("F")))
    assert eval(texto, {"C": chk, "F": fs}) == pytest.approx(
        gen.shocked(kind, chk, f, fs), rel=1e-15
    )


def test_tasa_con_kind_distinto_de_pct_falla(gen: Any) -> None:
    with pytest.raises(ValueError, match="lambdava"):
        gen.gams_line("lambdava", ("USA", "AGR"), "power", 1.0)


def test_warm_vars_estan_en_model_gms(gen: Any) -> None:
    """execute_loadpoint carga WARM_VARS: tienen que ser variables del modelo."""
    import re

    model = (GAMS_SRC / "model.gms").read_text()
    faltan = [v for v in gen.WARM_VARS if not re.search(rf"\b{v}\(", model)]
    assert faltan == []


def test_faltantes_del_dataset(gen: Any, tmp_path: Path) -> None:
    for f in ("basedata.har", "sets.har", "baserate.har", "default.prm"):
        (tmp_path / f).touch()
    faltan = gen.missing_inputs(tmp_path, ["TBL45A", "ME9A"])
    assert faltan == [tmp_path / "climatechange.prm"]
