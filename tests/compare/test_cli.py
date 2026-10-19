"""La CLI ``equilibria compare A B --map MAPA`` (issue #130), contra los oraculos 3, 5 y 11."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import yaml
from click.testing import CliRunner

from equilibria.cli import main
from tests.compare import _oracle as O


def _levels_json(path: Path, data: dict) -> Path:
    """Escribe ``{var: {(indices..., periodo): nivel}}`` como el JSON de niveles de
    la CLI: ``{var: [[[indices..., periodo], nivel], ...]}``."""
    path.write_text(
        json.dumps(
            {v: [[list(k), x] for k, x in cells.items()] for v, cells in data.items()}
        )
    )
    return path


def _yaml(path: Path, mapping: dict) -> Path:
    path.write_text(yaml.safe_dump(mapping))
    return path


def _run(*args: str):
    return CliRunner().invoke(main, ["compare", *args], catch_exceptions=False)


needs_gdx = pytest.mark.skipif(
    not O.GTAP3X3_GDX.exists(), reason="falta el GDX de gtap7_3x3"
)


@pytest.fixture
def ex5(tmp_path):
    b = _levels_json(tmp_path / "b.json", O.EX5_B)
    m = _yaml(tmp_path / "map.yaml", O.EX5_MAP)
    return [str(O.GTAP3X3_GDX), str(b), "--map", str(m), "--period-dim", "-1"]


@pytest.fixture
def ex3(tmp_path):
    a = _levels_json(tmp_path / "a.json", O.EX3_A)
    b = _levels_json(tmp_path / "b.json", O.EX3_B)
    m = _yaml(tmp_path / "map.yaml", O.EX3_MAP)
    return [str(a), str(b), "--map", str(m), "--period-dim", "-1"]


@needs_gdx
def test_imprime_el_resumen_por_variable_y_global(ex5):
    r = _run(*ex5)
    assert r.exit_code == 0, (
        r.output
    )  # sin --fail-below no falla aunque una celda falle
    lineas = r.output.splitlines()
    var = next(ln for ln in lineas if ln.split()[:1] == ["output"])
    glob = next(ln for ln in lineas if ln.split()[:1] == ["TOTAL"])
    for ln in (var, glob):
        assert str(O.EX5_MATCHED) in ln.split()
        assert O.EX11_EX5_PASS_PCT_TEXT in ln
    assert "USA.Food" in var  # la celda del maximo
    assert "pft" in r.output  # la variable excluida, declarada con su motivo
    assert "ejemplo: excluida a proposito" in r.output


@needs_gdx
def test_out_escribe_el_json_del_reporte(ex5, tmp_path):
    out = tmp_path / "reporte.json"
    r = _run(*ex5, "--out", str(out))
    assert r.exit_code == 0, r.output
    rep = json.loads(out.read_text())
    s = rep["summary"]["output"]
    assert s["cells"] == O.EX5_MATCHED
    assert s["max"] == pytest.approx(O.EX5_MAX, rel=1e-9)
    assert s["max_cell"] == {
        "period": O.EX5_MAX_CELL[0],
        "index": list(O.EX5_MAX_CELL[1]),
    }
    for thr, pct in O.EX5_PCT_WITHIN.items():
        assert s["pct_within"][repr(thr)] == pytest.approx(pct)
    assert rep["periods_only_in_a"] == O.EX5_PERIODS_ONLY_IN_A
    assert len(rep["cells"]) == O.EX5_MATCHED


@needs_gdx
@pytest.mark.parametrize(
    ("fail_below", "exit_code"), [("94", 0), ("95", O.EX11_FAIL_EXIT)]
)
def test_fail_below_controla_el_codigo_de_salida(ex5, fail_below, exit_code):
    """17/18 = 94.44% (oraculo 11): pasa con 94, falla con 95."""
    r = _run(*ex5, "--fail-below", fail_below)
    assert r.exit_code == exit_code, r.output


@pytest.mark.parametrize(
    ("fail_below", "exit_code"), [("50", 0), ("60", O.EX11_FAIL_EXIT)]
)
def test_fail_below_cuenta_las_celdas_sin_pareja_como_que_no_pasan(
    ex3, fail_below, exit_code
):
    """2 parejas que pasan + 2 sin pareja = 50%; sobre las parejas solas seria 100%."""
    r = _run(*ex3, "--fail-below", fail_below)
    assert r.exit_code == exit_code, r.output


def test_la_fila_total_muestra_el_mismo_pct_que_usa_fail_below(ex3):
    r = _run(*ex3)
    total = next(ln for ln in r.output.splitlines() if ln.split()[:1] == ["TOTAL"])
    assert O.EX11_EX3_PASS_PCT_ALL_TEXT in total.split()


def test_un_sl4_es_error_con_la_instruccion_de_conversion(ex3, tmp_path):
    sl4 = tmp_path / "sol.sl4"
    sl4.write_bytes(b"")
    args = [str(sl4), *ex3[1:]]
    r = _run(*args)
    assert r.exit_code == O.EX11_ERROR_EXIT
    assert "sltoht" in r.output


def test_sin_periodo_declarado_es_error_no_se_adivina(tmp_path):
    a = _levels_json(tmp_path / "a.json", O.EX3_A)
    b = _levels_json(tmp_path / "b.json", O.EX3_B)
    m = _yaml(tmp_path / "map.yaml", O.EX3_MAP)
    r = _run(str(a), str(b), "--map", str(m))
    assert r.exit_code == O.EX11_ERROR_EXIT
    assert "period" in r.output


def test_el_comando_equilibria_esta_instalado():
    from importlib.metadata import entry_points

    eps = {ep.name: ep.value for ep in entry_points(group="console_scripts")}
    assert eps.get("equilibria") == "equilibria.cli:main"


def _levels_har(tmp_path):
    """El .har de niveles del ejemplo 12: qo pre (LEVB) y post (LEVA), psave en NOLV."""
    from tests.compare.test_har import _nolv, _var, _write

    return _write(
        tmp_path,
        {
            "0001": _var("0001", "qo # output of activity #", [10.0, 20.0]),
            "B001": _var("B001", "LEVB qo # output of activity #", [8.0, 16.0]),
            "NOLV": _nolv(["psave"]),
        },
    )


QO_MAP = {
    "variables": [
        {
            "name": "output",
            "a": {"var": "qo", "dims": ["ACTS", "REG"]},
            "b": {"var": "qo", "dims": ["ACTS", "REG"]},
            "tol": {"rel": 1e-3, "abs_floor": 1e-6},
        }
    ],
    "thresholds": [1e-3],
}


def test_lee_un_har_de_niveles_de_gempack(tmp_path):
    """Ejemplo 12: A (JSON) trae los mismos niveles que el .har -> 4 celdas, 100 %."""
    a = _levels_json(
        tmp_path / "a.json",
        {"qo": {(*idx, period): v for (period, idx), v in O.EX12_QO.items()}},
    )
    m = _yaml(tmp_path / "map.yaml", QO_MAP)
    har = _levels_har(tmp_path)
    r = _run(str(a), str(har), "--map", str(m), "--a-period-dim", "-1")
    assert r.exit_code == 0, r.output
    total = next(ln for ln in r.output.splitlines() if ln.split()[:1] == ["TOTAL"])
    assert total.split()[1:3] == [str(len(O.EX12_QO)), "100.00"]


def test_un_har_de_cambios_pct_es_error_con_la_instruccion(tmp_path):
    from equilibria.babel.har.symbols import HeaderArray
    from tests.compare.test_har import _var, _write

    headers = {"0001": _var("0001", "qo # output of activity #", [3.9, 0.1])}
    for hid in ("XXCR", "XXCD", "XXCP", "XXTD", "XXTT"):
        headers[hid] = HeaderArray(
            name=hid,
            coeff_name=hid,
            long_name=hid,
            array=np.array([1.0], dtype=np.float32),
            set_names=[""],
            set_elements=[["x"]],
        )
    har = _write(tmp_path, headers, "pct.har")
    a = _levels_json(tmp_path / "a.json", {"qo": {("Food", "USA", "post"): 1.0}})
    m = _yaml(tmp_path / "map.yaml", QO_MAP)
    r = _run(str(a), str(har), "--map", str(m), "--a-period-dim", "-1")
    assert r.exit_code == O.EX11_ERROR_EXIT
    assert O.EX12_CONVERSION_HINT in r.output


def test_un_divide_by_con_parametro_en_una_fuente_json_es_error(tmp_path):
    """Una fuente .json no trae parametros: un divide_by con param en ese lado es
    error de datos (2), con el nombre del parametro."""
    a = _levels_json(tmp_path / "a.json", O.EX13_A)
    b = _levels_json(tmp_path / "b.json", O.EX13_B)
    m = _yaml(tmp_path / "map.yaml", O.EX13_MAP)
    r = _run(str(a), str(b), "--map", str(m), "--period-dim", "-1")
    assert r.exit_code == O.EX11_ERROR_EXIT
    assert "xscale" in r.output


def test_flags_de_periodo_para_un_har_son_error_no_se_ignoran(tmp_path):
    a = _levels_json(
        tmp_path / "a.json",
        {"qo": {(*idx, period): v for (period, idx), v in O.EX12_QO.items()}},
    )
    m = _yaml(tmp_path / "map.yaml", QO_MAP)
    har = _levels_har(tmp_path)
    r = _run(
        str(a), str(har), "--map", str(m), "--a-period-dim", "-1", "--b-period", "x"
    )
    assert r.exit_code == O.EX11_ERROR_EXIT
    assert ".har" in r.output
