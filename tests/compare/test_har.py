"""from_har: el .har de NIVELES de GEMPACK (issue #131). Oraculo: ejemplo 12."""

from __future__ import annotations

import numpy as np
import pytest

from equilibria.babel.har.symbols import HeaderArray
from equilibria.babel.har.writer import write_har
from equilibria.compare import CompareError, Mapping, compare, from_dict, from_har
from tests.compare import _oracle as O

ACTS = ["Food", "Mnfcs"]


def _var(hid: str, long_name: str, values: list[float]) -> HeaderArray:
    return HeaderArray(
        name=hid,
        coeff_name=long_name.split("#")[0].strip(),
        long_name=long_name,
        array=np.array(values, dtype=np.float32).reshape(2, 1, order="F"),
        set_names=["ACTS", "REG"],
        set_elements=[list(ACTS), ["USA"]],
    )


def _nolv(names: list[str]) -> HeaderArray:
    return HeaderArray(
        name="NOLV",
        coeff_name="NOLV",
        long_name="variables with NO level in the SL4 (sentinel 1e10): no LEVA/LEVB",
        array=np.array(names, dtype=object),
        set_names=[],
        set_elements=[],
    )


def _write(tmp_path, headers: dict[str, HeaderArray], name: str = "lev.har"):
    path = tmp_path / name
    write_har(str(path), headers)
    return path


@pytest.fixture
def levels_har(tmp_path):
    return _write(
        tmp_path,
        {
            "0001": _var("0001", "qo # output of activity #", [10.0, 20.0]),
            "B001": _var("B001", "LEVB qo # output of activity #", [8.0, 16.0]),
            "NOLV": _nolv(["psave"]),
        },
    )


def test_lee_pre_de_levb_y_post_de_leva(levels_har):
    sol = from_har(levels_har)
    assert sol.periods() == O.EX12_PERIODS
    assert sol.levels["qo"] == O.EX12_QO


def test_solo_leva_da_solo_post(tmp_path):
    path = _write(
        tmp_path, {"0001": _var("0001", "qo # output of activity #", [10.0, 20.0])}
    )
    sol = from_har(path)
    assert sol.periods() == O.EX12_LEVA_ONLY_PERIODS
    assert sol.levels["qo"] == {k: v for k, v in O.EX12_QO.items() if k[0] == "post"}


def _psave_map() -> Mapping:
    return Mapping.from_dict(
        {
            "variables": [
                {
                    "name": "saving_price",
                    "a": {"var": "psave", "dims": ["ACTS", "REG"]},
                    "b": {"var": "psave", "dims": ["ACTS", "REG"]},
                    "tol": {"rel": 1e-3, "abs_floor": 1e-6},
                }
            ],
            "thresholds": [1e-3],
        }
    )


def _a_psave():
    return from_dict({"psave": {("Food", "USA"): 1.0}}, period="post")


def test_una_variable_de_nolv_queda_excluida_y_declarada(levels_har):
    r = compare(_a_psave(), from_har(levels_har), _psave_map())
    assert "saving_price" not in r.summary
    assert O.EX12_NO_LEVEL_WHY in r.excluded["saving_price"]


def test_har_viejo_con_centinela_en_toda_la_variable_tambien(tmp_path):
    path = _write(
        tmp_path,
        {
            "0001": _var("0001", "qo # output of activity #", [10.0, 20.0]),
            "0002": _var("0002", "psave # price of savings #", [1e10, 1e10]),
        },
    )
    r = compare(_a_psave(), from_har(path), _psave_map())
    assert O.EX12_NO_LEVEL_WHY in r.excluded["saving_price"]


def test_centinela_en_solo_algunas_celdas_es_error(tmp_path):
    path = _write(
        tmp_path, {"0002": _var("0002", "psave # price of savings #", [1e10, 3.0])}
    )
    with pytest.raises(CompareError, match="psave"):
        from_har(path)


def test_un_har_de_cambios_pct_es_error_con_la_instruccion(tmp_path):
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
    path = _write(tmp_path, headers)
    with pytest.raises(CompareError, match=O.EX12_CONVERSION_HINT):
        from_har(path)


def test_una_variable_pedida_que_no_esta_es_error(levels_har):
    with pytest.raises(CompareError, match="qxs"):
        from_har(levels_har, variables=["qxs"])


def test_un_escalar_sin_conjuntos_tiene_indice_vacio(tmp_path):
    path = _write(
        tmp_path,
        {
            "0049": HeaderArray(
                name="0049",
                coeff_name="globalcgds",
                long_name="globalcgds # global supply of capital goods #",
                array=np.array([7.5], dtype=np.float32),
                set_names=[],
                set_elements=[],
            )
        },
    )
    sol = from_har(path)
    assert sol.levels["globalcgds"] == {("post", ()): 7.5}
    assert sol.ndim["globalcgds"] == 0


def test_una_sola_cabecera_de_cambios_pct_basta_para_rechazarlo(tmp_path):
    """Un .har de cambios % nunca se lee como niveles, aunque le falten cabeceras."""
    headers = {
        "0001": _var("0001", "qo # output of activity #", [3.9, 0.1]),
        "XXCR": HeaderArray(
            name="XXCR",
            coeff_name="XXCR",
            long_name="XXCR",
            array=np.array([1.0], dtype=np.float32),
            set_names=[""],
            set_elements=[["x"]],
        ),
    }
    with pytest.raises(CompareError, match=O.EX12_CONVERSION_HINT):
        from_har(_write(tmp_path, headers))


def test_el_centinela_en_pre_nunca_entra_como_nivel(tmp_path):
    path = _write(
        tmp_path,
        {
            "0001": _var("0001", "qo # output of activity #", [10.0, 20.0]),
            "B001": _var("B001", "LEVB qo # output of activity #", [1e10, 1e10]),
            "NOLV": _nolv([]),
        },
    )
    with pytest.raises(CompareError, match="qo"):
        from_har(path)


def test_un_har_nuevo_sin_el_levb_de_una_variable_es_error(tmp_path):
    """Con NOLV el archivo sigue la regla de #127: cada LEVA tiene su LEVB."""
    path = _write(
        tmp_path,
        {
            "0001": _var("0001", "qo # output of activity #", [10.0, 20.0]),
            "NOLV": _nolv([]),
        },
    )
    with pytest.raises(CompareError, match="B001"):
        from_har(path)


def test_un_levb_con_otro_long_name_es_error(tmp_path):
    path = _write(
        tmp_path,
        {
            "0001": _var("0001", "qo # output of activity #", [10.0, 20.0]),
            "B001": _var("B001", "LEVB qfd # otra variable #", [8.0, 16.0]),
            "NOLV": _nolv([]),
        },
    )
    with pytest.raises(CompareError, match="B001"):
        from_har(path)


def test_una_cabecera_leva_no_numerica_es_error(tmp_path):
    path = _write(
        tmp_path,
        {
            "0001": _var("0001", "qo # output of activity #", [10.0, 20.0]),
            "0002": HeaderArray(
                name="0002",
                coeff_name="txt",
                long_name="txt # texto #",
                array=np.array(["a", "b"], dtype=object),
                set_names=[],
                set_elements=[],
            ),
        },
    )
    with pytest.raises(CompareError, match="0002"):
        from_har(path)
