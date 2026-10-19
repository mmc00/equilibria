"""from_har con un .sl4: los niveles se leen directo del .sl4 (issue #148).

Oraculo: ejemplo 15. Los esperados salen de archivos que escribio sltoht, nunca del
lector nuevo.
"""

from __future__ import annotations

import numpy as np
import pytest

from equilibria.babel.har.reader import read_har
from equilibria.compare import CompareError, from_har
from tests.compare import _oracle as O


def _by_variable(path) -> dict:
    """{variable: HeaderArray} de un HAR de sltoht, por el nombre del long_name."""
    out = {}
    for ha in read_har(str(path)).values():
        long_name = ha.long_name or ""
        if "#" in long_name and not long_name.startswith("LEVB"):
            out[long_name.split("#", 1)[0].strip()] = ha
    return out


def _cells(ha) -> dict[tuple[str, ...], float]:
    arr = np.asarray(ha.array, dtype=np.float64)
    if not ha.set_elements:
        return {(): float(arr.ravel()[0])}
    return {
        tuple(str(ha.set_elements[d][i]) for d, i in enumerate(pos)): float(arr[pos])
        for pos in np.ndindex(arr.shape)
    }


def _oracle_pre() -> tuple[dict[str, dict], set[str]]:
    """Oraculo 1: pre = post / (1 + %/100) de los fixtures de sltoht del nus333."""
    post, pct = _by_variable(O.EX15_NUS333_POST), _by_variable(O.EX15_NUS333_PCT)
    pre, sentinel = {}, set()
    for name, ha in post.items():
        cells = _cells(ha)
        if all(v >= 1e10 for v in cells.values()):
            sentinel.add(name)
            continue
        change = _cells(pct[name])
        pre[name] = {k: v / (1 + change[k] / 100) for k, v in cells.items()}
    return pre, sentinel


@pytest.fixture(scope="module")
def tbl45a():
    return from_har(O.EX15_SL4["TBL45A"])


def test_qo_pre_es_el_del_oraculo(tbl45a):
    got = {
        idx: v for (period, idx), v in tbl45a.levels["qo"].items() if period == "pre"
    }
    assert got.keys() == O.EX15_QO_PRE.keys()
    for idx, want in O.EX15_QO_PRE.items():
        assert got[idx] == pytest.approx(want, rel=O.EX15_PRE_RTOL), idx


def test_el_pre_de_cada_variable_es_el_del_oraculo(tbl45a):
    pre, _ = _oracle_pre()
    assert len(pre) == O.EX15_N_WITH_LEVEL
    for name, cells in pre.items():
        assert name in tbl45a.levels, name
        got = {i: v for (p, i), v in tbl45a.levels[name].items() if p == "pre"}
        assert got.keys() == cells.keys(), name
        for idx, want in cells.items():
            assert got[idx] == pytest.approx(want, rel=O.EX15_PRE_RTOL), (name, idx)


def test_las_variables_con_nivel_son_las_del_oraculo(tbl45a):
    pre, _ = _oracle_pre()
    assert set(tbl45a.levels) == set(pre)


def test_lo_que_sltoht_marca_sin_nivel_queda_declarado(tbl45a):
    _, sentinel = _oracle_pre()
    assert sentinel
    for name in sentinel:
        assert name in tbl45a.no_level, name
        assert O.EX15_NO_LEVEL_WHY in tbl45a.no_level[name]


@pytest.mark.parametrize("exp", sorted(O.EX15_SL4))
def test_el_cambio_de_cada_celda_es_el_de_sltoht(exp):
    sol = from_har(O.EX15_SL4[exp])
    dump = _by_variable(O.EX15_DUMP[exp])
    checked = 0
    for name, cells in sol.levels.items():
        pct = _cells(dump[name])
        pre = {i: v for (p, i), v in cells.items() if p == "pre"}
        post = {i: v for (p, i), v in cells.items() if p == "post"}
        assert pre.keys() == post.keys() == pct.keys(), name
        for idx, b in pre.items():
            if b == 0.0:  # flujo cero: sin %, en niveles 0 -> 0
                assert post[idx] == 0.0, (name, idx)
                continue
            assert 100 * (post[idx] / b - 1) == pytest.approx(
                pct[idx], abs=O.EX15_PCT_ATOL
            ), (name, idx)
            checked += 1
    assert checked > 0


def test_variables_limita_la_lectura():
    sol = from_har(O.EX15_SL4["TBL45A"], variables=["qo"])
    assert set(sol.levels) == {"qo"}
    assert sol.ndim["qo"] == 2


def test_una_variable_pedida_que_no_esta_es_error():
    with pytest.raises(CompareError, match="no_existe"):
        from_har(O.EX15_SL4["TBL45A"], variables=["no_existe"])


def test_un_sl4_ilegible_es_error(tmp_path):
    path = tmp_path / "sol.sl4"
    path.write_bytes(b"\0")
    with pytest.raises(CompareError, match="sol.sl4"):
        from_har(path)
