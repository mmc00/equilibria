"""El .sl4 de GEMPACK: headers 2RFULL y el offset real de los datos.

Fija dos cosas que el par lector/escritor interno no probaba contra un
archivo real de GEMPACK:

  - 2RFULL (matriz real densa 2-D) no estaba implementado, asi que todo
    .sl4 fallaba al abrirse.
  - 2RFULL y 2IFULL ponen los datos en el offset 32, detras de PAD(4) + 7
    int32 de geometria. El lector leia desde el 8, asi que arrastraba seis
    enteros de dimensiones como datos y perdia los ultimos seis valores.
    El escritor emitia el prefijo corto, de modo que leer-escribir-leer
    cerraba y el error no se veia.

El oraculo es un .sl4 producido por GEMPACK 11.3 / sltoht 5.53.
"""

from __future__ import annotations

import numpy as np
import pytest

from equilibria.babel.har import read_har

SL4 = (
    __import__("pathlib").Path(__file__).resolve().parents[3]
    / "tools/gempack-oracle/out/TBL45A/TBL45A.sl4"
)

# Los tests que leen el .sl4 necesitan la corrida de GEMPACK, que no esta en
# el repo. El round-trip del escritor NO la necesita y debe correr siempre:
# es el unico guardia del offset de 2IFULL, que corrompia cualquier HAR.
needs_sl4 = pytest.mark.skipif(
    not SL4.exists(),
    reason="falta el .sl4 de la corrida GEMPACK (tools/gempack-oracle/out/)",
)


@needs_sl4
def test_sl4_opens() -> None:
    """Antes de implementar 2RFULL esto tiraba NotImplementedError."""
    d = read_har(SL4)
    assert len(d) > 60
    for k in ("CUMS", "LEVB", "LEVA", "UVAL", "SHOC"):
        assert k in d, f"falta el header {k}"


@needs_sl4
def test_2rfull_scalar_value() -> None:
    """UVAL es 1x1 y vale 0.1 (el parametro U de Harwell).

    Leido desde el offset equivocado da 1.4e-45, el denormal de interpretar
    un entero chico como float: un valor plausible que no salta a la vista.
    """
    d = read_har(SL4)
    assert d["UVAL"].array.shape == (1, 1)
    assert float(d["UVAL"].array.ravel()[0]) == pytest.approx(0.1, abs=1e-7)


@needs_sl4
def test_shock_value_is_the_experiment() -> None:
    """SHOC guarda el shock aplicado: TBL45A es avaall("SER","USA") = 10."""
    d = read_har(SL4)
    assert float(d["SHOC"].array.ravel()[0]) == pytest.approx(10.0)


@needs_sl4
def test_2ifull_pointers_are_coherent() -> None:
    """PCUM avanza exactamente lo que dice VNCP.

    Es el chequeo que delata el offset: con el prefijo de geometria metido
    en los datos, PCUM[0] daba 263 en vez de 1 y las sumas no cerraban.
    """
    d = read_har(SL4)
    ncomp = np.asarray(d["VNCP"].array, dtype=int).ravel()
    pcum = np.asarray(d["PCUM"].array, dtype=int).ravel()
    assert pcum[0] == 1, "los punteros de CUMS son base 1"
    # Donde la variable tiene resultados, el puntero siguiente es este mas
    # sus componentes.
    acc = 1
    for j in range(len(pcum)):
        if pcum[j] == 0:
            continue
        assert pcum[j] == acc, f"PCUM[{j}] = {pcum[j]}, esperado {acc}"
        acc += int(ncomp[j])


@needs_sl4
def test_levels_and_percent_agree() -> None:
    """(LEVA/LEVB - 1)*100 reproduce CUMS donde los dos arrays se alinean.

    Solo valen los primeros: CUMS lista las 1212 componentes endogenas y
    LEVB/LEVA solo las 253 que tienen nivel, con otro orden mas adelante.
    """
    d = read_har(SL4)
    b = np.asarray(d["LEVB"].array, dtype=float).ravel()
    a = np.asarray(d["LEVA"].array, dtype=float).ravel()
    c = np.asarray(d["CUMS"].array, dtype=float).ravel()
    n = 40
    nz = b[:n] != 0
    pct = (a[:n][nz] / b[:n][nz] - 1.0) * 100.0
    np.testing.assert_allclose(pct, c[:n][nz], atol=1e-4)


def test_2ifull_roundtrip_matches_gempack_prefix() -> None:
    """Escribir y releer un 2IFULL conserva los valores.

    Con el escritor emitiendo el prefijo corto y el lector leyendo desde 32
    esto se rompia, que es justo lo que hay que evitar: el par interno tiene
    que hablar el formato de GEMPACK, no uno propio.
    """
    import tempfile
    from pathlib import Path

    from equilibria.babel.har import write_har
    from equilibria.babel.har.symbols import HeaderArray

    arr = np.arange(1, 13, dtype=np.int32).reshape((3, 4))
    ha = HeaderArray(
        name="TEST",
        coeff_name="TEST",
        long_name="ida y vuelta de 2IFULL",
        array=arr,
        set_names=[],
        set_elements=[],
    )
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "t.har"
        write_har(p, {"TEST": ha})
        back = read_har(p)
    np.testing.assert_array_equal(back["TEST"].array, arr)


@needs_sl4
def test_writing_2rfull_raises_instead_of_losing_data() -> None:
    """Re-escribir un .sl4 perdia 1211 de 1212 valores sin avisar.

    No hay _write_2rfull; el enrutador mandaba los float 2-D a _write_refull,
    cuyo data record guarda solo el primer valor. write_har devolvia OK.
    """
    import tempfile
    from pathlib import Path

    from equilibria.babel.har import write_har

    d = read_har(SL4)
    assert np.asarray(d["CUMS"].array).size > 1000, "CUMS deberia traer >1000 valores"
    with tempfile.TemporaryDirectory() as td:
        with pytest.raises(NotImplementedError, match="2RFULL"):
            write_har(Path(td) / "rt.har", d)


@needs_sl4
def test_float_scalar_still_writes_through_refull() -> None:
    """El corte no debe alcanzar a los escalares: GEMPACK los guarda como REFULL."""
    import tempfile
    from pathlib import Path

    from equilibria.babel.har import write_har

    d = read_har(SL4)
    only_scalar = {"UVAL": d["UVAL"]}
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "scalar.har"
        write_har(out, only_scalar)
        back = read_har(out)
    assert float(np.asarray(back["UVAL"].array).ravel()[0]) == pytest.approx(
        0.1, abs=1e-7
    )
