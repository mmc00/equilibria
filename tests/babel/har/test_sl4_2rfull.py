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

from pathlib import Path

import numpy as np
import pytest

from equilibria.babel.har import read_har, write_har

REPO_ROOT = Path(__file__).resolve().parents[3]
SL4 = REPO_ROOT / "tools/gempack-oracle/out/TBL45A/TBL45A.sl4"

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


def test_2ifull_roundtrip_matches_gempack_prefix(tmp_path: Path) -> None:
    """Escribir y releer un 2IFULL conserva los valores.

    Con el escritor emitiendo el prefijo corto y el lector leyendo desde 32
    esto se rompia, que es justo lo que hay que evitar: el par interno tiene
    que hablar el formato de GEMPACK, no uno propio.
    """
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
    p = tmp_path / "t.har"
    write_har(p, {"TEST": ha})
    np.testing.assert_array_equal(read_har(p)["TEST"].array, arr)


@needs_sl4
def test_rewriting_a_sl4_keeps_every_value(tmp_path: Path) -> None:
    """Re-escribir un .sl4 completo conserva todos los valores.

    Historia: sin _write_2rfull el enrutador mandaba los float 2-D a
    _write_refull, cuyo data record guarda solo el primer valor, y write_har
    devolvia OK -- se perdian 1211 de 1212 valores de CUMS sin aviso. Despues
    paso a lanzar NotImplementedError, que era honesto pero dejaba el .sl4
    sin poder re-escribirse. Ahora el round-trip cierra de verdad.
    """
    d = read_har(SL4)
    assert np.asarray(d["CUMS"].array).size > 1000, "CUMS deberia traer >1000 valores"
    out = tmp_path / "rt.har"
    write_har(out, d)
    back = read_har(out)
    assert len(back) == len(d)
    for k in ("CUMS", "LEVB", "LEVA", "UVAL", "SHOC"):
        a = np.asarray(d[k].array)
        b = np.asarray(back[k].array)
        assert b.shape == a.shape, f"{k}: {a.shape} -> {b.shape}"
        assert b.dtype == a.dtype, f"{k}: dtype {a.dtype} -> {b.dtype}"
        np.testing.assert_allclose(b, a, atol=1e-6)


def test_set_less_float_roundtrips_at_every_rank(tmp_path: Path) -> None:
    """Un REFULL sin sets debe releerse completo, cualquiera sea su rango.

    _read_refull derivaba la cantidad de valores de los SETS: sin sets daba
    n == 1, asi que devolvia el primer valor y descartaba el resto -- aunque
    el escritor los habia puesto todos en disco. La forma real solo vive en
    el dim-summary record, y de ahi se recupera.

    No necesita la corrida de GEMPACK: arma los arrays a mano.
    """
    from equilibria.babel.har.symbols import HeaderArray

    # 2-D sin sets queda afuera a proposito: esa forma ES un 2RFULL para el
    # escritor (ver el test siguiente), no un REFULL.
    for shape in [(5,), (2, 2, 2), (7,), (2, 3, 4)]:
        arr = np.arange(1, int(np.prod(shape)) + 1, dtype=np.float32).reshape(shape)
        ha = HeaderArray(
            name="T",
            coeff_name="T",
            long_name=f"set-less float {shape}",
            array=arr,
            set_names=[],
            set_elements=[],
        )
        out = tmp_path / f"r{len(shape)}_{arr.size}.har"
        write_har(out, {"T": ha})
        back = np.asarray(read_har(out)["T"].array)
        assert back.shape == arr.shape, f"{shape}: releido como {back.shape}"
        np.testing.assert_allclose(back, arr)


@needs_sl4
def test_2rfull_keeps_its_on_disk_type(tmp_path: Path) -> None:
    """UVAL es 2RFULL 1x1 en el .sl4 y tiene que volver a disco como 2RFULL.

    El corte no puede ser por tamano: GEMPACK guarda UVAL y SHOC (1x1) como
    2RFULL y el escalar DVER de default.prm como REFULL, asi que la forma no
    dice nada del tipo. Lo que lo decide es float + 2-D + sin sets.
    """
    d = read_har(SL4)
    assert np.asarray(d["UVAL"].array).shape == (1, 1)
    out = tmp_path / "uval.har"
    write_har(out, {"UVAL": d["UVAL"]})
    assert b"2RFULL" in out.read_bytes(), "se escribio con otro tipo"
    assert b"REFULL" not in out.read_bytes().replace(b"2RFULL", b"")
    np.testing.assert_allclose(
        np.asarray(read_har(out)["UVAL"].array), np.asarray(d["UVAL"].array)
    )


def test_int_header_stays_2ifull_when_edited(tmp_path: Path) -> None:
    """Editar un 2IFULL no lo puede convertir en 2RFULL.

    Es la regresion que rompio `run_gempack_matrix --rordelta`: RDLT
    (RORDELTA) es un coeficiente ENTERO de GTAPv7.tab guardado como 2IFULL, y
    _force_rdlt lo casteaba a float, con lo cual pasaba a escribirse 2RFULL.
    GEMPACK no lee eso -- "(E-incompatible data types on file and demanded by
    user)" -- y el .prm resultante no corre. Medido en gtap7_20x41: como
    2IFULL el solve reproduce la fixture capFix (rore spread 15.2428); como
    2RFULL no arranca.
    """
    from equilibria.babel.har.symbols import HeaderArray

    ha = HeaderArray(
        name="RDLT",
        coeff_name="RDLT",
        long_name="RORDELTA, entero 1x1",
        array=np.array([[1]], dtype=np.int32),
        set_names=[],
        set_elements=[],
    )
    p = tmp_path / "prm.har"
    arr = np.asarray(ha.array).copy()
    arr[...] = 0  # lo que hace _force_rdlt: preservar el dtype de disco
    ha.array = arr
    write_har(p, {"RDLT": ha})
    assert b"2IFULL" in p.read_bytes(), "el header dejo de ser 2IFULL"
    assert b"2RFULL" not in p.read_bytes()
    back = read_har(p)["RDLT"]
    assert back.array.dtype == np.int32
    assert int(np.asarray(back.array).ravel()[0]) == 0
