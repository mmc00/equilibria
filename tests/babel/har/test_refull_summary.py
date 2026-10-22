"""El dim-summary record de un REFULL lleva la cuenta de registros, no el rango.

GEMPACK lee el primer entero de ese registro como la cantidad de registros que
quedan en la seccion de datos (resumen + encabezado de bloque + datos: 3 con un
solo bloque) y aborta con "E-NREC not as expected" si no coincide. El escritor
ponia ahi arr.ndim, asi que solo los REFULL de 3 dimensiones eran legibles para
GEMPACK: balchk, signchk y taxchk fallaban en la primera matriz 2-D.
"""

from __future__ import annotations

import struct
import warnings
from pathlib import Path

import numpy as np
import pytest

from equilibria.babel.har import HeaderArray, read_har
from equilibria.babel.har.writer import write_har

REPO_ROOT = Path(__file__).resolve().parents[3]
FIXTURE = REPO_ROOT / "src/equilibria/templates/reference/gtap/data/9x10/basedata.har"


def _records(path: Path) -> list[bytes]:
    data = path.read_bytes()
    out, i = [], 0
    while i < len(data):
        n = struct.unpack_from("<i", data, i)[0]
        out.append(data[i + 4 : i + 4 + n])
        i += 8 + n
    return out


def _refull_summaries(path: Path) -> list[tuple[str, int, int]]:
    """Por cada REFULL: nombre, primer entero del dim-summary y registros de datos.

    Los registros de datos se cuentan desde el dim-summary hasta el registro de
    nombre (4 bytes) del encabezado siguiente.
    """
    recs = _records(path)
    res = []
    for k, rec in enumerate(recs):
        if len(rec) != 4 or k + 1 >= len(recs) or recs[k + 1][4:10] != b"REFULL":
            continue
        n_unique = struct.unpack_from("<i", recs[k + 2], 4)[0]
        inicio = k + 3 + n_unique
        fin = inicio
        while fin + 1 < len(recs) and len(recs[fin + 1]) != 4:
            fin += 1
        cuenta = struct.unpack_from("<i", recs[inicio], 4)[0]
        res.append((rec.decode().strip(), cuenta, fin - inicio + 1))
    return res


@pytest.mark.parametrize(
    ("shape", "sets"),
    [
        ((), []),
        ((5,), []),
        ((5,), ["A"]),
        ((3, 4), ["A", "B"]),
        ((2, 3, 4), ["A", "B", "C"]),
        ((2, 3, 4, 2), ["A", "B", "C", "D"]),
    ],
)
def test_resumen_lleva_la_cuenta_de_registros(tmp_path: Path, shape, sets) -> None:
    arr = np.arange(1, int(np.prod(shape)) + 1, dtype=np.float32).reshape(shape)
    elems = (
        [[f"{s}{j}" for j in range(n)] for s, n in zip(sets, shape, strict=True)]
        if sets
        else []
    )
    ha = HeaderArray(
        name="T",
        coeff_name="T",
        long_name="t",
        array=arr,
        set_names=sets,
        set_elements=elems,
    )
    out = tmp_path / "t.har"
    write_har(out, {"T": ha})

    [(_, cuenta, registros)] = _refull_summaries(out)
    assert cuenta == registros == 3, f"{shape}: resumen dice {cuenta}"
    np.testing.assert_allclose(np.asarray(read_har(out)["T"].array), arr)


def test_gempack_escribe_la_cuenta_de_registros() -> None:
    """La convencion sale de GEMPACK: cuenta == registros en todos sus REFULL."""
    if not FIXTURE.exists():
        pytest.skip(f"fixture not present: {FIXTURE}")
    resumenes = _refull_summaries(FIXTURE)
    assert resumenes
    assert [r for r in resumenes if r[1] != r[2]] == []


def test_reescribir_un_archivo_de_gempack_conserva_la_cuenta(tmp_path: Path) -> None:
    if not FIXTURE.exists():
        pytest.skip(f"fixture not present: {FIXTURE}")
    out = tmp_path / "basedata.har"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        write_har(out, read_har(FIXTURE))
    resumenes = _refull_summaries(out)
    assert resumenes
    assert [r for r in resumenes if r[1] != r[2]] == []
