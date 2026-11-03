"""_force_rdlt (run_gempack_matrix.py): el .prm capFix que GEMPACK 11.3 si lee.

Antes reescribia el .prm entero con ``write_har``, que no conserva el ancho de los
1CFULL (XXCR/XXCD 70 -> 41, XXCP 6 -> 12) ni el troceo de registros; GEMSIM 11.3 lo
rechaza (``E-UNEXPECTED LENGTH OF STRINGS``, luego ``E-NREC not as expected``). Ahora
cambia solo el int32 de RDLT, in situ.

Oraculo: ``default_gtap7_20x41_rdlt0_windows.prm``, hecho a mano en Windows el
2026-10-19 (``4_niveles/a_mano/patch_rdlt.py`` en la rama de datos
windows/corridas-paper-2026-10-19_1928) y
con el que GEMSIM 11.3 resolvio el 20x41 capFix (max residual ratio 1.05e-7).
"""

from __future__ import annotations

import importlib.util
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from equilibria.babel.har.reader import read_har
from equilibria.sources import dataset_dir

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "data" / "gtap7" / "gempack" / "tools" / "run_gempack_matrix.py"
PRM = dataset_dir("gtap7_20x41") / "default.prm"
ORACLE = (
    ROOT
    / "tests"
    / "fixtures"
    / "gtap7_gempack"
    / "default_gtap7_20x41_rdlt0_windows.prm"
)

pytestmark = pytest.mark.skipif(not PRM.exists(), reason="falta data/gtap7/gtap7_20x41")


def _load_module() -> Any:
    sys.path.insert(0, str(SCRIPT.parent))
    spec = importlib.util.spec_from_file_location("run_gempack_matrix", SCRIPT)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_capfix_es_byte_a_byte_el_prm_que_corrio_en_windows(tmp_path: Path) -> None:
    prm = tmp_path / "default_capfix.prm"
    shutil.copy2(PRM, prm)
    _load_module()._force_rdlt(prm, 0)
    assert prm.read_bytes() == ORACLE.read_bytes()


def test_solo_cambia_rdlt(tmp_path: Path) -> None:
    prm = tmp_path / "default_capfix.prm"
    shutil.copy2(PRM, prm)
    _load_module()._force_rdlt(prm, 0)
    before, after = read_har(str(PRM)), read_har(str(prm))
    assert before.keys() == after.keys()
    assert int(np.asarray(before["RDLT"].array).item()) == 1
    assert int(np.asarray(after["RDLT"].array).item()) == 0
    for k in before:
        if k != "RDLT":
            assert np.array_equal(
                np.asarray(before[k].array), np.asarray(after[k].array)
            ), k
    diff = np.frombuffer(PRM.read_bytes(), np.uint8) != np.frombuffer(
        prm.read_bytes(), np.uint8
    )
    assert 1 <= int(diff.sum()) <= 4  # un int32


def test_capflex_sobre_un_prm_que_ya_es_1_no_cambia_nada(tmp_path: Path) -> None:
    prm = tmp_path / "default_capflex.prm"
    shutil.copy2(PRM, prm)
    _load_module()._force_rdlt(prm, 1)
    assert prm.read_bytes() == PRM.read_bytes()


def test_sin_rdlt_es_error(tmp_path: Path) -> None:
    prm = tmp_path / "x.prm"
    prm.write_bytes(PRM.read_bytes().replace(b"RDLT", b"XDLT"))
    with pytest.raises(ValueError, match="RDLT"):
        _load_module()._force_rdlt(prm, 0)
