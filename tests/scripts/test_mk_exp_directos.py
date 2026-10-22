"""mk_exp_directos.py: traduce el ``rate% N from file X.shk`` de la GUI de RunGTAP a
shocks directos, celda por celda (ME8, TBL813).

Oraculo: los logs de GEMPACK de la corrida en Windows (2026-09-30,
tests/fixtures/gtap7_gempack/burfisher/<exp>.log.gz), que repiten el command file
que corrio. No salen de este script.
"""

from __future__ import annotations

import gzip
import importlib.util
import shutil
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "gtap" / "mk_exp_directos.py"
PKG = ROOT / "datasets" / "nus333"
LOGS = ROOT / "tests" / "fixtures" / "gtap7_gempack" / "burfisher"


def _load() -> Any:
    spec = importlib.util.spec_from_file_location("mk_exp_directos", SCRIPT)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _shocks(text: str) -> list[str]:
    return sorted(
        ln.strip() for ln in text.replace("\r", "").splitlines()
        if ln.strip().startswith("Shock ")
    )  # fmt: skip


@pytest.fixture(scope="module")
def generados(tmp_path_factory: pytest.TempPathFactory) -> Path:
    dest = tmp_path_factory.mktemp("pkg")
    for f in PKG.iterdir():
        if not f.name.endswith("-dir.exp"):
            shutil.copy(f, dest / f.name)
    assert _load().main(str(dest)) == 0
    return dest


@pytest.mark.parametrize("exp", ["me8", "tbl813"])
def test_los_shocks_son_los_que_corrieron_en_windows(generados: Path, exp: str) -> None:
    with gzip.open(LOGS / f"{exp}.log.gz", "rt", errors="replace") as fh:
        want = _shocks(fh.read())
    got = _shocks((generados / f"{exp}-dir.exp").read_text())
    assert got == want
    assert len(got) == {"me8": 63, "tbl813": 1}[exp]


def test_los_dir_del_paquete_son_los_que_genera_el_script(generados: Path) -> None:
    for exp in ("me8", "tbl813"):
        assert (PKG / f"{exp}-dir.exp").read_text() == (
            generados / f"{exp}-dir.exp"
        ).read_text()


def test_target_no_se_traduce(generados: Path) -> None:
    assert not (generados / "tbl53-dir.exp").exists()


def test_no_pisa_un_dir_existente(generados: Path) -> None:
    (generados / "me8-dir.exp").write_text("editado a mano")
    _load().main(str(generados))
    assert (generados / "me8-dir.exp").read_text() == "editado a mano"
