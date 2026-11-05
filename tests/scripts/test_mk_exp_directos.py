"""mk_exp_directos.py: traduce el ``rate% N from file X.shk`` de la GUI de RunGTAP a
shocks directos, celda por celda (ME8, TBL813).

Oraculo: los logs de GEMPACK de la corrida en Windows (2026-09-30), que repiten el
command file que corrio. No salen de este script. Since phase 3b the shocks are read
from the command file of the book's records (``nus333/gempack_burfisher/<exp>/run.cmf``);
OLD_LOG_SHOCKS_SHA256 pins them to the 2026-09-30 logs, so the oracle stays
independent of the ``-dir.exp`` this script writes.
"""

from __future__ import annotations

import hashlib
import importlib.util
import shutil
from pathlib import Path
from typing import Any

import pytest

from equilibria.sources import dataset_dir, reference

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "gtap" / "mk_exp_directos.py"
PKG = dataset_dir("nus333")
# sha256 of the sorted "Shock" lines of the 2026-09-30 logs (<exp>.log.gz, d333cf63).
OLD_LOG_SHOCKS_SHA256 = {
    "me8": "1fff7dcb50c9ceb5edbafbc63d1ec5a872f9b91ecf44e947a0fcd712621b0c65",
    "tbl813": "b737f40b186c5b4805cfb4398a39071df2ecf5fc792c4541ca99b0242a9c1a65",
}


def _ran(exp: str) -> str:
    """The command file of the book's Windows run of ``exp``."""
    rec = reference("nus333", "gempack", experiment=exp, method="book")
    return (rec / "run.cmf").read_text(errors="replace")


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
        # the package's files; recorded results sit in subfolders
        if f.is_file() and not f.name.endswith("-dir.exp"):
            shutil.copy(f, dest / f.name)
    assert _load().main(str(dest)) == 0
    return dest


@pytest.mark.parametrize("exp", ["me8", "tbl813"])
def test_los_shocks_son_los_que_corrieron_en_windows(generados: Path, exp: str) -> None:
    want = _shocks(_ran(exp))
    got = _shocks((generados / f"{exp}-dir.exp").read_text())
    assert got == want
    assert len(got) == {"me8": 63, "tbl813": 1}[exp]


@pytest.mark.parametrize("exp", ["me8", "tbl813"])
def test_the_cmf_shocks_are_the_2026_09_30_logs(exp: str) -> None:
    digest = hashlib.sha256("\n".join(_shocks(_ran(exp))).encode()).hexdigest()
    assert digest == OLD_LOG_SHOCKS_SHA256[exp]


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
