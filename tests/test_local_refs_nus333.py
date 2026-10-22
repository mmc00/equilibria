"""``nus333_dir()``: el paquete de Burfisher viene en el repo (``datasets/nus333``).

Antes apuntaba a ``~/Downloads/10284``, el disco del autor. Ahora lo lee del repo, y
``EQUILIBRIA_NUS333_DIR`` sigue pudiendo apuntar a otra copia.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from equilibria._local_refs import nus333_dir

ROOT = Path(__file__).resolve().parents[1]


def test_por_defecto_es_el_paquete_del_repo(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("EQUILIBRIA_NUS333_DIR", raising=False)
    assert nus333_dir() == ROOT / "datasets" / "nus333"
    for f in ("sets.har", "basedata.har", "default.prm", "esubd0.8.prm", "tbl46a.exp"):
        assert (nus333_dir() / f).is_file(), f


def test_la_variable_de_entorno_manda(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("EQUILIBRIA_NUS333_DIR", str(tmp_path))
    assert nus333_dir() == tmp_path
