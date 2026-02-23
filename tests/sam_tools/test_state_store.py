from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import equilibria.sam_tools.state_store as io_module
from equilibria.sam_tools.models import SAM, SAMState
from equilibria.sam_tools.state_store import load_state, write_state


def _write_canonical_excel(
    path: Path,
    row_keys: list[tuple[str, str]],
    col_keys: list[tuple[str, str]],
    matrix: np.ndarray,
) -> None:
    grid = np.full((len(row_keys) + 2, len(col_keys) + 2), "", dtype=object)
    for j, (cat, elem) in enumerate(col_keys):
        grid[0, 2 + j] = cat
        grid[1, 2 + j] = elem
    for i, (cat, elem) in enumerate(row_keys):
        grid[2 + i, 0] = cat
        grid[2 + i, 1] = elem
        for j in range(len(col_keys)):
            grid[2 + i, 2 + j] = float(matrix[i, j])

    path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        pd.DataFrame(grid).to_excel(writer, sheet_name="SAM", index=False, header=False)


def _write_ieem_raw_excel(path: Path, matrix: np.ndarray, sheet_name: str = "RAW") -> None:
    groups = [
        ("actividades productivas", "act_agr"),
        ("Bienes y servicios", "com_agr"),
        ("Margenes", "marg"),
        ("Factores", "usk"),
        ("Hogares", "hh"),
        ("Empresas", "firm"),
        ("Gobierno", "gvt"),
        ("Resto del mundo", "row"),
        ("Ahorro", "s_hh"),
        ("Inversión", "inv"),
    ]
    if matrix.shape != (len(groups), len(groups)):
        raise ValueError("IEEM raw test matrix has invalid shape")

    n_rows = 4 + len(groups) + 5
    n_cols = 3 + len(groups) + 5
    raw = np.full((n_rows, n_cols), np.nan, dtype=object)
    start_row = 4

    for i, (group_name, label) in enumerate(groups):
        row = start_row + i
        raw[row, 1] = group_name
        raw[row, 2] = label

    for i in range(len(groups)):
        for j in range(len(groups)):
            raw[start_row + i, 3 + j] = float(matrix[i, j])

    path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        pd.DataFrame(raw).to_excel(writer, sheet_name=sheet_name, index=False, header=False)


def _build_state(
    matrix: np.ndarray,
    keys: list[tuple[str, str]],
    source_path: Path,
    source_format: str,
) -> SAMState:
    sam = SAM.from_matrix(matrix, keys, keys)
    return SAMState(
        sam=sam,
        row_keys=keys,
        col_keys=keys,
        source_path=source_path,
        source_format=source_format,
    )


def _to_cell_map(state: SAMState) -> dict[tuple[tuple[str, str], tuple[str, str]], float]:
    out: dict[tuple[tuple[str, str], tuple[str, str]], float] = {}
    for i, r_key in enumerate(state.row_keys):
        for j, c_key in enumerate(state.col_keys):
            value = float(state.matrix[i, j])
            if abs(value) <= 1e-14:
                continue
            out[(r_key, c_key)] = value
    return out


def test_io_excel_roundtrip_small_fixture(tmp_path: Path) -> None:
    output_excel = tmp_path / "sam.xlsx"
    keys = [("A", "a"), ("A", "b")]
    state = _build_state(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=float), keys, tmp_path / "in.xlsx", "excel")

    info = write_state(state, output_excel, output_format="excel", output_symbol="SAM")
    assert info["format"] == "excel"
    loaded = load_state(output_excel, "excel")
    assert loaded.matrix.shape == (2, 2)
    assert loaded.row_keys == [("A", "a"), ("A", "b")]
    assert loaded.col_keys == [("A", "a"), ("A", "b")]
    assert np.allclose(loaded.matrix, state.matrix)


def test_io_write_gdx_returns_metadata(tmp_path: Path) -> None:
    output_gdx = tmp_path / "sam.gdx"
    keys = [("AG", "gvt"), ("I", "agr")]
    state = _build_state(np.array([[1.5, 0.0], [0.0, 2.5]], dtype=float), keys, tmp_path / "in.gdx", "gdx")

    info = write_state(state, output_gdx, output_format="gdx", output_symbol="SAM")
    assert info["format"] == "gdx"
    assert info["records"] == 2
    assert info["symbol"] == "SAM"
    assert output_gdx.exists()


def test_io_load_gdx_with_mocked_reader(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    fake_path = tmp_path / "fake.gdx"
    fake_path.write_bytes(b"fake")

    values = {
        ("AG", "gvt", "I", "agr"): 1.5,
        ("I", "agr", "AG", "gvt"): 2.5,
    }

    monkeypatch.setattr(io_module, "read_gdx", lambda _path: {"filepath": str(fake_path), "symbols": []})
    monkeypatch.setattr(io_module, "read_parameter_values", lambda _gdx, _name: values)

    loaded = load_state(fake_path, "gdx")
    assert loaded.source_format == "gdx"
    assert _to_cell_map(loaded) == {
        (("AG", "gvt"), ("I", "agr")): 1.5,
        (("I", "agr"), ("AG", "gvt")): 2.5,
    }


def test_io_load_ieem_raw_excel_with_sheet_option(tmp_path: Path) -> None:
    raw_file = tmp_path / "raw.xlsx"
    matrix = np.zeros((10, 10), dtype=float)
    matrix[0, 1] = 12.0
    _write_ieem_raw_excel(raw_file, matrix, sheet_name="CUSTOM_SHEET")

    state = load_state(raw_file, "ieem_raw_excel", options={"sheet_name": "CUSTOM_SHEET"})
    assert state.source_format == "ieem_raw_excel"
    assert state.matrix.shape == (10, 10)
    assert all(cat == "RAW" for cat, _ in state.row_keys)
    assert all(cat == "RAW" for cat, _ in state.col_keys)


def test_io_load_ieem_raw_excel_uses_sam_class(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    input_file = tmp_path / "raw.xlsx"
    input_file.write_bytes(b"dummy")

    keys = [("RAW", "a")]
    expected = _build_state(np.array([[1.0]], dtype=float), keys, input_file, "ieem_raw_excel")

    class _DummySAM:
        def to_raw_state(
            self,
            *,
            source_path: Path | None = None,
            source_format: str = "raw",
        ) -> SAMState:
            _ = (source_path, source_format)
            return expected

    called: dict[str, object] = {}

    def _fake_from_ieem_excel(path: Path, sheet_name: str = "MCS2016") -> _DummySAM:
        called["path"] = path
        called["sheet_name"] = sheet_name
        return _DummySAM()

    monkeypatch.setattr(io_module.IEEMRawSAM, "from_ieem_excel", staticmethod(_fake_from_ieem_excel))

    loaded = load_state(input_file, "ieem_raw_excel", options={"sheet_name": "CUSTOM"})
    assert loaded == expected
    assert called["path"] == input_file
    assert called["sheet_name"] == "CUSTOM"


def test_write_state_rejects_unsupported_format(tmp_path: Path) -> None:
    keys = [("AG", "gvt")]
    state = _build_state(np.array([[1.0]], dtype=float), keys, tmp_path / "in.xlsx", "excel")
    with pytest.raises(ValueError, match="Unsupported output format"):
        write_state(state, tmp_path / "out.unknown", output_format="unknown", output_symbol="SAM")
