"""Hosoe's stdcge (gamslib SEQ=276) and its GAMS reference: base, check, shock."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from equilibria.babel.gdx.reader import (
    get_variables,
    read_gdx,
    read_parameter_values,
    read_variable_values,
)
from equilibria.sources import dataset_dir, load_source, reference


def _folder() -> Path:
    return reference("hosoe_2x2", "gams", form="nlp", experiment="tariff0")


def _levels(gdx: dict) -> dict[tuple[str, tuple[str, ...]], float]:
    return {
        (v["name"], key): rec["level"]
        for v in get_variables(gdx)
        for key, rec in read_variable_values(gdx, v["name"]).items()
    }


def test_reference_has_three_points() -> None:
    names = {p.name for p in _folder().glob("*.gdx")}
    assert names == {"base.gdx", "check.gdx", "shock.gdx"}


def test_gamslib_copy_is_a_model() -> None:
    root = dataset_dir("hosoe_2x2").parent
    assert load_source(root / "gamslib").kind == "model"
    assert load_source(root / "gams").kind == "model"


def test_gams_check_returns_base() -> None:
    base = _levels(read_gdx(_folder() / "base.gdx"))
    check = _levels(read_gdx(_folder() / "check.gdx"))
    # UU is not initialized by the .gms before the first solve
    compared = [k for k in base if k[0] != "UU"]
    assert len(compared) > 40
    for key in compared:
        assert check[key] == pytest.approx(base[key], rel=1e-6, abs=1e-9), key


def test_shock_moves_imports_and_zeroes_tariffs() -> None:
    base = read_gdx(_folder() / "base.gdx")
    shock = read_gdx(_folder() / "shock.gdx")
    m0 = {k: r["level"] for k, r in read_variable_values(base, "M").items()}
    m1 = {k: r["level"] for k, r in read_variable_values(shock, "M").items()}
    assert set(m0) == {("BRD",), ("MLK",)}
    for good in m0:
        assert abs(m1[good] / m0[good] - 1) > 0.01, good
    assert all(v == 0 for v in read_parameter_values(shock, "taum").values())
    assert read_parameter_values(base, "taum")


def test_sam_csv_matches_gdx_sam() -> None:
    sam = read_parameter_values(read_gdx(_folder() / "base.gdx"), "SAM")
    with (dataset_dir("hosoe_2x2") / "sam.csv").open(newline="") as fh:
        rows = list(csv.reader(fh))
    header = rows[0][1:]
    from_csv = {
        (row[0], col): float(val)
        for row in rows[1:]
        for col, val in zip(header, row[1:], strict=True)
        if float(val) != 0.0
    }
    assert from_csv == sam
