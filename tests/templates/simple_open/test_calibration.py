"""simple_open calibrates every parameter of stdcge from the SAM, as GAMS does.

GAMS's parameters are read only here, to compare: the model never reads them.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from equilibria.babel.gdx.reader import read_parameter_values, read_variable_values
from equilibria.templates.simple_open import calibrate, load_sam

# equilibria parameter -> GAMS parameter (stdcge.gms)
PARAMS = [
    ("b_VA", "b"),
    ("beta_VA", "beta"),
    ("a_X", "ax"),
    ("a_VA", "ay"),
    ("theta_T", "theta"),
    ("xi_E", "xie"),
    ("xi_D", "xid"),
    ("psi_T", "psi"),
    ("tau_z", "tauz"),
    ("gamma_A", "gamma"),
    ("delta_M", "deltam"),
    ("delta_D", "deltad"),
    ("sigma_A", "sigma"),
    ("tau_m", "taum"),
    ("pWe", "pWe"),
    ("pWm", "pWm"),
    ("Sf", "Sf"),
    ("alpha_H", "alpha"),
    ("ssp", "ssp"),
    ("FF", "FF"),
    ("tau_d", "taud"),
    ("mu_G", "mu"),
    ("ssg", "ssg"),
    ("lambda_V", "lambda"),
]

# equilibria variable -> GAMS variable
VARIABLES = [
    ("VA", "Y"),
    ("FD", "F"),
    ("PVA", "py"),
    ("WF", "pf"),
    ("XI", "X"),
    ("Z", "Z"),
    ("PZ", "pz"),
    ("E", "E"),
    ("D", "D"),
    ("PD", "pd"),
    ("Q", "Q"),
    ("M", "M"),
    ("PQ", "pq"),
    ("PE", "pe"),
    ("PM", "pm"),
    ("EPS", "epsilon"),
    ("XP", "Xp"),
    ("SP", "Sp"),
    ("TD", "Td"),
    ("TZ", "Tz"),
    ("TM", "Tm"),
    ("XG", "Xg"),
    ("SG", "Sg"),
    ("XV", "Xv"),
]


def _cells(value: Any, axes: list[tuple[str, ...]]) -> dict[tuple[str, ...], float]:
    """An array (or scalar) of the calibration as {GAMS key: value}."""
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 0:
        return {(): float(arr)}
    out = {}
    for idx in np.ndindex(arr.shape):
        out[tuple(axes[d][i] for d, i in enumerate(idx))] = float(arr[idx])
    return out


def _axes(calibration: Any, dims: int, name: str) -> list[tuple[str, ...]]:
    # two-dimensional arrays: (factor, sector) for beta_VA/FD, (good, sector) else
    if dims == 2 and name in ("beta_VA", "FD"):
        return [calibration.factors, calibration.goods]
    if dims == 1 and name in ("FF", "WF"):
        return [calibration.factors]
    return [calibration.goods] * dims


def test_sets_come_from_the_sam(hosoe_sam: Any) -> None:
    assert hosoe_sam.goods == ("BRD", "MLK")
    assert hosoe_sam.factors == ("CAP", "LAB")


@pytest.mark.parametrize(("ours", "gams"), PARAMS)
def test_parameter_matches_gams(
    calibration: Any, base_gdx: dict[str, Any], ours: str, gams: str
) -> None:
    value = calibration.values[ours]
    expected = read_parameter_values(base_gdx, gams)
    got = _cells(value, _axes(calibration, np.ndim(value), ours))
    # GAMS does not write zeros: compare on the union, a missing cell is 0
    for key in set(got) | set(expected):
        assert got.get(key, 0.0) == pytest.approx(
            expected.get(key, 0.0), rel=1e-12, abs=1e-14
        ), (ours, key)


@pytest.mark.parametrize(("ours", "gams"), VARIABLES)
def test_base_level_matches_gams(
    calibration: Any, base_gdx: dict[str, Any], ours: str, gams: str
) -> None:
    value = calibration.base[ours]
    expected = {k: r["level"] for k, r in read_variable_values(base_gdx, gams).items()}
    got = _cells(value, _axes(calibration, np.ndim(value), ours))
    for key in set(got) | set(expected):
        assert got.get(key, 0.0) == pytest.approx(
            expected.get(key, 0.0), rel=1e-12, abs=1e-14
        ), (ours, key)


def _write(path: Path, rows: list[list[str]]) -> Path:
    with path.open("w", newline="") as fh:
        csv.writer(fh).writerows(rows)
    return path


def _hosoe_rows(hosoe_sam_path: Path) -> list[list[str]]:
    with hosoe_sam_path.open(newline="") as fh:
        return list(csv.reader(fh))


def test_unbalanced_sam_names_account(tmp_path: Path, hosoe_sam_path: Path) -> None:
    rows = _hosoe_rows(hosoe_sam_path)
    rows[1][2] = "9.0"  # SAM(BRD, MLK) +1: the BRD row no longer balances
    with pytest.raises(ValueError, match="BRD"):
        load_sam(_write(tmp_path / "sam.csv", rows))


def test_zero_import_flow_is_rejected(tmp_path: Path, hosoe_sam_path: Path) -> None:
    rows = _hosoe_rows(hosoe_sam_path)
    header = rows[0]
    ext_row = next(r for r in rows if r[0] == "EXT")
    brd_row = next(r for r in rows if r[0] == "BRD")
    col = header.index("BRD")
    imports = float(ext_row[col])
    # move BRD's imports to domestic intermediate use, keeping the SAM balanced
    ext_row[col] = "0.0"
    brd_row[col] = repr(float(brd_row[col]) + imports)
    inv_row = next(r for r in rows if r[0] == "INV")
    # EXT now receives less: foreign saving and BRD's investment fall by as much
    inv_row[header.index("EXT")] = repr(float(inv_row[header.index("EXT")]) - imports)
    brd_row[header.index("INV")] = repr(float(brd_row[header.index("INV")]) - imports)
    with pytest.raises(ValueError, match=r"BRD.*import"):
        calibrate(load_sam(_write(tmp_path / "sam.csv", rows)))
