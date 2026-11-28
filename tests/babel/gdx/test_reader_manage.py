"""The GDX reader on MANAGE-WB's Ghana GDX (data/manage/gha_gtap9), against gdxdump.

Measured on 2026-11-27 at origin/main de84529a: in ``GHA_BaU.gdx`` the reader
returned no record for 276 of the 280 non-empty variables (``xp``: 0 of 102) and
for 134 of the 185 non-empty parameters; 8 of the 22 non-empty parameters of the
bridge GDX failed too. These files carry the empty label ``''`` in their UEL
(GDXXRW keeps empty Excel labels), which the GTAP and PEP specimens never had, and
the reader stopped reading the UEL there.
"""

from __future__ import annotations

import csv
import io
import math
import subprocess
from pathlib import Path

import pytest

from equilibria.babel.gdx.gdxdump import locate_gdxdump
from equilibria.babel.gdx.reader import (
    get_equations,
    get_parameters,
    get_variables,
    read_equation_values,
    read_gdx,
    read_parameter_values,
    read_variable_values,
)
from equilibria.sources import dataset_dir, reference

FIELDS = ("level", "marginal", "lower", "upper", "scale")
SPECIAL = {"+Inf": math.inf, "-Inf": -math.inf, "Eps": 0.0, "NA": math.nan}


def _bau() -> Path:
    return (
        reference(
            "gha_gtap9", "gams", form="mcp", task="baseline", year=2015,
            mcp_solver="nlpec-ipopt",
        )
        / "GHA_BaU.gdx"
    )  # fmt: skip


def _bridge() -> Path:
    return dataset_dir("gha_gtap9") / "BaseBridge_GHA_GTAP_9.gdx"


# The PATH run's out.gdx (9,452 scalars x1..xN, no empty label) read correctly
# before the fix too; dumping it symbol by symbol takes about 4 minutes, so it is
# not a case here.
CASES = {"bau": _bau, "bridge": _bridge}


@pytest.fixture(scope="module")
def bau() -> dict:
    return read_gdx(_bau())


def test_uel_keeps_the_empty_label(bau: dict) -> None:
    """The UEL header says 1,532 labels; the 271st is the empty label '' (gdxdump
    prints it between '0' and '2011'), and the reader used to stop there."""
    elements = bau["elements"]
    assert len(elements) == 1532
    assert elements[269:272] == ["0", "", "2011"]


def test_xp_has_every_record(bau: dict) -> None:
    xp = read_variable_values(bau, "xp")
    assert len(xp) == 102
    assert xp[("a-pdr", "2014")]["level"] == pytest.approx(0.43534821117816, rel=1e-14)
    assert xp[("a-pdr", "2015")]["level"] == pytest.approx(0.446127849305925, rel=1e-14)


def test_scalar_rescale(bau: dict) -> None:
    assert read_parameter_values(bau, "reScale")[()] == pytest.approx(
        3.95423752973879e-4, rel=1e-14
    )


# ---- every symbol against gdxdump ------------------------------------------------


def _text_value(text: str) -> float | None:
    if text in SPECIAL:
        return SPECIAL[text]
    try:
        return float(text)
    except ValueError:
        return None  # an acronym: gdxdump prints its name


def _same(ours: float, text: str) -> bool:
    ref = _text_value(text)
    if ref is None:
        return True
    if math.isnan(ref):
        return math.isnan(ours)
    if math.isinf(ref):
        return ours == ref
    return abs(ours - ref) <= 1e-13 * max(1.0, abs(ref))  # 15 significant digits


def _gdxdump(gdx: Path, name: str, nvals: int, dim: int) -> dict[tuple, list[str]]:
    exe = locate_gdxdump()
    assert exe is not None
    args = [exe, str(gdx), f"symb={name}", "format=csv"]
    if nvals == 5:
        args.append("CSVAllFields")
    out = subprocess.run(args, capture_output=True, text=True, check=True).stdout
    rows = list(csv.reader(io.StringIO(out)))
    if not rows:
        return {}
    assert len(rows[0]) == dim + nvals, (name, rows[0])
    return {tuple(r[:dim]): r[dim:] for r in rows[1:]}


def _ours(data: dict, kind: str, name: str) -> dict[tuple, list[float]]:
    """The reader's records of one symbol, as value lists in gdxdump's order."""
    if kind == "parameter":
        return {k: [v] for k, v in read_parameter_values(data, name).items()}
    read = read_variable_values if kind == "variable" else read_equation_values
    return {k: [r[f] for f in FIELDS] for k, r in read(data, name).items()}


@pytest.mark.needs_gdxdump
@pytest.mark.parametrize("case", sorted(CASES))
def test_every_symbol_matches_gdxdump(case: str) -> None:
    gdx = CASES[case]()
    data = read_gdx(gdx)
    checked = 0
    for kind, symbols in (
        ("parameter", get_parameters(data)),
        ("variable", get_variables(data)),
        ("equation", get_equations(data)),
    ):
        for sym in symbols:
            if sym["records"] == 0:
                continue
            name, dim = sym["name"], sym["dimension"]
            ref = _gdxdump(gdx, name, 1 if kind == "parameter" else 5, dim)
            ours = _ours(data, kind, name)
            assert ours.keys() == ref.keys(), (
                case,
                name,
                f"{len(ours)} records vs {len(ref)} from gdxdump",
            )
            for key, texts in ref.items():
                for value, text in zip(ours[key], texts, strict=True):
                    assert _same(value, text), (case, name, key, value, text)
            checked += 1
    assert checked > 0
