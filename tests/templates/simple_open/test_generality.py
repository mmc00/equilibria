"""simple_open is not tied to bread and milk: an invented three-good SAM solves."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, cast

import pyomo.environ as pyo
import pytest

from equilibria.sources import dataset_dir
from equilibria.templates.simple_open import SimpleOpenEconomy

SRC = Path(__file__).resolve().parents[3] / "src" / "equilibria"
CODE = [
    SRC / "templates" / "simple_open",
    SRC / "blocks" / "_levels.py",
    SRC / "blocks" / "production" / "__init__.py",
    SRC / "blocks" / "trade" / "__init__.py",
    SRC / "blocks" / "demand" / "__init__.py",
    SRC / "blocks" / "institutions" / "__init__.py",
    SRC / "blocks" / "equilibrium" / "__init__.py",
]


@pytest.fixture(scope="module")
def three_goods() -> SimpleOpenEconomy:
    return SimpleOpenEconomy(sam_path=dataset_dir("invented_3x2") / "sam.csv")


def _levels(pm: Any) -> dict[tuple[str, Any], float]:
    return {
        (v.name, k): pyo.value(v[k])
        for v in pm.component_objects(pyo.Var, active=True)
        for k in v
    }


def test_three_goods_build_and_return_base(three_goods: SimpleOpenEconomy) -> None:
    cal = three_goods.calibration()
    assert len(cal.goods) == 3
    run = three_goods.run()
    base, check = _levels(run["base"]), _levels(run["check"])
    for key, b in base.items():
        assert check[key] == pytest.approx(b, rel=1e-8, abs=1e-8), key
    shock = cast("Any", run["shock"])
    assert abs(pyo.value(shock.WALRAS)) <= 1e-8
    assert any(
        abs(pyo.value(shock.M[i]) / base[("M", i)] - 1) > 0.01 for i in cal.goods
    )


def _new_level_blocks(text: str) -> str:
    """The code of the level-form blocks only (the old demo blocks name goods)."""
    marker = "# --- level-form blocks"
    if marker in text:
        return text[text.index(marker) :]
    if "LevelBlock" not in text:
        return ""
    return text[text.index("class ", text.index("LevelBlock")) :]


def test_no_dataset_names_in_code() -> None:
    found = []
    for root in CODE:
        files = [root] if root.is_file() else sorted(root.rglob("*.py"))
        for f in files:
            text = f.read_text()
            if "blocks" in f.parts and f.name == "__init__.py":
                text = _new_level_blocks(text)
            found += [
                f"{f.name}: {m}" for m in re.findall(r"\b(?:BRD|MLK|CAP|LAB)\b", text)
            ]
    assert found == []


def _sparse_sam(tmp_path: Path) -> Path:
    """The book's SAM with two zero cells, rebalanced: BRD buys no MLK
    (paid to labour instead, spent by the household on MLK) and the
    government buys no MLK (bought as BRD, the household swaps the same)."""
    with (dataset_dir("hosoe_2x2") / "sam.csv").open() as fh:
        rows = [line.rstrip("\n").split(",") for line in fh]
    col = {name: k for k, name in enumerate(rows[0])}
    row = {r[0]: r for r in rows[1:]}

    def add(r: str, c: str, v: float) -> None:
        row[r][col[c]] = repr(float(row[r][col[c]]) + v)

    add("MLK", "BRD", -17.0)
    add("LAB", "BRD", 17.0)
    add("HOH", "LAB", 17.0)
    add("MLK", "HOH", 17.0)
    add("MLK", "GOV", -14.0)
    add("BRD", "GOV", 14.0)
    add("BRD", "HOH", -14.0)
    add("MLK", "HOH", 14.0)
    path = tmp_path / "sam.csv"
    path.write_text("\n".join(",".join(r) for r in rows) + "\n")
    (tmp_path / "closure.yaml").write_text("numeraire: LAB\n")
    return path


def test_zero_cells_outside_trade_solve(tmp_path: Path) -> None:
    template = SimpleOpenEconomy(sam_path=_sparse_sam(tmp_path))
    cal = template.calibration()
    assert cal.base["XI"][cal.goods.index("MLK"), cal.goods.index("BRD")] == 0
    assert cal.base["XG"][cal.goods.index("MLK")] == 0
    run = template.run()
    base, check = _levels(run["base"]), _levels(run["check"])
    for key, b in base.items():
        assert check[key] == pytest.approx(b, rel=1e-8, abs=1e-8), key
    assert abs(pyo.value(cast("Any", run["shock"]).WALRAS)) <= 1e-8


def test_numeraire_comes_from_the_data() -> None:
    template = SimpleOpenEconomy()
    assert template.numeraire is None
    assert template.numeraire_factor() == "LAB"
    other = SimpleOpenEconomy(sam_path=dataset_dir("invented_3x2") / "sam.csv")
    assert other.numeraire_factor() == "LAB"


def test_missing_closure_file_is_rejected(tmp_path: Path) -> None:
    sam = tmp_path / "sam.csv"
    sam.write_text((dataset_dir("hosoe_2x2") / "sam.csv").read_text())
    with pytest.raises(ValueError, match="closure.yaml"):
        SimpleOpenEconomy(sam_path=sam).build()
    # an explicit numeraire needs no file
    assert SimpleOpenEconomy(sam_path=sam, numeraire="CAP").numeraire_factor() == "CAP"
