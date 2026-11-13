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
            text = text.replace('default="LAB"', "")
            found += [
                f"{f.name}: {m}" for m in re.findall(r"\b(?:BRD|MLK|CAP|LAB)\b", text)
            ]
    assert found == []
