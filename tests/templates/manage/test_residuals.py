"""The block-side residual harness: the rows of named equation families of a
built Pyomo model, evaluated at the values it holds, with no solve.

Checked on simple_open, whose blocks hold at the calibrated base (stdcge)."""

from __future__ import annotations

from tests.templates.simple_open._helpers import block_at_base

from equilibria.blocks.production import CobbDouglasValueAdded
from equilibria.templates.manage.residuals import block_residuals
from equilibria.templates.simple_open import SimpleOpenEconomy


def _value_added():  # noqa: ANN202
    cal = SimpleOpenEconomy().calibration()
    block = CobbDouglasValueAdded(values=cal.values, start=cal.base)
    return block_at_base(block, cal)


def test_zero_at_the_calibrated_base() -> None:
    res = block_residuals(_value_added(), ["eq_va", "eq_fd"])
    assert set(res) == {"eq_va", "eq_fd"}
    assert len(res["eq_fd"]) == 4  # 2 factors x 2 sectors
    assert max(abs(r) for fam in res.values() for r in fam.values()) <= 1e-10


def test_a_moved_value_shows_in_its_rows_only() -> None:
    pm = _value_added()
    pm.VA["BRD"].value = pm.VA["BRD"].value * 1.01
    res = block_residuals(pm, ["eq_va", "eq_fd"])
    assert abs(res["eq_va"][("BRD",)]) > 1e-3
    assert abs(res["eq_va"][("MLK",)]) <= 1e-10


def test_unknown_family_is_an_error() -> None:
    import pytest

    with pytest.raises(KeyError, match="eq_nope"):
        block_residuals(_value_added(), ["eq_nope"])
