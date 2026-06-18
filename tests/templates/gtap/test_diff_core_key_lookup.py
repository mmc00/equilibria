"""Comparator key-lookup tests for scripts/gtap/_diff_core.py.

Guards the prefix-stripping fix (commit 1f7a1f4) that recovered the 144
"missing" gtap7_3x3 shock cells (xd/xm/xcshr/zcons): GAMS keys carry set-type
prefixes (c_Food, a_Food) that Pyomo sets lack (Food), and the household
dimension (hhd) is a GAMS-only singleton.

The fix MUST be strictly monotonic: a raw key that already resolved must keep
resolving to the same value (an existing match can never flip to a miss/diff).
These tests are solver-free and run in milliseconds.
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts" / "gtap"))

from _diff_core import (  # noqa: E402
    _strip_pfx_keepcase,
    _value_or_zero,
    get_py_var_value,
)


class _FakeVar:
    """Mimics a Pyomo indexed Var enough for the lookups under test: dict-keyed,
    KeyError on miss, membership via `in`, and `var[key]` returns a plain float so
    both pyomo.value(var[key]) (get_py_var_value path) and the bare v[idx] branch
    (_value_or_zero path) resolve it."""

    def __init__(self, data):
        self._data = dict(data)

    def __getitem__(self, key):
        return self._data[key]

    def __contains__(self, key):
        return key in self._data

    def __iter__(self):
        return iter(self._data)


class _FakeModel:
    def __init__(self, **vars_):
        for k, v in vars_.items():
            setattr(self, k, v)


def test_strip_pfx_keepcase_preserves_case():
    assert _strip_pfx_keepcase("c_Food") == "Food"
    assert _strip_pfx_keepcase("a_Svces") == "Svces"
    assert _strip_pfx_keepcase("f_Land") == "Land"
    assert _strip_pfx_keepcase("r_USA") == "USA"
    # Only one prefix stripped; non-prefixed left untouched.
    assert _strip_pfx_keepcase("hhd") == "hhd"
    assert _strip_pfx_keepcase("Food") == "Food"


def test_value_or_zero_strips_gams_prefix():
    # Python set has bare names; GAMS key has c_/a_ prefixes.
    xda = _FakeVar({("USA", "Food", "Food"): 0.5423549999})
    m = _FakeModel(xda=xda)
    # GAMS-style key resolves via the prefix-stripped fallback.
    assert _value_or_zero(m, "xda", ("USA", "c_Food", "a_Food")) == pytest.approx(0.5423549999)


def test_value_or_zero_raw_key_tried_first_monotonic():
    # If the raw (already-correct) key exists, it must be returned unchanged even
    # when a stripped variant would also exist — proves monotonicity.
    xda = _FakeVar({
        ("USA", "c_Food", "a_Food"): 111.0,   # raw GAMS-style key present
        ("USA", "Food", "Food"): 999.0,       # stripped variant present too
    })
    m = _FakeModel(xda=xda)
    assert _value_or_zero(m, "xda", ("USA", "c_Food", "a_Food")) == 111.0


def test_value_or_zero_missing_var_returns_none():
    m = _FakeModel()
    assert _value_or_zero(m, "nonexistent", ("USA", "Food")) is None


def test_get_py_var_value_drops_hhd_and_strips_prefix():
    # GAMS xcshr(r, c_Food, hhd) -> Python xcshr(r, Food): drop hhd + strip c_.
    xcshr = _FakeVar({("USA", "Food"): 0.0594829340})
    m = _FakeModel(xcshr=xcshr)
    assert get_py_var_value(xcshr, ("USA", "c_Food", "hhd")) == pytest.approx(0.0594829340)


def test_get_py_var_value_exact_key_unchanged():
    # A direct, prefix-free key must still resolve (no regression for normal vars).
    pa = _FakeVar({("USA", "Food", "hhd"): 1.234})
    m = _FakeModel(pa=pa)
    assert get_py_var_value(pa, ("USA", "Food", "hhd")) == pytest.approx(1.234)
