"""``family_levels``: the ``model_levels`` shape (``scripts/gtap/compare_gempack_levels``)
read from ``MPArrays`` instead of a Pyomo model."""

from __future__ import annotations

import importlib.util

import numpy as np

from equilibria.family.altertax.levels import family_levels
from equilibria.family.mp_arrays import MPArrays
from equilibria.family.prep_capture import pairs
from tests.family._prep import ROOT, capture_dir


def _eq_vars() -> tuple[str, ...]:
    path = ROOT / "scripts" / "gtap" / "compare_gempack_levels.py"
    spec = importlib.util.spec_from_file_location("compare_gempack_levels", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.EQ_VARS


def test_family_levels_matches_model_levels_shape():
    eq_vars = _eq_vars()
    mp = MPArrays.from_photo(pairs(capture_dir("gtap7_3x3"))[0][0])
    out = family_levels(mp, eq_vars)
    assert set(out) == set(eq_vars)
    for v in eq_vars:
        va = mp.vars[v]
        assert len(out[v]) == len(va.keys) > 0
        for key, want in zip(va.keys, va.value, strict=True):
            assert key[-1] in {"base", "check", "shock"}
            got = out[v][tuple(key)]
            assert isinstance(got, float)
            assert got == float(want) or (np.isnan(got) and np.isnan(want))


def test_family_levels_scalar_variable_key_is_a_tuple():
    mp = MPArrays.from_photo(pairs(capture_dir("gtap7_3x3"))[0][0])
    out = family_levels(mp, ["pnum"])
    assert all(
        isinstance(k, tuple) and k[-1] in {"base", "check", "shock"}
        for k in out["pnum"]
    )
