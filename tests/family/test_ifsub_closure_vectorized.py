"""The array form of ``_apply_ifsub_closure`` equals the cell-by-cell closure bitwise.

``tests/family/golden/ifsub_closure_*.npz`` snapshot the nine report variables (levels
and fixed flags) that the cell-by-cell code left in the family state, and the count the
closure returned. The array path must reproduce them with ``np.array_equal``.
"""

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("pyomo")

from tests.family._data import load_params  # noqa: E402

VARS = ("pfa", "pfy", "pp_rai", "pwmg", "pefob", "pmcif", "pm", "xwmg", "xmgm")
GOLDEN = Path(__file__).parent / "golden"


def closure():
    from equilibria.templates.gtap.gtap_contract import GTAPClosureConfig

    # The closure scripts/family/budget_20x41.py composes with (ifSUB=1, capFix).
    return GTAPClosureConfig(
        name="base",
        closure_type="MCP",
        capital_mobility="sluggish",
        fix_endowments=False,
        fix_taxes=False,
        fix_technology=False,
        if_sub=True,
        savf_flag="capFix",
        numeraire="pnum",
        va_subsidy_basis="gempack",
    )


def _compose(monkeypatch, dataset, impl=None):
    import equilibria.templates.gtap.gtap_block_model as gbm

    p = load_params(dataset)
    rr = list(p.sets.r)[-1]
    calls = []
    real = gbm._apply_ifsub_closure
    target = impl or real

    def spy(pm, params=None):
        calls.append(target(pm, params))
        return calls[-1]

    with monkeypatch.context() as mp:
        mp.setattr(gbm, "_apply_ifsub_closure", spy)
        _, state = gbm.compose_block_state(p, p.sets, closure(), rr)
    return state, calls


@pytest.mark.parametrize("dataset", ["gtap7_3x3", "gtap7_10x7"])
def test_matches_golden(dataset, monkeypatch):
    gold = np.load(GOLDEN / f"ifsub_closure_{dataset}.npz")
    state, calls = _compose(monkeypatch, dataset)
    assert calls == gold["count"].tolist()
    for v in VARS:
        assert np.array_equal(state.values[v], gold["val_" + v], equal_nan=True), v
        assert np.array_equal(state.fixed[v], gold["fx_" + v]), v


@pytest.mark.parametrize("dataset", ["gtap7_3x3", "gtap7_10x7"])
def test_matches_cell_by_cell(dataset, monkeypatch):
    import equilibria.templates.gtap.gtap_block_model as gbm

    cells = getattr(gbm, "_apply_ifsub_closure_cells", None)
    if cells is None:
        pytest.skip("no separate cell-by-cell function yet")
    ref, ref_calls = _compose(monkeypatch, dataset, impl=cells)
    new, new_calls = _compose(monkeypatch, dataset)
    assert new_calls == ref_calls
    for name in new.values:
        assert np.array_equal(new.values[name], ref.values[name], equal_nan=True), name
        assert np.array_equal(new.fixed[name], ref.fixed[name]), name


def test_every_report_var_takes_the_array_path(monkeypatch):
    import equilibria.family.ifsub_arrays as arrays

    took = []
    real = arrays.fix_report_var

    def spy(state, params, name, *a):
        ok = real(state, params, name, *a)
        took.append((name, ok))
        return ok

    monkeypatch.setattr(arrays, "fix_report_var", spy)
    _compose(monkeypatch, "gtap7_3x3")
    assert sorted(took) == sorted((v, True) for v in VARS)


def test_failing_macro_raises_with_cell_names(monkeypatch):
    import equilibria.templates.gtap.gtap_block_model as gbm

    p = load_params("gtap7_3x3")
    rr = list(p.sets.r)[-1]
    real = gbm._apply_ifsub_closure
    seen = []

    def poison(pm, params=None):
        # lambdamg + 1e-12 == 0 makes the scalar macro divide by zero.
        pm.lambdamg._vals.flat[0] = -1e-12
        for impl in (gbm._apply_ifsub_closure_cells, real):
            try:
                impl(pm, params)
            except RuntimeError as exc:
                seen.append(str(exc))
            # the first call fixed the cells; free them again for the second
            for v in VARS:
                getattr(pm, v)._fx[...] = False
        return 0

    monkeypatch.setattr(gbm, "_apply_ifsub_closure", poison)
    gbm.compose_block_state(p, p.sets, closure(), rr)
    assert len(seen) == 2 and seen[0] == seen[1]
    assert "macro fallido: xmgm" in seen[0] or "macro fallido: pwmg" in seen[0]
