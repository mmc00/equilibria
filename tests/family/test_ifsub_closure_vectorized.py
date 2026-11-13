"""The array form of ``_apply_ifsub_closure`` equals the cell-by-cell closure bitwise.

``tests/family/golden/ifsub_closure_*.npz`` snapshot the nine report variables (levels
and fixed flags) that the cell-by-cell code left in the family state, and the count the
closure returned. The array path must reproduce them with ``np.array_equal``.
"""

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("pyomo")

from tests.family._data import load_params, skip_unless_golden_env  # noqa: E402

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
    skip_unless_golden_env()
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


# --- perturbed states ---------------------------------------------------------------
OPS = ("pf", "fctts", "fcttx", "prdtx_rai", "p_rai", "amgm", "ptmg", "lambdamg")
OPS += ("pe", "tmarg", "imptx", "xw")
TAX = ("rtxs", "etax", "mtax", "chipm", "kappaf_activity", "kappaf")


def _snapshot(st):
    out = {}
    for n, c in st._components.items():
        if hasattr(c, "_vals"):
            out[n] = tuple(a.copy() for a in (c._vals, c._fx, c._lo, c._hi))
        else:
            out[n] = (c.data.copy(),)
    return out


def _restore(st, snap):
    for n, arrs in snap.items():
        c = st._components[n]
        if hasattr(c, "_vals"):
            for a, s in zip((c._vals, c._fx, c._lo, c._hi), arrs, strict=True):
                if isinstance(a, np.ndarray):
                    a[...] = s
        else:
            c.data[...] = arrs[0]


def _perturb(st, p, rng):
    """Random operands (zeros, negatives), pre-fixed cells, clamping bounds, taxes."""
    for name in OPS:
        c = st._components[name]
        arr = c.data if hasattr(c, "data") else c._vals
        arr[...] = rng.uniform(-2, 3, arr.shape) * (rng.random(arr.shape) > 0.2)
    for v in VARS:
        c = st._components[v]
        c._fx[...] = rng.random(c.shape) < 0.2
        c._lo[...] = np.where(
            rng.random(c.shape) < 0.3, rng.uniform(-1, 1, c.shape), -np.inf
        )
        c._hi[...] = np.where(
            rng.random(c.shape) < 0.3, rng.uniform(1, 3, c.shape), np.inf
        )
    r, i = list(p.sets.r), list(p.sets.i)
    for attr in TAX:
        d = getattr(p.taxes, attr, None)
        if d is None:
            if attr in ("etax", "mtax"):
                d = {(a, b): 0.0 for a in r for b in i}
            else:
                d = {(a, b, c): 1.0 for a in r for b in i for c in r}
            setattr(p.taxes, attr, d)
        for k in list(d):
            d[k] = float(rng.choice([0.0, rng.uniform(-0.5, 0.5)]))
        if attr == "chipm":
            for k in list(d):
                d[k] = float(rng.choice([0.0, 1e-14, rng.uniform(0.5, 1.5)]))
            for k in list(d)[:2]:
                d[k] = None


def _run(impl, st, p, snap):
    _restore(st, snap)
    try:
        err = None
        n = impl(st, p)
    except RuntimeError as exc:
        err, n = str(exc), None
    return err, n, _snapshot(st)


def _compare_paths(monkeypatch, seeds, break_labels=False, calls=None):
    import copy

    import equilibria.family.ifsub_arrays as ia
    import equilibria.templates.gtap.gtap_block_model as gbm

    p = load_params("gtap7_3x3")
    rr = list(p.sets.r)[-1]
    real = gbm._apply_ifsub_closure
    bad = []
    broken = {"on": False}
    real_fix = ia.fix_report_var

    def spy(state, params, var_name, *args):
        out = real_fix(state, params, var_name, *args)
        if calls is not None and broken["on"]:
            calls.append((var_name, out))
        return out

    def grab(st, params=None):
        base = _snapshot(st)
        saved = {
            a: copy.deepcopy(getattr(p.taxes, a)) if hasattr(p.taxes, a) else None
            for a in TAX
        }
        present = {a for a in TAX if hasattr(p.taxes, a)}
        for seed in seeds:
            _restore(st, base)
            _perturb(st, p, np.random.default_rng(seed))
            if break_labels:
                lab = st._components["tmarg"]._labels
                st._components["tmarg"]._labels = [list(reversed(lab[0])), *lab[1:]]
            pert = _snapshot(st)
            a = _run(gbm._apply_ifsub_closure_cells, st, p, pert)
            broken["on"] = break_labels
            b = _run(real, st, p, pert)
            broken["on"] = False
            if break_labels:
                st._components["tmarg"]._labels = lab
            if a[:2] != b[:2]:
                bad.append((seed, "error/count", a[:2], b[:2]))
            for n in a[2]:
                for x, y in zip(a[2][n], b[2][n], strict=True):
                    if not np.array_equal(x, y, equal_nan=True):
                        bad.append((seed, n, int((x != y).sum())))
        for attr, d in saved.items():
            if attr in present:
                setattr(p.taxes, attr, d)
            elif hasattr(p.taxes, attr):  # created by the perturbation: remove it
                delattr(p.taxes, attr)
        _restore(st, base)
        return real(st, params)

    with monkeypatch.context() as mp:
        mp.setattr(gbm, "_apply_ifsub_closure", grab)
        mp.setattr(ia, "fix_report_var", spy)
        gbm.compose_block_state(p, p.sets, closure(), rr)
    assert not bad, bad[:10]


def test_perturbed_states_cell_path_equals_array_path(monkeypatch):
    _compare_paths(monkeypatch, range(15))


def test_perturbed_states_with_unaligned_operand(monkeypatch):
    calls: list = []
    _compare_paths(monkeypatch, range(3), break_labels=True, calls=calls)
    # The fallback ran: with tmarg's labels reversed, the vectorised path declines
    # the variables that read it (it returns False and touches nothing).
    declined = {name for name, ok in calls if not ok}
    assert {"pmcif", "pm", "xwmg", "xmgm"} <= declined, calls
