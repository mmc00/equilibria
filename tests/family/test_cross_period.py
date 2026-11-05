"""Cross-period rows as recorded families against the Pyomo ``build_equations_fisher``.

Oracle: the multiperiod Pyomo model ``build_block_model(base_calibrated=False)`` built
here (ifSUB=1, capFix, the capture route's closure). Rows are matched by name; F is
``body - bound`` of the Pyomo row, J its numeric derivative over every Var in the row
(fixed or not), at the build point and at a perturbed one.
"""

from __future__ import annotations

import itertools
import sys
from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest

pytest.importorskip("jax")
pytest.importorskip("pyomo.environ")

from tests.family._data import load_params  # noqa: E402
from tests.family.test_prep_recipe import _params  # noqa: E402
from tests.family.test_squared_system import _route_closure  # noqa: E402

DATASETS = ("gtap7_3x3", "gtap7_10x7")
RTOL = 1e-10

PYOMO_FAMILIES = {
    "eq_rgdpmp",
    "eq_pgdpmp",
    "eq_pabs_base",
    "eq_pfact_base",
    "eq_pabs",
    "eq_pfact",
    "eq_pwfact",
    "eq_chisave",
}


@dataclass
class Case:
    dataset: str
    m: Any
    backend: Any  # cross-period recording
    sets: dict[str, list[str]]
    layout: Any
    system: Any
    values: dict[str, np.ndarray]  # dense levels at the build point
    rows: dict[str, tuple[Any, float]]  # Pyomo row name -> (constraint, bound)


def _key(idx: tuple):
    return idx if len(idx) > 1 else idx[0]


def _dense(m: Any, backend: Any) -> dict[str, np.ndarray]:
    from pyomo.environ import value

    out = {}
    for v, doms in backend.var_domains.items():
        comp = getattr(m, v)
        shape = tuple(len(backend.sets[d]) for d in doms)
        arr = np.zeros(shape)
        for pos, idx in zip(
            itertools.product(*(range(n) for n in shape)),
            itertools.product(*(backend.sets[d] for d in doms)),
            strict=True,
        ):
            k = _key(idx)
            if k in comp:
                arr[pos] = value(comp[k])
        out[v] = arr
    return out


def _layout(backend: Any, values: dict[str, np.ndarray], fixed=()):
    from equilibria.family.layout import VarLayout

    free = {v: np.full(a.shape, v not in fixed) for v, a in values.items()}
    return VarLayout.from_state(backend.sets, backend.var_domains, values, free)


@pytest.fixture(scope="module", params=DATASETS)
def case(request) -> Case:
    from equilibria.backends.family_backend import FamilyBackend  # noqa: F401
    from equilibria.family import cross_period
    from equilibria.family.recorder import RecordingModel
    from equilibria.templates.gtap.gtap_block_model import (
        build_block_model,
        compose_block_state,
    )

    ds = request.param
    load_params(ds)  # skip when the dataset is missing
    p = _params(ds)
    closure = _route_closure()
    rr = list(p.sets.r)[-1]
    m, _mp = build_block_model(p, p.sets, closure, rr, base_calibrated=False)
    model, _state = compose_block_state(p, p.sets, closure, rr)
    sp = RecordingModel.from_model(model)
    backend = cross_period.record_cross_period(
        p, rr, sets=sp.sets, var_domains=sp.var_domains
    )
    values = _dense(m, backend)
    layout = _layout(backend, values)
    rows = {}
    for fam, recs in backend.records.items():
        comp = getattr(m, fam)
        for idx, _ in recs:
            c = comp[_key(idx)]
            rows[c.name] = (c, float(c.upper))
    return Case(
        ds, m, backend, backend.sets, layout, backend.system(layout), values, rows
    )


def _perturbed(values: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(0)
    return {
        v: a * (1.0 + 1e-3 * rng.standard_normal(a.shape)) for v, a in values.items()
    }


def _set_pyomo(case: Case, values: dict[str, np.ndarray]) -> None:
    """Write dense levels into the Pyomo Vars (cells that exist) and unfix them all."""
    backend = case.backend
    for v, doms in backend.var_domains.items():
        comp = getattr(case.m, v)
        for pos, idx in zip(
            itertools.product(*(range(len(backend.sets[d])) for d in doms)),
            itertools.product(*(backend.sets[d] for d in doms)),
            strict=True,
        ):
            k = _key(idx)
            if k in comp:
                comp[k].unfix()
                comp[k].set_value(float(values[v][pos]), skip_validation=True)


def _pyomo_FJ(case: Case):
    """{row: F} and {(row, col): dF/dcol} for every cross-period row."""
    from pyomo.core.expr.calculus.derivatives import Modes, differentiate
    from pyomo.core.expr.visitor import identify_variables
    from pyomo.environ import value

    F, J = {}, {}
    for name, (c, bound) in case.rows.items():
        F[name] = float(value(c.body)) - bound
        vs = list(identify_variables(c.body, include_fixed=True))
        d = differentiate(c.body, wrt_list=vs, mode=Modes.reverse_numeric)
        for v, dv in zip(vs, d, strict=True):
            J[(name, v.name)] = J.get((name, v.name), 0.0) + float(dv)
    return F, J


def _family_FJ(case: Case, layout, system):
    z = layout.free_values
    f = system.F(z)
    jac = system.jac(z)
    names, cols = system.row_names, layout.col_names
    F = {names[k]: float(f[k]) for k in range(len(names))}
    J: dict[tuple, float] = {}
    for r, c, v in zip(jac.row, jac.col, jac.data, strict=True):
        J[(names[r], cols[c])] = J.get((names[r], cols[c]), 0.0) + float(v)
    pattern = {(names[r], cols[c]) for r, c in zip(jac.row, jac.col, strict=True)}
    return F, J, pattern


def test_same_rows_and_mq_keys(case):
    m = case.m
    recorded = set(case.backend.records)
    pyomo_mq = {n.removeprefix("eq_") for n in recorded if n.startswith("eq_mq_")}
    assert recorded - {f"eq_{n}" for n in pyomo_mq} == PYOMO_FAMILIES
    assert pyomo_mq == set(m._fisher_aggregates)
    # Every Pyomo row of those families is recorded, and nothing else is.
    for fam in recorded:
        got = {",".join(i) for i, _ in case.backend.records[fam]}
        want = {
            ",".join(map(str, k if isinstance(k, tuple) else (k,)))
            for k in getattr(m, fam)
        }
        assert got == want, fam

    from equilibria.family.cross_period import mq_keys

    keys = mq_keys(case.sets["r"], case.sets["t"])
    assert list(keys) == list(m._fisher_aggregates)  # Pyomo's creation order
    for name, (var, _defs) in m._fisher_aggregates.items():
        want = [k if isinstance(k, tuple) else (k,) for k in var]
        assert keys[name] == want, name


def _check(case: Case, layout, system, label: str, capsys):
    Fp, Jp = _pyomo_FJ(case)
    Ff, Jf, pattern = _family_FJ(case, layout, system)
    assert set(Ff) == set(Fp)
    x = dict(zip(layout.col_names, layout.free_values, strict=True))
    scale: dict[str, float] = {}
    for (r, c), v in Jp.items():
        scale[r] = scale.get(r, 0.0) + abs(v * x.get(c, 0.0))
    per_family: dict[str, list[float]] = {}  # family -> [max rel F, max rel J]
    for n, b in Fp.items():
        e = abs(Ff[n] - b) / max(abs(b), scale.get(n, 0.0), 1e-300)
        rec = per_family.setdefault(n.partition("[")[0], [0.0, 0.0])
        rec[0] = max(rec[0], e)
    # pattern: family ⊆ Pyomo incidence; every other Pyomo entry has derivative 0
    assert pattern <= set(Jp), (
        f"pattern outside Pyomo's: {sorted(pattern - set(Jp))[:5]}"
    )
    extra = {k: v for k, v in Jp.items() if k not in pattern and v != 0.0}
    assert not extra, (
        f"Pyomo entries outside the family pattern: {list(extra.items())[:5]}"
    )
    for (r, c), b in Jp.items():
        if (r, c) in pattern:
            e = abs(Jf[(r, c)] - b) / abs(b) if b else abs(Jf[(r, c)])
            rec = per_family[r.partition("[")[0]]
            rec[1] = max(rec[1], e)
    with capsys.disabled():
        for fam in sorted(per_family):
            print(
                f"[cross-period] {case.dataset} {label} {fam}: "
                f"max rel F {per_family[fam][0]:.2e}, max rel J {per_family[fam][1]:.2e}",
                file=sys.stderr,
            )
    for fam, (ef, ej) in per_family.items():
        assert ef <= RTOL, f"{fam}: F rel diff {ef:.2e}"
        assert ej <= RTOL, f"{fam}: J rel diff {ej:.2e}"


def test_F_and_J_at_the_build_point(case, capsys):
    _set_pyomo(case, case.values)
    _check(case, case.layout, case.system, "build", capsys)


def test_F_and_J_at_a_perturbed_point(case, capsys):
    pv = _perturbed(case.values)
    _set_pyomo(case, pv)
    layout = _layout(case.backend, pv)
    _check(case, layout, case.backend.system(layout), "perturbed", capsys)


def _mp_from_pyomo(case: Case):
    from equilibria.family.mp_arrays import MPArrays, VarArr

    out = {}
    for v in case.backend.var_domains:
        if v.startswith("mq_"):
            continue
        comp = getattr(case.m, v)
        keys = [k if isinstance(k, tuple) else (k,) for k in comp]
        val = np.array([comp[k].value for k in comp], dtype=float)
        z = np.zeros(len(keys))
        out[v] = VarArr(keys, val, z.astype(bool), z, z)
    return MPArrays(out, {})


def test_mq_values_are_the_seeded_aggregates(case):
    from pyomo.environ import value

    from equilibria.family.cross_period import mq_values

    # The model as built: the seeds Pyomo stamped at construction.
    _set_pyomo(case, case.values)
    mp = _mp_from_pyomo(case)
    got = mq_values(case.backend, mp)
    worst = 0.0
    for name, (var, defs) in case.m._fisher_aggregates.items():
        assert set(got[name]) == {k if isinstance(k, tuple) else (k,) for k in var}
        for k in var:
            key = k if isinstance(k, tuple) else (k,)
            want = float(var[k].value)
            assert float(value(defs[key])) == pytest.approx(want, rel=1e-12, abs=1e-12)
            worst = max(worst, abs(got[name][key] - want) / max(1.0, abs(want)))
    assert worst <= 1e-12, worst


def test_report_vars_are_slots_not_macros(case):
    """Under ifSUB ``pefob``/``pmcif`` are fixed Vars read by the rows: they are leaves
    of the recording and, fixed, become data slots with no Jacobian column."""
    leaves = {
        var
        for recs in case.backend.records.values()
        for _, res in recs
        for var, _ in _leaves(res)
    }
    assert {"pefob", "pmcif"} <= leaves
    layout = _layout(case.backend, case.values, fixed=("pefob", "pmcif"))
    assert (layout.slots["pefob"] >= layout.n_cols).all()
    assert (layout.slots["pmcif"] >= layout.n_cols).all()
    system = case.backend.system(layout)
    base = case.backend.system(case.layout)
    # Same F with the two Vars as data, and fewer columns.
    zfull = case.layout.free_values
    mask = np.ones(len(zfull), bool)
    names = case.layout.col_names
    mask &= np.array([not n.startswith(("pefob[", "pmcif[")) for n in names])
    np.testing.assert_allclose(
        system.F(zfull[mask]), base.F(zfull), rtol=1e-12, atol=1e-12
    )
    assert layout.n_cols == int(mask.sum())


def _leaves(res):
    from equilibria.family.recorder import residual_terms

    for _, t in residual_terms(res):
        if hasattr(t, "var_leaves"):
            yield from t.var_leaves()


def test_disk_cache_roundtrip(tmp_path):
    from equilibria.family import cross_period
    from equilibria.family.recorder import RecordingModel
    from equilibria.templates.gtap.gtap_block_model import compose_block_state

    ds = "gtap7_3x3"
    load_params(ds)
    p = _params(ds)
    closure = _route_closure()
    rr = list(p.sets.r)[-1]
    model, _ = compose_block_state(p, p.sets, closure, rr)
    sp = RecordingModel.from_model(model)
    path = cross_period.cache_path(p, closure, rr, tmp_path)
    assert path is not None and not path.exists()
    first = cross_period.load_or_record(
        p, closure, rr, tmp_path, sets=sp.sets, var_domains=sp.var_domains
    )
    assert path.exists()
    again = cross_period.load_or_record(
        p, closure, rr, tmp_path, sets=sp.sets, var_domains=sp.var_domains
    )
    assert again is not first
    assert list(again.records) == list(first.records)
    assert again.var_domains == first.var_domains and again.sets == first.sets
    values = {
        v: np.random.default_rng(1).uniform(
            0.5, 1.5, tuple(len(first.sets[d]) for d in doms)
        )
        for v, doms in first.var_domains.items()
    }
    layout = _layout(first, values)
    z = layout.free_values
    np.testing.assert_array_equal(first.system(layout).F(z), again.system(layout).F(z))
