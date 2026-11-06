"""Cross-period rows (Fisher / Tornqvist indices and their ``mq_*`` aggregates) as recorded families.

``GTAPMultiPeriodModel.build_equations_fisher`` writes these rows over the multiperiod
Pyomo model. The rules below mirror it operation by operation (same guards, operand
order and ``xscale`` filter) over a Pyomo-like ``m`` with period-indexed access
(``m.pa[r, i, a, t]``), so ``RecordingModel`` can run them on multiperiod domains
(each Var of the single-period model with a trailing ``t``) and the result is a
``FamilyBackend``-compatible recording (``records``, ``var_domains``, ``param_domains``,
``sets``). The Pyomo code stays as it is; ``tests/family/test_cross_period.py`` ties the
two copies row by row.

The named aggregates ``mq_{gdp,abs,factr,factw}_{bb,bs,sb,ss}`` are Vars of their own
with a defining row ``eq_mq_*``. ``mq_keys`` gives their cells (Pyomo's, in Pyomo's
order) and ``mq_values`` their value at a state, replacing the seed Pyomo stamps at
construction and ``refresh_fisher_aggregates``.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from equilibria.family.recorder import (
    Node,
    RecordingModel,
    Residual,
    intrinsic,
)

sqrt = intrinsic("sqrt")
log = intrinsic("log")
exp = intrinsic("exp")

# Single-period Vars the cross-period rows read or define; each gets a trailing ``t``.
CROSS_VARS: tuple[str, ...] = (
    "pa",
    "xaa",
    "pefob",
    "pmcif",
    "xw",
    "pf",
    "xf",
    "pabs",
    "pfact",
    "pwfact",
    "rgdpmp",
    "gdpmp",
    "pgdpmp",
    "pi",
    "xiagg",
    "kstock",
    "rsav",
    "chiSave",
)
# Names of the aggregates, in the order Pyomo creates them (first use in the rules),
# with the non-period part of their index.
_AGG_ORDER: tuple[tuple[str, str], ...] = (
    ("gdp", "bs"),
    ("gdp", "sb"),
    ("abs", "bb"),
    ("abs", "sb"),
    ("abs", "ss"),
    ("abs", "bs"),
    ("factr", "bb"),
    ("factr", "sb"),
    ("factr", "ss"),
    ("factr", "bs"),
    ("factw", "bb"),
    ("factw", "sb"),
    ("factw", "ss"),
    ("factw", "bs"),
)
_AGG_DOMAINS = {
    "gdp": ("r", "t"),
    "abs": ("r", "t"),
    "factr": ("r", "t"),
    "factw": ("t",),
}
BASE = "base"


def _rest_keys(prefix: str, regions: list[str]) -> list[tuple]:
    return [(r,) for r in regions] if prefix != "factw" else [()]


def mq_keys(
    regions: list[str], periods: list[str] | tuple[str, ...]
) -> dict[str, list]:
    """``{mq_<prefix>_<slot>: [cell, ...]}`` exactly as Pyomo declares them: the
    ``(base, base)`` cross belongs to the base period only, the others to the non-base
    ones. Cells are tuples ``(*rest, t)``."""
    nonbase = [t for t in periods if t != BASE]
    out = {}
    for prefix, slot in _AGG_ORDER:
        ts = [BASE] if slot == "bb" else nonbase
        out[f"mq_{prefix}_{slot}"] = [
            (*rest, t) for rest in _rest_keys(prefix, regions) for t in ts
        ]
    return out


class _Aggregates:
    """Python side of ``_named``: the Var cell standing for a cross sum, plus the sum."""

    def __init__(self, m: Any):
        self.m = m
        self.defs: dict[str, dict[tuple, Any]] = {}

    def named(self, prefix: str, tp: str, tq: str, rest: tuple, expr: Any) -> Any:
        slot = ("b" if tp == BASE else "s") + ("b" if tq == BASE else "s")
        name = f"mq_{prefix}_{slot}"
        t_own = tp if tp != BASE else (tq if tq != BASE else BASE)
        idx = (*rest, t_own)
        self.defs.setdefault(name, {})[idx] = expr
        return getattr(self.m, name)[idx]


def _fd() -> tuple[str, ...]:
    from equilibria.blocks.gtap.agents import (
        GTAP_GOVERNMENT_AGENT,
        GTAP_HOUSEHOLD_AGENT,
        GTAP_INVESTMENT_AGENT,
        GTAP_MARGIN_AGENT,
    )

    return (
        GTAP_HOUSEHOLD_AGENT,
        GTAP_GOVERNMENT_AGENT,
        GTAP_INVESTMENT_AGENT,
        GTAP_MARGIN_AGENT,
    )


def _mqgdp(m: Any, fd: tuple[str, ...], tp: str, tq: str, r: str):
    """Fisher cross-product of absorption + net exports."""
    ab = sum(m.pa[r, i, a, tp] * m.xaa[r, i, a, tq] for i in m.i for a in fd)
    tr = sum(
        m.pefob[r, i, rp, tp] * m.xw[r, i, rp, tq]
        - m.pmcif[rp, i, r, tp] * m.xw[rp, i, r, tq]
        for i in m.i
        for rp in m.r
    )
    return ab + tr


def _mqabs_cross(m: Any, fd: tuple[str, ...], tp: str, tq: str, r: str):
    return sum(m.pa[r, i, a, tp] * m.xaa[r, i, a, tq] for i in m.i for a in fd)


def _mqfactr_cross(m: Any, tp: str, tq: str, r: str):
    return sum(
        m.pf[r, f, a, tp] * m.xf[r, f, a, tq] / m.xscale[r, a]
        for f in m.f
        for a in m.a
        if m.xscale[r, a] > 1e-12
    )


def _mqfactw_cross(m: Any, tp: str, tq: str):
    return sum(
        m.pf[r, f, a, tp] * m.xf[r, f, a, tq] / m.xscale[r, a]
        for r in m.r
        for f in m.f
        for a in m.a
        if m.xscale[r, a] > 1e-12
    )


def _pos(arg):
    """Smooth positive guard: (arg + sqrt(arg^2 + 1e-8)) / 2."""
    return (arg + sqrt(arg * arg + 1e-8)) * 0.5


def rgdpmp_rule(m: Any, agg: _Aggregates, fd, r: str, t: str):
    if t == BASE:
        return m.rgdpmp[r, BASE] == m.gdpmp[r, BASE]
    mq_base_t = agg.named("gdp", BASE, t, (r,), _mqgdp(m, fd, BASE, t, r))
    mq_t_base = agg.named("gdp", t, BASE, (r,), _mqgdp(m, fd, t, BASE, r))
    arg = (m.gdpmp[r, t] / m.gdpmp[r, BASE]) * (mq_base_t / (mq_t_base + 1e-12))
    arg_pos = _pos(arg)
    return m.rgdpmp[r, t] == m.rgdpmp[r, BASE] * sqrt(arg_pos + 1e-12)


def pgdpmp_rule(m: Any, r: str, t: str):
    return m.pgdpmp[r, t] * m.rgdpmp[r, t] == m.gdpmp[r, t]


def pabs_rule(m: Any, agg: _Aggregates, fd, r: str, t: str):
    mq_bb = agg.named("abs", BASE, BASE, (r,), _mqabs_cross(m, fd, BASE, BASE, r))
    mq_tb = agg.named("abs", t, BASE, (r,), _mqabs_cross(m, fd, t, BASE, r))
    mq_tt = agg.named("abs", t, t, (r,), _mqabs_cross(m, fd, t, t, r))
    mq_bt = agg.named("abs", BASE, t, (r,), _mqabs_cross(m, fd, BASE, t, r))
    arg = (mq_tb / (mq_bb + 1e-12)) * (mq_tt / (mq_bt + 1e-12))
    arg_pos = _pos(arg)
    return m.pabs[r, t] == m.pabs[r, BASE] * sqrt(arg_pos + 1e-12)


def pfact_rule(m: Any, agg: _Aggregates, r: str, t: str):
    m_bb = agg.named("factr", BASE, BASE, (r,), _mqfactr_cross(m, BASE, BASE, r))
    m_sb = agg.named("factr", t, BASE, (r,), _mqfactr_cross(m, t, BASE, r))
    m_ss = agg.named("factr", t, t, (r,), _mqfactr_cross(m, t, t, r))
    m_bs = agg.named("factr", BASE, t, (r,), _mqfactr_cross(m, BASE, t, r))
    arg = (m_sb / (m_bb + 1e-12)) * (m_ss / (m_bs + 1e-12))
    arg_pos = _pos(arg)
    return m.pfact[r, t] == sqrt(arg_pos + 1e-12)


def pwfact_rule(m: Any, agg: _Aggregates, t: str):
    m_bb = agg.named("factw", BASE, BASE, (), _mqfactw_cross(m, BASE, BASE))
    m_sb = agg.named("factw", t, BASE, (), _mqfactw_cross(m, t, BASE))
    m_ss = agg.named("factw", t, t, (), _mqfactw_cross(m, t, t))
    m_bs = agg.named("factw", BASE, t, (), _mqfactw_cross(m, BASE, t))
    arg = (m_sb / (m_bb + 1e-12)) * (m_ss / (m_bs + 1e-12))
    arg_pos = _pos(arg)
    return m.pwfact[t] == sqrt(arg_pos + 1e-12)


def _shares(m: Any, regions: list[str], t: str):
    netinv = {
        r: m.pi[r, t] * (m.xiagg[r, t] - m.depr[r] * m.kstock[r, t]) for r in regions
    }
    inv_tot = sum(netinv.values())
    sav_tot = sum(m.rsav[r, t] for r in regions)
    return (
        {r: netinv[r] / inv_tot for r in regions},
        {r: m.rsav[r, t] / sav_tot for r in regions},
    )


def chisave_rule(m: Any, regions: list[str], t: str):
    if t == BASE:
        numer = sum(m.invwgt[r] * m.pi[r, t] for r in regions)
        denom = sum(m.savwgt[r] * m.pi[r, t] for r in regions)
        return m.chiSave[t] * denom - numer == 0
    wi0, ws0 = _shares(m, regions, BASE)
    wi1, ws1 = _shares(m, regions, t)
    dlog = sum(
        0.5 * (wi0[r] + wi1[r] - ws0[r] - ws1[r]) * log(m.pi[r, t] / m.pi[r, BASE])
        for r in regions
    )
    return m.chiSave[t] == m.chiSave[BASE] * exp(dlog)


def build_rows(m: Any) -> dict[str, list[tuple[tuple, Residual]]]:
    """Every cross-period row, ``{family: [(index, Residual), ...]}``, in the order
    ``build_equations_fisher`` declares them (which is also the order in which it
    creates the aggregates)."""
    fd = _fd()
    regions = list(m.r)
    periods = list(m.t)
    nonbase = [t for t in periods if t != BASE]
    agg = _Aggregates(m)
    rows: dict[str, list[tuple[tuple, Residual]]] = {}

    def add(name: str, idx: tuple, res: Any) -> None:
        rows.setdefault(name, []).append((idx, res))

    for r in regions:
        for t in periods:
            add("eq_rgdpmp", (r, t), rgdpmp_rule(m, agg, fd, r, t))
    for r in regions:
        for t in periods:
            add("eq_pgdpmp", (r, t), pgdpmp_rule(m, r, t))
    for r in regions:
        add("eq_pabs_base", (r, BASE), m.pabs[r, BASE] == 1.0)
    for r in regions:
        add("eq_pfact_base", (r, BASE), m.pfact[r, BASE] == 1.0)
    for r in regions:
        for t in nonbase:
            add("eq_pabs", (r, t), pabs_rule(m, agg, fd, r, t))
    for r in regions:
        for t in nonbase:
            add("eq_pfact", (r, t), pfact_rule(m, agg, r, t))
    for t in nonbase:
        add("eq_pwfact", (t,), pwfact_rule(m, agg, t))
    for t in periods:
        add("eq_chisave", (t,), chisave_rule(m, regions, t))

    # Defining rows of the aggregates, once every rule above has registered the
    # expression each cell stands for.
    keys = mq_keys(regions, periods)
    for name, defs in agg.defs.items():
        cells = keys[name]
        assert set(defs) == set(cells), (
            f"{name}: {len(cells)} cells but {len(defs)} defining rows"
        )
        var = getattr(m, name)
        for idx in cells:
            add("eq_" + name, idx, var[idx] == defs[idx])
    assert list(agg.defs) == [n for n in keys if n in agg.defs]
    return rows


def _params_arrays(params: Any, residual_region: str) -> dict[str, tuple]:
    """``xscale`` (clamped as Pyomo clamps it) and the investment/saving weights, from
    the same functions ``build_equations_fisher`` calls."""
    from equilibria.blocks.gtap._derived_params import (
        demand_income_params,
        xscale_data,
    )

    sets = params.sets
    regions, acts = list(sets.r), list(sets.a)
    xs = xscale_data(params, sets)
    xscale = np.ones((len(regions), len(acts)))
    for i, r in enumerate(regions):
        for j, a in enumerate(acts):
            try:
                xscale[i, j] = max(float(xs[(r, a)]), 1e-12)
            except (KeyError, TypeError, ValueError):
                xscale[i, j] = 1.0
    di = demand_income_params(params, sets, residual_region=residual_region)

    def per_region(name: str) -> np.ndarray:
        return np.array([float(di[name].get((r,), 0.0)) for r in regions])

    return {
        "xscale": (("r", "a"), xscale),
        "depr": (("r",), per_region("depr")),
        "invwgt": (("r",), per_region("invwgt")),
        "savwgt": (("r",), per_region("savwgt")),
    }


def cross_var_domains(sp_var_domains: dict[str, tuple[str, ...]]) -> dict[str, tuple]:
    """Multiperiod domains of every Var the cross-period rows touch."""
    out = {}
    for v in CROSS_VARS:
        if v not in sp_var_domains:
            raise KeyError(f"single-period variable {v!r} missing")
        out[v] = (*sp_var_domains[v], "t")
    for prefix, slot in _AGG_ORDER:
        out[f"mq_{prefix}_{slot}"] = _AGG_DOMAINS[prefix]
    return out


def record_cross_period(
    params: Any,
    residual_region: str,
    *,
    sets: dict[str, list[str]],
    var_domains: dict[str, tuple[str, ...]],
) -> Any:
    """Record the cross-period rows: a ``FamilyBackend`` holding them.

    ``sets`` and ``var_domains`` are the single-period backend's (``FamilyBackend.sets`` /
    ``.var_domains``); the multiperiod ones are derived from them."""
    from equilibria.backends.family_backend import FamilyBackend
    from equilibria.blocks.gtap.periods import PERIODS

    mp_sets = {**{k: list(v) for k, v in sets.items()}, "t": list(PERIODS)}
    doms = cross_var_domains(var_domains)
    variables = {
        v: (d, np.zeros(tuple(len(mp_sets[s]) for s in d))) for v, d in doms.items()
    }
    rec = RecordingModel(mp_sets, _params_arrays(params, residual_region), variables)
    backend = FamilyBackend()
    backend.records = build_rows(rec)
    backend.var_domains = dict(rec.var_domains)
    backend.param_domains = dict(rec.param_domains)
    backend.sets = rec.sets
    return backend


def cache_path(
    params: Any, closure: Any, residual_region: str | None, root: Path
) -> Path | None:
    """Next to the single-period records (same key; its digest covers this module)."""
    from equilibria.family.disk_cache import cache_key

    key = cache_key(params, closure, residual_region)
    return None if key is None else Path(root) / key / "cross_records.pkl"


def load_or_record(
    params: Any,
    closure: Any,
    residual_region: str,
    root: Path,
    *,
    sets: dict[str, list[str]],
    var_domains: dict[str, tuple[str, ...]],
) -> Any:
    """The cross-period backend from the disk cache, recording (and saving) on a miss."""
    from equilibria.backends.family_backend import FamilyBackend
    from equilibria.family.phases import phase

    with phase("cross_cache_key"):
        path = cache_path(params, closure, residual_region, root)
    if path is not None and path.exists():
        with phase("cross_records_load"):
            return FamilyBackend.load(path)
    with phase("cross_records_build"):
        backend = record_cross_period(
            params, residual_region, sets=sets, var_domains=var_domains
        )
        if path is not None:
            backend.save(path)
    return backend


def _value(x: Any, leaf: Callable[[str, int], float]) -> float:
    """``Node.evaluate`` with the leaf levels supplied by ``leaf(var, flat)``."""
    if not isinstance(x, Node):
        return float(x)
    op, a = x.op, x.args
    if op == "v":
        return leaf(a[0], a[1])
    if op == "add":
        return float(sum(_value(c, leaf) for c in a[: x._n]))
    if op == "mul":
        return _value(a[0], leaf) * _value(a[1], leaf)
    if op == "div":
        return _value(a[0], leaf) / _value(a[1], leaf)
    if op == "neg":
        return -_value(a[0], leaf)
    raise NotImplementedError(f"mq_values: {op!r} in an aggregate definition")


def mq_values(backend: Any, mp: Any) -> dict[str, dict[tuple, float]]:
    """``{mq_*: {cell: value}}`` at the levels of ``mp`` (an ``MPArrays``): the value of
    the sum each aggregate stands for. ``backend`` is the cross-period recording."""
    sets = backend.sets
    pos = {k: {e: i for i, e in enumerate(v)} for k, v in sets.items()}
    dense: dict[str, np.ndarray] = {}

    def level(var: str, flat: int) -> float:
        arr = dense.get(var)
        if arr is None:
            doms = backend.var_domains[var]
            shape = tuple(len(sets[d]) for d in doms)
            arr = np.full(shape, np.nan).ravel()
            strides = np.cumprod((1, *shape[:0:-1]))[::-1]
            va = mp.var(var)
            for k, v in zip(va.keys, va.value, strict=True):
                f = sum(pos[d][e] * s for d, e, s in zip(doms, k, strides, strict=True))
                arr[f] = v
            dense[var] = arr
        return float(arr[flat])

    out: dict[str, dict[tuple, float]] = {}
    for fam, recs in backend.records.items():
        if not fam.startswith("eq_mq_"):
            continue
        name = fam[3:]
        out[name] = {idx: _value(res.rhs, level) for idx, res in recs}
    return out


def refresh_aggregates(backend: Any, mp: Any) -> int:
    """Write ``mq_values`` into the ``mq_*`` cells of ``mp`` (in place): the defined
    sums at the levels just installed, as ``refresh_fisher_aggregates`` leaves them.
    Returns the number of cells written."""
    n = 0
    for name, cells in mq_values(backend, mp).items():
        var = mp.var(name)
        for key, val in cells.items():
            var.value[var.pos(key)] = val
            n += 1
    return n
