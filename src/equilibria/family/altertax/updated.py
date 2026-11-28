"""``updated.har`` of ALTERTAX on the family route: the updated GTAP database.

GEMPACK's ``gtapv7.tab`` ends every run with the ``Update`` statements that turn each
value coefficient into ``price * quantity`` at the post-shock solution (the ``.upd`` file,
``altax_updated.har``). This module computes the same 33 headers from the solved
``MPArrays`` of an :class:`AltertaxRun` (shock period), in GEMPACK units: the model's
quantity scaling (``inScale`` 1e-6 and, for activity agents, ``xscale[r,a]``) is undone, so
a header is ``price * quantity / (inScale * xscale)``. At the ``check`` period (which
reproduces the base) the same builders give back the input ``basedata.har``.

Mapping (equilibria variables; ``gtapv7.tab`` line of each ``Update``; ``d`` = scaling
divisor, ``pdp = (1 + dintx) * pd`` and ``pmp = (1 + mintx) * pmt`` as in
``blocks/gtap/trade_armington_bilateral.py``):

=======  =========================================  =========
header   value                                      tab line
=======  =========================================  =========
VDFB     pd[r,c] * xda[r,c,a]                       277
VMFB     pmt[r,c] * xma[r,c,a]                      287
VDFP     pdp[r,c,a] * xda[r,c,a]                    297
VMFP     pmp[r,c,a] * xma[r,c,a]                    307
EVOS     pfy[r,e,a] * xf[r,e,a]                     321
EVFB     pf[r,e,a] * xf[r,e,a]                      331
EVFP     pfa[r,e,a] * xf[r,e,a]                     341
MAKS     p_rai[r,a,c] * x[r,a,c]                    358
MAKB     p_rai[r,a,c] * (1+prdtx_rai) * x[r,a,c]    368
VDGB/VMGB/VDGP/VMGP   as VDFB..VMFP, agent gov     383-413
VDPB/VMPB/VDPP/VMPP   as VDFB..VMFP, agent hhd     428-458
VDIB/VMIB/VDIP/VMIP   as VDFB..VMFP, agent inv     473-503
SAVE     rsav[r]                                    519
VDEP     rate[r] * pi[r] * kstock[r]                535
VXSB     pd[s,c] * xw[s,c,d]                        555
VFOB     pefob[s,c,d] * xw[s,c,d]                   566
VCIF     pmcif[s,c,d] * xw[s,c,d]                   577
VMSB     pm[s,c,d] * xw[s,c,d]                      588
VST      pd[r,m] * xda[r,m,tmg]                     603
DPSM     DPARSUM: constant, see ``_dpsm``           1479
VTWR     ptmg[m] * amgm * tmarg * xw[s,c,d]        1883
VKB      pi[r] * kstock[r]                          2377
=======  =========================================  =========

``DVER`` is copied from the input ``basedata.har``; ``XXHS`` is the history text.
"""

from __future__ import annotations

import dataclasses
import weakref
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from equilibria.babel.har.reader import read_har
from equilibria.babel.har.symbols import HeaderArray
from equilibria.babel.har.writer import write_har
from equilibria.family.altertax.run import AltertaxRun

IN_SCALE = 1.0e-6  # the model's quantity scale (``inScale``), not stored as a parameter

# (header, coefficient name, long name, sets) of the GEMPACK ``.upd`` (as in
# ``altax_updated.har``).
_META: tuple[tuple[str, str, str, tuple[str, ...]], ...] = (
    ("DVER", "VERNUM", "Format of GTAP data", ()),
    ("VDFB", "VDFB", " domestic purchases, by firms, at basic prices", ("COMM", "ACTS", "REG")),
    ("VMFB", "VMFB", " imported purchases, by firms, at basic prices", ("COMM", "ACTS", "REG")),
    ("VDFP", "VDFP", " domestic purchases, by firms, at producers prices", ("COMM", "ACTS", "REG")),
    ("VMFP", "VMFP", " imported purchases, by firms, at producers prices", ("COMM", "ACTS", "REG")),
    ("EVOS", "EVOS", " value of endowment supply in region r, net of direct tax", ("ENDW", "ACTS", "REG")),
    ("EVFB", "EVFB", " primary factor purchases, by firms, at basic prices", ("ENDW", "ACTS", "REG")),
    ("EVFP", "EVFP", " primary factor purchases, at producers prices", ("ENDW", "ACTS", "REG")),
    ("MAKS", "MAKES", " MAKE matrix (before output tax/subsidy)", ("COMM", "ACTS", "REG")),
    ("MAKB", "MAKEB", " MAKE matrix at basic prices", ("COMM", "ACTS", "REG")),
    ("VDGB", "VDGB", " domestic purchases, by government, at basic prices", ("COMM", "REG")),
    ("VMGB", "VMGB", " import purchases, by government, at basic prices", ("COMM", "REG")),
    ("VDGP", "VDGP", " domestic purchases, by government, at producers prices", ("COMM", "REG")),
    ("VMGP", "VMGP", " import purchases, by government, at producers prices", ("COMM", "REG")),
    ("VDPB", "VDPB", " domestic purchases, by households, at basic prices", ("COMM", "REG")),
    ("VMPB", "VMPB", " import purchases, by households, at basic prices", ("COMM", "REG")),
    ("VDPP", "VDPP", " domestic purchases, by households, at producers prices", ("COMM", "REG")),
    ("VMPP", "VMPP", " import purchases, by households, at producers prices", ("COMM", "REG")),
    ("VDIB", "VDIB", " domestic purchases, by investment, at basic prices", ("COMM", "REG")),
    ("VMIB", "VMIB", " import purchases, by investment, at basic prices", ("COMM", "REG")),
    ("VDIP", "VDIP", " domestic purchases, by investment, at producers prices", ("COMM", "REG")),
    ("VMIP", "VMIP", " import purchases, by investment, at producers prices", ("COMM", "REG")),
    ("SAVE", "SAVE", "Savings - Net Expenditure at Agents' Prices", ("REG",)),
    ("VDEP", "VDEP", "Capital Stock - Value of Depreciation", ("REG",)),
    ("VXSB", "VXSB", " non-margin exports, at basic prices", ("COMM", "REG", "REG")),
    ("VFOB", "VFOB", " non-margin exports, at FOB prices", ("COMM", "REG", "REG")),
    ("VCIF", "VCIF", " imports, at CIF prices", ("COMM", "REG", "REG")),
    ("VMSB", "VMSB", " imports, at basic prices", ("COMM", "REG", "REG")),
    ("VST", "VST", " margin exports", ("MARG", "REG")),
    ("DPSM", "DPARSUM", "Sum of Distribution Parameters in Household Demand System", ("REG",)),
    ("VTWR", "VTMFSD", " margins by margin commodity", ("MARG", "COMM", "REG", "REG")),
    ("VKB", "VKB", "Capital Stock - Value at Beginning-of-Period", ("REG",)),
)  # fmt: skip
_XXHS = ("XXHS", "XXHS", "HISTORY OF THIS HEADER ARRAY FILE", ())

UPDATED_HEADERS: tuple[str, ...] = (*(m[0] for m in _META), "XXHS")


class _Period(dict):
    """Levels of one period of the arrays as ``{var: {cell without period: value}}``.

    A ``dict`` of the variables read so far: ``P[var]`` is a plain lookup (millions of
    them while the headers are built), and ``__missing__`` reads a new one."""

    def __init__(self, run: AltertaxRun, period: str) -> None:
        from equilibria.blocks.gtap._derived_params import xscale_data

        mp = run.result.mp
        self.p = run.params
        self.xscale = dict(xscale_data(run.params, run.params.sets))
        self.period = period
        from equilibria.family.post_solve import mp_margin_params

        margin = mp_margin_params(run.params, run.params.sets)
        self.amgm = margin["amgm"][1]
        self.tmarg = margin["tmarg"][1]
        from equilibria.blocks.gtap._derived_params import fctts_data

        self.fctts = fctts_data(run.params, run.params.sets)
        self._vars = mp.vars
        self.trade: dict[tuple[str, str, str], dict[str, float]] = {}
        self.dense: dict[tuple, np.ndarray] = {}  # see _dense

    def __missing__(self, name: str) -> dict[tuple[str, ...], float]:
        v = self._vars[name]
        hit = self[name] = {
            tuple(k[:-1]): float(x)
            for k, x in zip(v.keys, v.value, strict=True)
            if k[-1] == self.period
        }
        return hit

    def unscale(self, r: str, agent: str) -> float:
        """Divisor that takes a model quantity of ``agent`` in ``r`` to data units."""
        return IN_SCALE * self.xscale.get((r, agent), 1.0)


def _g(table: dict, key: tuple[str, ...]) -> float:
    return table.get(key, 0.0)


def _firm(kind: str) -> Callable[[_Period, str, str, str], float]:
    """``VDFB``/``VMFB``/``VDFP``/``VMFP`` of (commodity c, activity a, region r)."""

    def f(P: _Period, c: str, a: str, r: str) -> float:
        dom = kind[1] == "D"
        q = _g(P["xda" if dom else "xma"], (r, c, a))
        price = _g(P["pd" if dom else "pmt"], (r, c))
        if kind[3] == "P":  # producer price: the agent's domestic/import tax wedge
            tax = P["dintx" if dom else "mintx"]
            price *= 1.0 + _g(tax, (r, c, a))
        return price * q / P.unscale(r, a)

    return f


def _final(agent: str, kind: str) -> Callable[[_Period, str, str], float]:
    """``VD?B``/``VM?B``/``VD?P``/``VM?P`` of (commodity c, region r), fixed agent."""
    dom = kind[1] == "D"

    def f(P: _Period, c: str, r: str) -> float:
        q = _g(P["xda" if dom else "xma"], (r, c, agent))
        price = _g(P["pd" if dom else "pmt"], (r, c))
        if kind[3] == "P":
            price *= 1.0 + _g(P["dintx" if dom else "mintx"], (r, c, agent))
        return price * q / P.unscale(r, agent)

    return f


def _factor_price(P: _Period, pvar: str, r: str, e: str, a: str) -> float:
    """``pf`` is the basic price (``EVFB``). ``pfa`` (producer price) and ``pfy`` (after
    income tax) are report vars that ``post_solve`` rewrites only at the shock period, so
    they are rebuilt from the live ``pf`` with the macros ``M_PFA = pf*(1+fctts+fcttx)``
    and ``M_PFY = pf*(1-kappaf)`` (``_ifsub_macros``)."""
    from equilibria.blocks.gtap import _ifsub_macros as mac

    pf = _g(P["pf"], (r, e, a))
    if pvar == "pf":
        return pf
    if pvar == "pfa":
        return pf * (1.0 + P.fctts.get((r, e, a), 0.0) + _g(P["fcttx"], (r, e, a)))
    return pf * (1.0 - mac.kappaf_value(P.p, r, e, a))


def _factor(pvar: str) -> Callable[[_Period, str, str, str], float]:
    def f(P: _Period, e: str, a: str, r: str) -> float:
        price = _factor_price(P, pvar, r, e, a)
        return price * _g(P["xf"], (r, e, a)) / P.unscale(r, a)

    return f


def _make(basic: bool) -> Callable[[_Period, str, str, str], float]:
    """``MAKES`` (supplier price ``p_rai``) and ``MAKEB`` (basic price, output tax
    inclusive: ``p_rai * (1 + prdtx_rai)``, the macro of ``ifsub_arrays.py``). ``pp_rai``
    is not read: it is a report var that ``post_solve`` overwrites only at the shock
    period (stale at check in one cell, driver-parity behaviour). ``x`` carries
    ``inScale`` only, not ``xscale``."""

    def f(P: _Period, c: str, a: str, r: str) -> float:
        price = _g(P["p_rai"], (r, a, c))
        if basic:
            price *= 1.0 + _g(P["prdtx_rai"], (r, a, c))
        return price * _g(P["x"], (r, a, c)) / IN_SCALE

    return f


def _trade_prices(P: _Period, s: str, c: str, d: str) -> dict[str, float]:
    """``pefob``/``pmcif``/``pm`` of exporter ``s``, commodity ``c``, importer ``d`` from
    the macros of ``_ifsub_macros`` (``M_PEFOB``, ``M_PWMG``, ``M_PMCIF``, ``M_PM``),
    on the live ``pe``/``ptmg``/``imptx``. The report vars ``pefob``/``pmcif``/``pm``
    are not read: ``post_solve`` rewrites them only at the shock period, so at check they
    keep their benchmark seed and shock-minus-check would mix two definitions.
    Computed once per cell of ``P`` (``VFOB``, ``VCIF``, ``VMSB`` and the decomposition
    all ask for it); callers must not change the dict."""
    hit = P.trade.get((s, c, d))
    if hit is None:
        hit = P.trade[(s, c, d)] = _trade_prices_cell(P, s, c, d)
    return hit


def _trade_prices_cell(P: _Period, s: str, c: str, d: str) -> dict[str, float]:
    from equilibria.blocks.gtap import _ifsub_macros as mac

    p = P.p
    pefob = (
        1.0 + mac._safe(p, "taxes.rtxs", (s, c, d)) + mac.etax_value(p, s, c, d)
    ) * (_g(P["pe"], (s, c, d)))
    # lambdamg is a unit shifter in ALTERTAX (1 in every cell, never shocked)
    pwmg = sum(
        P.amgm.get((m, s, c, d), 0.0)
        * _g(P["ptmg"], (m,))
        / (_g(P["lambdamg"], (m, s, c, d)) + 1e-12)
        for m in p.sets.marg
    )
    pmcif = pefob + pwmg * P.tmarg.get((s, c, d), 0.0)
    pm = (
        (1.0 + _g(P["imptx"], (s, c, d)) + mac.mtax_value(p, d, c, s))
        * pmcif
        / mac.chipm_value(p, s, c, d)
    )
    return {"pefob": pefob, "pmcif": pmcif, "pm": pm}


_TRADE_FIXED: dict[int, tuple[tuple, dict[str, np.ndarray]]] = {}


def _trade_fixed(P: _Period, C: list[str], R: list[str], M: list[str]) -> dict:
    """The level-free factors of :func:`_trade_prices_cell` over the (s, c, d) grid
    (``amgm`` over (m, s, c, d)), through the same macros cell by cell. Built once per
    params (and grid) and shared: as ``mp_margin_params``, it assumes the params are
    not edited after load; dropped when they are collected."""
    from equilibria.blocks.gtap import _ifsub_macros as mac

    p = P.p
    key = (tuple(C), tuple(R), tuple(M))
    hit = _TRADE_FIXED.get(id(p))
    if hit is not None and hit[0] == key:
        return hit[1]
    shape = (len(R), len(C), len(R))
    fob, tmarg, mtax, chipm = (np.empty(shape) for _ in range(4))
    amgm = np.empty((len(M), *shape))
    for i, s in enumerate(R):
        for j, c in enumerate(C):
            for k, d in enumerate(R):
                fob[i, j, k] = (
                    1.0 + mac._safe(p, "taxes.rtxs", (s, c, d)) + mac.etax_value(p, s, c, d)
                )
                tmarg[i, j, k] = P.tmarg.get((s, c, d), 0.0)
                mtax[i, j, k] = mac.mtax_value(p, d, c, s)
                chipm[i, j, k] = mac.chipm_value(p, s, c, d)
                for m_i, m in enumerate(M):
                    amgm[m_i, i, j, k] = P.amgm.get((m, s, c, d), 0.0)
    out = {"fob": fob, "tmarg": tmarg, "mtax": mtax, "chipm": chipm, "amgm": amgm}
    try:
        weakref.finalize(p, _TRADE_FIXED.pop, id(p), None)
    except TypeError:  # not weak-referenceable: no cache
        return out
    _TRADE_FIXED[id(p)] = (key, out)
    return out


def _trade_arrays(P: _Period, els: dict) -> dict[str, np.ndarray]:
    """:func:`_trade_prices_cell` over the whole (s, c, d) grid, in its operation
    order (bit for bit): ``{"pefob", "pmcif", "pm"}``, built once per period."""
    hit = P.dense.get(("_trade",))
    if hit is not None:
        return hit
    C, R, M = els["COMM"], els["REG"], list(P.p.sets.marg)
    f = _trade_fixed(P, C, R, M)
    pefob = f["fob"] * _dense(P, "pe", [R, C, R])
    ptmg = _dense(P, "ptmg", [M])
    lam = _dense(P, "lambdamg", [M, R, C, R])
    pwmg: Any = 0  # as ``sum``: an int start
    for m_i in range(len(M)):
        pwmg = pwmg + f["amgm"][m_i] * ptmg[m_i] / (lam[m_i] + 1e-12)
    pmcif = pefob + pwmg * f["tmarg"]
    imptx = _dense(P, "imptx", [R, C, R])
    pm = (1.0 + imptx + f["mtax"]) * pmcif / f["chipm"]
    hit = P.dense[("_trade",)] = {"pefob": pefob, "pmcif": pmcif, "pm": pm}
    return hit


def _trade_array(pvar: str) -> Callable[[_Period, dict], np.ndarray]:
    """:func:`_trade` over the whole (COMM, REG, REG) grid."""

    def f(P: _Period, els: dict) -> np.ndarray:
        C, R = els["COMM"], els["REG"]
        price = _trade_arrays(P, els)[pvar]
        val = price * _dense(P, "xw", [R, C, R]) / IN_SCALE
        return np.ascontiguousarray(val.transpose(1, 0, 2))

    return f


def _trade(pvar: str) -> Callable[[_Period, str, str, str], float]:
    def f(P: _Period, c: str, s: str, d: str) -> float:
        return _trade_prices(P, s, c, d)[pvar] * _g(P["xw"], (s, c, d)) / IN_SCALE

    return f


def _vxsb(P: _Period, c: str, s: str, d: str) -> float:
    return _g(P["pd"], (s, c)) * _g(P["xw"], (s, c, d)) / IN_SCALE


def _save(P: _Period, r: str) -> float:
    return _g(P["rsav"], (r,)) / IN_SCALE


def _vkb(P: _Period, r: str) -> float:
    return _g(P["pi"], (r,)) * _g(P["kstock"], (r,)) / IN_SCALE


def _vdep(P: _Period, r: str) -> float:
    bm = P.p.benchmark
    rate = bm.vdep[r] / bm.vkb[r]  # exogenous depreciation rate
    return rate * _vkb(P, r)


def _vst(P: _Period, m: str, r: str) -> float:
    return _g(P["pd"], (r, m)) * _g(P["xda"], (r, m, "tmg")) / IN_SCALE


def _vtwr(P: _Period, m: str, c: str, s: str, d: str) -> float:
    """``ptmg[m] * amgm[m,s,c,d] * tmarg[s,c,d] * xw[s,c,d]``. The report vars ``xmgm`` and
    ``xwmg`` are not recomputed at the shock period under ifSUB (stale: equal at check and
    shock while ``xw`` moves), so the margin quantity is rebuilt from the margin share
    ``amgm``, the margin rate ``tmarg`` (``xwmg = tmarg * xw``) and ``xw``."""
    amgm = P.amgm.get((m, s, c, d), 0.0)
    tmarg = P.tmarg.get((s, c, d), 0.0)
    lam = _g(P["lambdamg"], (m, s, c, d)) + 1e-12  # as ``_xmgm`` of ifsub_arrays
    return _g(P["ptmg"], (m,)) * amgm * tmarg * _g(P["xw"], (s, c, d)) / lam / IN_SCALE


def _dense_table(table: dict, axes: list[list[str]]) -> np.ndarray:
    """``table.get(cell, 0.0)`` over the grid ``axes`` (one label list per key element);
    a cell with a label outside ``axes`` is not on the grid."""
    pos = [{e: i for i, e in enumerate(a)} for a in axes]
    out = np.zeros([len(a) for a in axes], dtype=np.float64)
    n = len(axes)
    for cell, x in table.items():
        if len(cell) != n:
            continue
        try:
            ix = tuple(p[e] for p, e in zip(pos, cell, strict=True))
        except KeyError:
            continue
        out[ix] = x
    return out


def _dense(P: _Period, name: str, axes: list[list[str]]) -> np.ndarray:
    """``_g(P[name], cell)`` over the grid ``axes``, built once per period."""
    key = (name, tuple(tuple(a) for a in axes))
    hit = P.dense.get(key)
    if hit is None:
        hit = P.dense[key] = _dense_table(P[name], axes)
    return hit


def _firm_array(kind: str) -> Callable[[_Period, dict], np.ndarray]:
    """:func:`_firm` over the whole (COMM, ACTS, REG) grid, same operation order."""

    def f(P: _Period, els: dict) -> np.ndarray:
        C, A, R = els["COMM"], els["ACTS"], els["REG"]
        dom = kind[1] == "D"
        q = _dense(P, "xda" if dom else "xma", [R, C, A])
        price = _dense(P, "pd" if dom else "pmt", [R, C])[:, :, None]
        if kind[3] == "P":
            price = price * (1.0 + _dense(P, "dintx" if dom else "mintx", [R, C, A]))
        xs = np.array([[P.xscale.get((r, a), 1.0) for a in A] for r in R])
        unscale = (IN_SCALE * xs)[:, None, :]
        return np.ascontiguousarray((price * q / unscale).transpose(1, 2, 0))

    return f


def _make_array(basic: bool) -> Callable[[_Period, dict], np.ndarray]:
    """:func:`_make` over the whole (COMM, ACTS, REG) grid, same operation order."""

    def f(P: _Period, els: dict) -> np.ndarray:
        C, A, R = els["COMM"], els["ACTS"], els["REG"]
        price = _dense(P, "p_rai", [R, A, C])
        if basic:
            price = price * (1.0 + _dense(P, "prdtx_rai", [R, A, C]))
        val = price * _dense(P, "x", [R, A, C]) / IN_SCALE
        return np.ascontiguousarray(val.transpose(2, 1, 0))

    return f


def _vxsb_array(P: _Period, els: dict) -> np.ndarray:
    """:func:`_vxsb` over the whole (COMM, REG, REG) grid."""
    C, R = els["COMM"], els["REG"]
    val = _dense(P, "pd", [R, C])[:, :, None] * _dense(P, "xw", [R, C, R]) / IN_SCALE
    return np.ascontiguousarray(val.transpose(1, 0, 2))


def _vtwr_array(P: _Period, els: dict) -> np.ndarray:
    """:func:`_vtwr` over the whole (MARG, COMM, REG, REG) grid, same operation order."""
    M, C, R = els["MARG"], els["COMM"], els["REG"]
    ptmg = _dense(P, "ptmg", [M])[:, None, None, None]
    amgm = _dense_table(P.amgm, [M, R, C, R])
    tmarg = _dense_table(P.tmarg, [R, C, R])[None]
    xw = _dense(P, "xw", [R, C, R])[None]
    lam = _dense(P, "lambdamg", [M, R, C, R]) + 1e-12
    val = ptmg * amgm * tmarg * xw / lam / IN_SCALE
    return np.ascontiguousarray(val.transpose(0, 2, 1, 3))


# Headers built as whole arrays (bit for bit the cell functions of _VALUE)
_ARRAY: dict[str, Callable[[_Period, dict], np.ndarray]] = {
    "VDFB": _firm_array("VDFB"),
    "VMFB": _firm_array("VMFB"),
    "VDFP": _firm_array("VDFP"),
    "VMFP": _firm_array("VMFP"),
    "MAKS": _make_array(False),
    "MAKB": _make_array(True),
    "VXSB": _vxsb_array,
    "VFOB": _trade_array("pefob"),
    "VCIF": _trade_array("pmcif"),
    "VMSB": _trade_array("pm"),
    "VTWR": _vtwr_array,
}


_VALUE: dict[str, Callable[..., float]] = {
    "VDFB": _firm("VDFB"),
    "VMFB": _firm("VMFB"),
    "VDFP": _firm("VDFP"),
    "VMFP": _firm("VMFP"),
    "EVOS": _factor("pfy"),
    "EVFB": _factor("pf"),
    "EVFP": _factor("pfa"),
    "MAKS": _make(False),
    "MAKB": _make(True),
    "SAVE": _save,
    "VXSB": _vxsb,
    "VFOB": _trade("pefob"),
    "VCIF": _trade("pmcif"),
    "VMSB": _trade("pm"),
    "VST": _vst,
    "VTWR": _vtwr,
    "VKB": _vkb,
    "VDEP": _vdep,
}
for _ag, _l in (("gov", "G"), ("hhd", "P"), ("inv", "I")):
    for _k in ("VD{}B", "VM{}B", "VD{}P", "VM{}P"):
        _VALUE[_k.format(_l)] = _final(_ag, _k.format(_l))


def _elements(run: AltertaxRun) -> dict[str, list[str]]:
    s = run.params.sets
    return {
        "COMM": list(s.i),
        "ACTS": list(s.a),
        "REG": list(s.r),
        "ENDW": list(s.f),
        "MARG": list(s.marg),
    }


def headers_at(run: AltertaxRun, period: str) -> dict[str, HeaderArray]:
    """The value headers of ``run`` at ``period`` (``shock``: the updated data;
    ``check``: reproduces the input database). Built once per run and period (every
    cell is a Python call); each caller gets its own copy of the arrays."""
    cache = getattr(run, "_headers", None)
    built = None if cache is None else cache.get(period)
    if built is None:
        built = _build_headers(run, period)
        if cache is not None:
            cache[period] = built
    return {n: dataclasses.replace(h, array=h.array.copy()) for n, h in built.items()}


def _build_headers(run: AltertaxRun, period: str) -> dict[str, HeaderArray]:
    P = _Period(run, period)
    els = _elements(run)
    out: dict[str, HeaderArray] = {}
    for name, coeff, long_name, sets in _META:
        if name not in _VALUE:
            continue
        lists = [els[x] for x in sets]
        if name in _ARRAY:
            arr = _ARRAY[name](P, els)
        else:
            arr = np.zeros([len(x) for x in lists], dtype=np.float64)
            fn = _VALUE[name]
            for ix in np.ndindex(arr.shape):
                arr[ix] = fn(P, *(lists[d][i] for d, i in enumerate(ix)))
        out[name] = HeaderArray(
            name=name,
            coeff_name=coeff,
            long_name=long_name,
            array=arr,
            set_names=list(sets),
            set_elements=[list(x) for x in lists],
        )
    return out


def rebase(shock: np.ndarray, check: np.ndarray, stored: np.ndarray) -> np.ndarray:
    """GEMPACK updates the stored base value by the percent change, so the updated value
    is ``stored * shock / check`` (``check``: our rebuild of the base, which differs from
    ``stored`` by the input data's rounding residual). Cells with a zero ``check`` or a
    zero ``stored`` keep our ``shock`` value."""
    ok = (check != 0.0) & (stored != 0.0)
    ratio = np.divide(shock, check, out=np.ones_like(shock), where=ok)
    return np.where(ok, stored * ratio, shock)


def _rebased(
    run: AltertaxRun, shock: dict[str, HeaderArray], base: dict[str, HeaderArray]
) -> dict[str, HeaderArray]:
    """``shock`` value headers re-based on the stored ``base`` headers (see :func:`rebase`)."""
    check = headers_at(run, "check")
    out: dict[str, HeaderArray] = {}
    for name, ha in shock.items():
        array = rebase(
            np.asarray(ha.array, dtype=np.float64),
            np.asarray(check[name].array, dtype=np.float64),
            np.asarray(base[name].array, dtype=np.float64),
        )
        out[name] = HeaderArray(
            name=ha.name,
            coeff_name=ha.coeff_name,
            long_name=ha.long_name,
            array=array,
            set_names=list(ha.set_names),
            set_elements=[list(x) for x in ha.set_elements],
        )
    return out


def _copied(basedata: Path) -> dict[str, HeaderArray]:
    """``DVER`` (``gtapv7.tab`` 163: ``VERNUM = 0``, copied to the update file) and
    ``DPSM`` (1479: ``DPARSUM = dpsum``, the sum of the distribution parameters; ``dpsum``
    moves only with the shifters ``dppriv``/``dpgov``/``dpsave``, exogenous at zero in
    ALTERTAX and absent from equilibria) are read from the input database."""
    har = read_har(str(basedata))
    meta = {m[0]: m for m in _META}
    out: dict[str, HeaderArray] = {}
    for name in ("DVER", "DPSM"):
        src = har[name]
        _, coeff, long_name, sets = meta[name]
        out[name] = HeaderArray(
            name=name,
            coeff_name=coeff,
            long_name=long_name,
            array=np.asarray(src.array),
            set_names=list(src.set_names),
            set_elements=[list(x) for x in src.set_elements],
        )
    return out


def _history() -> HeaderArray:
    lines = [
        "This updated data file was created",
        "by equilibria (family route, no GEMPACK) after ALTERTAX:",
        "each value is price * quantity of the shock-period solution,",
        "with the model's quantity scaling (inScale, xscale) undone.",
        "Source run: equilibria.family.altertax.run.solve_altertax",
        "Update rules: gtapv7.tab Update statements",
        "(see equilibria.family.altertax.updated).",
    ]
    return HeaderArray(
        name="XXHS",
        coeff_name=_XXHS[1],
        long_name=_XXHS[2],
        array=np.array(lines, dtype=object),
        set_names=[],
        set_elements=[],
    )


def post_database(
    run: AltertaxRun, basedata: Path | None = None
) -> dict[str, HeaderArray]:
    """The value headers of the updated database (shock period re-based on the stored
    ``basedata``, see :func:`rebase`): what GEMPACK's ``Update`` statements write, and
    what its post-simulation checks (``*Q`` of the summary) are evaluated on."""
    if basedata is None:
        basedata = run.dataset_dir / "basedata.har"
    return _rebased(run, headers_at(run, "shock"), read_har(str(basedata)))


def updated_headers(
    run: AltertaxRun, basedata: Path | None = None
) -> dict[str, HeaderArray]:
    """The 33 headers of the updated database at the shock period, in
    :data:`UPDATED_HEADERS` order. ``basedata`` (default: ``run.dataset_dir / "basedata.har"``)
    supplies ``DVER`` and ``DPSM`` and the stored base the value headers are re-based on."""
    if basedata is None:
        basedata = run.dataset_dir / "basedata.har"
    values = post_database(run, basedata)
    have = {**values, **_copied(Path(basedata)), "XXHS": _history()}
    return {name: have[name] for name in UPDATED_HEADERS}


def write_updated_har(
    run: AltertaxRun, path: Path, basedata: Path | None = None
) -> None:
    """Write the updated database of ``run`` to ``path`` (GEMPACK ``.upd`` shape)."""
    write_har(path, updated_headers(run, basedata))
