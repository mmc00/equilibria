"""What the multiperiod driver does around a solve, on arrays (no Pyomo model).

``_solve_multiperiod_inner`` leaves each period in a state the next ``prepare`` reads.
After the Newton solve of a period (or, for the base, in place of it) it runs, in order:

* the writeback of the solution into the squared columns (``path_capi``: ``set_value``
  of every free variable of the NLP);
* for the shock under ifSUB, ``_recompute_ifsub_report_vars``: the report variables the
  closure fixed are rewritten from the solved values;
* ``_solve_welfare_tail``: each muted welfare row (inactive, its variable fixed) is
  solved for its variable, with ``pyomo.util.calc_var_value.calculate_variable_from_
  constraint`` ported here (``calculate_variable``);
* for the shock with ``code == 1``, ``check_endowment_shock_entered``;
* ``freeze_period``: every variable of the period fixed at its level;
* the ``pabs`` pins (``pabs[r, base] = 1``; ``pabs[r, check]`` at its level), which only
  act if ``freeze_period`` left a cell free.

The functions modify ``mp`` in place: callers pass the copy ``prep_square.square``
returns, which carries the closure fixings, deactivated rows and 1e-3 floors that Pyomo
leaves on its model. Rows are evaluated from the single-period recording
(``FamilyBackend.records``): the same expressions the Pyomo rows have, over the levels
of ``mp``. Evaluation follows Pyomo's: plain Python arithmetic over the nodes (so a
negative base to a fractional power is complex, as there), ``value(..., exception=False)``
suppressing only ``ValueError`` and ``TypeError``.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable
from typing import Any

import numpy as np

from equilibria.family.mp_arrays import MPArrays, Squared
from equilibria.family.prep_recipe import _table
from equilibria.family.recorder import Node

_log = logging.getLogger(__name__)

# u depends on ug/us (and the live uh), so it goes last (``_WELFARE_SOLVE_ORDER``).
WELFARE_SOLVE_ORDER = (
    ("eq_ug", "ug"),
    ("eq_us", "us"),
    ("eq_ev", "ev"),
    ("eq_cv", "cv"),
    ("eq_u", "u"),
)
_INVALID = (complex, type(None))  # Pyomo's ``_invalid_types``


class IterationLimitError(RuntimeError):
    """Pyomo's ``IterationLimitError``: a plain ``RuntimeError`` subclass, so the Newton
    failure that is not this (a vanishing derivative) is not caught by the welfare tail."""


# ── Writeback ───────────────────────────────────────────────────────────────


def _split(name: str) -> tuple[str, tuple]:
    comp, _, rest = name.partition("[")
    return comp, tuple(rest[:-1].split(",")) if rest else ()


def writeback(mp: MPArrays, sq: Squared, x: np.ndarray) -> int:
    """``set_value`` of the solution into the squared columns (all of them free).
    Returns the count written. A column ``mp`` lacks raises ``KeyError``."""
    if len(x) != len(sq.col_names):
        raise ValueError(f"{len(x)} values for {len(sq.col_names)} columns")
    by_var: dict[str, tuple[list[int], list[float]]] = {}
    for name, val in zip(sq.col_names, x, strict=True):
        comp, key = _split(name)
        v = mp.var(comp)
        pos, vals = by_var.setdefault(comp, ([], []))
        pos.append(v.pos(key))
        vals.append(float(val))
    for comp, (pos, vals) in by_var.items():
        mp.vars[comp].value[np.array(pos, dtype=np.int64)] = vals
    return len(x)


# ── freeze_period and the pabs pins ─────────────────────────────────────────


def freeze_period(mp: MPArrays, period: str) -> int:
    """Fix every Var cell of ``period`` at its level (``iterloop.gms`` holdfixed).

    Returns the number of cells fixed. Pyomo fixes a cell with no value at 1.0; a cell
    of ``mp`` always has one (NaN is a level, kept)."""
    n = 0
    for v in mp.vars.values():
        sel = v.period == period
        v.fixed[sel] = True
        n += int(sel.sum())
    return n


def pin_pabs(mp: MPArrays, period: str, regions: list[str]) -> None:
    """The driver's pins after ``freeze_period``: ``pabs[r, base]`` fixed at 1, and
    ``pabs[r, check]`` at its level (1.0 if that is falsy: 0 or no value). A cell
    ``freeze_period`` already fixed is left alone, so this is a no-op after it."""
    pabs = mp.vars.get("pabs")
    if pabs is None:
        return
    tab = _table(pabs, period)
    for r in regions:
        i = tab.get((r,))
        if i is None or pabs.fixed[i]:
            continue
        if period == "base":
            pabs.value[i] = 1.0
        else:
            val = pabs.value[i]
            pabs.value[i] = float(val or 1.0)
        pabs.fixed[i] = True


# ── Row evaluation over the levels of mp ────────────────────────────────────


class Levels:
    """Level of a recording leaf ``(var, flat)`` at ``period``: a Var cell of ``mp`` or a
    mutable Parameter of ``state``. ``override`` replaces one leaf (the unknown)."""

    def __init__(self, mp: MPArrays, backend: Any, state: Any, period: str):
        self.mp, self.backend, self.state, self.period = mp, backend, state, period
        self._cells: dict[tuple[str, int], tuple[np.ndarray, int]] = {}
        self.override: dict[tuple[str, int], Any] = {}

    def key(self, name: str, flat: int) -> tuple:
        doms = self.backend.var_domains[name]
        shape = tuple(len(self.backend.sets[d]) for d in doms)
        idx = np.unravel_index(flat, shape) if doms else ()
        return tuple(
            self.backend.sets[d][int(i)] for d, i in zip(doms, idx, strict=True)
        )

    def flat(self, name: str, key: tuple) -> int:
        doms = self.backend.var_domains[name]
        pos = [self.backend.sets[d].index(e) for d, e in zip(doms, key, strict=True)]
        shape = tuple(len(self.backend.sets[d]) for d in doms)
        return int(np.ravel_multi_index(pos, shape)) if doms else 0

    def __call__(self, name: str, flat: int) -> Any:
        k = (name, flat)
        if k in self.override:
            return self.override[k]
        hit = self._cells.get(k)
        if hit is None:
            if name in self.backend.param_domains:
                data = self.state.find_component(name).data
                hit = (np.asarray(data, dtype=float).ravel(), flat)
            else:
                v = self.mp.var(name)
                hit = (v.value, v.pos((*self.key(name, flat), self.period)))
            self._cells[k] = hit
        arr, i = hit
        return float(arr[i])


def _sum(items: list) -> Any:
    """Pyomo's ``SumExpression._apply_operation``."""
    it = iter(items)
    return sum(it, start=next(it))


def evaluate(x: Any, lv: Callable[[str, int], Any]) -> Any:
    """``Node.evaluate`` as Pyomo's visitor does it, leaves from ``lv``."""
    if not isinstance(x, Node):
        return float(x)
    op, a = x.op, x.args
    if op == "v":
        return lv(a[0], a[1])
    if op == "add":
        return _sum([evaluate(c, lv) for c in a[: x._n]])
    if op == "mul":
        return evaluate(a[0], lv) * evaluate(a[1], lv)
    if op == "div":
        return evaluate(a[0], lv) / evaluate(a[1], lv)
    if op == "pow":
        return evaluate(a[0], lv) ** evaluate(a[1], lv)
    if op == "neg":
        return -evaluate(a[0], lv)
    if op == "sqrt":
        return math.sqrt(evaluate(a[0], lv))
    if op == "log":
        return math.log(evaluate(a[0], lv))
    if op == "exp":
        return math.exp(evaluate(a[0], lv))
    raise NotImplementedError(f"post-solve row evaluation: {op!r}")


def _dual(x: Any, lv: Callable[[str, int], Any], wrt: tuple[str, int]) -> tuple:
    """(value, d value / d leaf ``wrt``) by forward mode: the exact derivative Pyomo's
    symbolic differentiation gives."""
    if not isinstance(x, Node):
        return float(x), 0.0
    op, a = x.op, x.args
    if op == "v":
        return lv(a[0], a[1]), (1.0 if (a[0], a[1]) == wrt else 0.0)
    if op == "add":
        parts = [_dual(c, lv, wrt) for c in a[: x._n]]
        return _sum([p[0] for p in parts]), _sum([p[1] for p in parts])
    if op == "mul":
        (u, du), (w, dw) = _dual(a[0], lv, wrt), _dual(a[1], lv, wrt)
        return u * w, du * w + u * dw
    if op == "div":
        (u, du), (w, dw) = _dual(a[0], lv, wrt), _dual(a[1], lv, wrt)
        return u / w, (du * w - u * dw) / w**2
    if op == "pow":
        (u, du), (w, dw) = _dual(a[0], lv, wrt), _dual(a[1], lv, wrt)
        val = u**w
        d = w * u ** (w - 1) * du if dw == 0 else val * (dw * math.log(u) + w * du / u)
        return val, d
    if op == "neg":
        u, du = _dual(a[0], lv, wrt)
        return -u, -du
    u, du = _dual(a[0], lv, wrt)
    if op == "sqrt":
        r = math.sqrt(u)
        return r, du / (2.0 * r)
    if op == "log":
        return math.log(u), du / u
    if op == "exp":
        r = math.exp(u)
        return r, r * du
    raise NotImplementedError(f"post-solve row derivative: {op!r}")


def _body_upper(res: Any) -> tuple[Callable[[Callable], Any], float]:
    """Pyomo's (body, upper) of ``lhs == rhs``: a constant right-hand side is the bound
    (``sum(terms) == 1.0``), otherwise the body is ``lhs - rhs`` and the bound 0."""
    lhs, rhs = res.lhs, res.rhs
    if not isinstance(rhs, Node):
        return (lambda lv: evaluate(lhs, lv)), float(rhs)
    return (lambda lv: evaluate(lhs, lv) - evaluate(rhs, lv)), 0.0


def row_residual(res: Any, lv: Callable[[str, int], Any]) -> Any:
    """``value(row.body) - value(row.upper)``."""
    body, upper = _body_upper(res)
    return body(lv) - upper


def _no_exception(fn: Callable[[], Any]) -> Any:
    """``value(expr, exception=False)``: ``ValueError`` and ``TypeError`` give ``None``."""
    try:
        return fn()
    except (ValueError, TypeError):
        return None


# ── calculate_variable_from_constraint ──────────────────────────────────────


def calculate_variable(
    x0: float,
    body: Callable[[Any], Any],
    deriv: Callable[[Any], Any],
    upper: float,
    *,
    appears: bool = True,
    eps: float = 1e-8,
    iterlim: int = 1000,
    linesearch: bool = True,
    alpha_min: float = 1e-8,
) -> Any:
    """``pyomo.util.calc_var_value.calculate_variable_from_constraint`` for one variable.

    ``body(x)`` is the constraint body with the variable at ``x`` and ``deriv(x)`` its
    derivative there; the constraint is ``body == upper``. Same steps: the guess that
    assumes a unit coefficient, the secant (linear) solution, then Newton with the
    backtracking line search. Returns the variable's final value. ``ValueError`` (the
    variable does not appear; a complex residual; a derivative ~0 at the start),
    ``ArithmeticError`` (a division by zero or overflow while evaluating) and
    ``IterationLimitError`` are the failures the caller treats as "no solution";
    ``RuntimeError`` (a vanishing derivative inside the loop) is not one of them."""
    orig = x0
    x1 = x0
    residual_1 = body(x1)
    residual_2: Any
    body_val: Any
    try:
        v = x1 - (residual_1 - upper)
        residual_2 = _no_exception(lambda: body(v))
    except OverflowError:
        residual_2 = None

    if residual_2.__class__ not in _INVALID:
        if abs(residual_2 - upper) < eps:  # ty: ignore[unsupported-operator]
            return v
        x2 = v
        slope = float(residual_1 - residual_2) / (x1 - x2)
        intercept = (residual_1 - upper) - slope * x1
        if slope:
            v = -intercept / slope
            try:
                body_val = _no_exception(lambda: body(v))
            except OverflowError:
                body_val = None
            if (
                body_val.__class__ not in _INVALID and abs(body_val - upper) < eps  # ty: ignore[unsupported-operator]
            ):
                return v

    # The variable appears nonlinearly: Newton from the initial value.
    v = orig
    if not appears:
        raise ValueError("variable derivative == 0 in the constraint, cannot solve")
    fp0 = deriv(v)
    if abs(fp0) < 1e-12:
        raise ValueError("derivative of the constraint is ~0 at the initial value")

    iter_left = iterlim
    fk = residual_1 - upper
    while abs(fk) > eps and iter_left:
        iter_left -= 1
        if not iter_left:
            raise IterationLimitError("iteration limit reached")
        xk = v
        fk = body(xk) - upper
        if fk.__class__ in _INVALID and fk is not None:
            raise ValueError("Complex numbers are not allowed in Newton's method.")
        fpk = deriv(xk)
        if abs(fpk) < 1e-12:
            raise RuntimeError("derivative of the constraint ~0 in Newton's method")
        pk = -fk / fpk
        alpha = 1.0
        v = xk + alpha * pk
        if linesearch:
            c1 = 0.999  # ensure sufficient progress
            while alpha > alpha_min:
                try:
                    fkp1 = _no_exception(lambda: body(v) - upper)
                    if fkp1.__class__ in _INVALID:
                        fkp1 = None
                    if fkp1 is not None and fkp1**2 < c1 * fk**2:
                        fk = fkp1
                        break
                except OverflowError:
                    pass
                alpha /= 2.0
                v = xk + alpha * pk
            if alpha <= alpha_min:
                raise IterationLimitError("line search iteration limit reached")
    return v


# ── The welfare tail ────────────────────────────────────────────────────────


def _records(backend: Any, family: str) -> dict[tuple, Any]:
    cache = backend.__dict__.setdefault("_post_solve_records", {})
    if family not in cache:
        cache[family] = {tuple(idx): res for idx, res in backend.records[family]}
    return cache[family]


def solve_welfare_tail(
    mp: MPArrays, backend: Any, state: Any, period: str
) -> tuple[int, list[str]]:
    """Solve each row muted by the recipe for its own variable (``_solve_welfare_tail``).

    Applies to inactive rows of ``period`` whose variable is fixed; the row stays off and
    the variable fixed. A row with no solution (``ValueError``, ``ArithmeticError``,
    ``IterationLimitError``) leaves its variable NaN. Returns the number solved and the
    names left undefined."""
    n = 0
    undefined: list[str] = []
    lv = Levels(mp, backend, state, period)
    for eqn, vn in WELFARE_SOLVE_ORDER:
        row, var = mp.rows.get(eqn), mp.vars.get(vn)
        if row is None or var is None:
            continue
        recs = _records(backend, eqn)
        vtab = _table(var, period)
        for key, ir in _table(row, period).items():
            j = vtab.get(key)
            if row.active[ir] or j is None or not var.fixed[j]:
                continue
            res = recs[key]
            unknown = (vn, lv.flat(vn, key))
            body, upper = _body_upper(res)

            def f(x, body=body, unknown=unknown):
                lv.override[unknown] = x
                return body(lv)

            def df(x, res=res, unknown=unknown):
                lv.override[unknown] = x
                d = _dual(res.lhs, lv, unknown)[1]
                if isinstance(res.rhs, Node):
                    d = d - _dual(res.rhs, lv, unknown)[1]
                return d

            leaves = set(res.lhs.var_leaves() if isinstance(res.lhs, Node) else [])
            if isinstance(res.rhs, Node):
                leaves |= set(res.rhs.var_leaves())
            try:
                var.value[j] = calculate_variable(
                    float(var.value[j]),
                    f,
                    df,
                    upper,
                    appears=unknown in leaves,
                )
                n += 1
            except (ValueError, ArithmeticError, IterationLimitError):
                var.value[j] = float("nan")
                undefined.append(mp.name(vn, j))
            finally:
                lv.override.pop(unknown, None)
    if undefined:
        _log.warning(
            "%s period: welfare undefined (no real solution), set to NaN: %s",
            period,
            ", ".join(undefined),
        )
    return n, undefined


# ── ifSUB report variables ──────────────────────────────────────────────────


def mp_margin_params(params: Any, sets: Any) -> dict[str, tuple[int, dict]]:
    """The ``tmarg`` and ``amgm`` Params of the multiperiod Pyomo model
    (``GTAPMultiPeriodModel.build_sets``): ``{name: (number of indices, {key: value})}``,
    0.0 where absent (the Param default). Same formulas as there, which fall back to
    ``vxmd`` for the trade base where ``vxsb`` is 0 (``tmarg_data`` does not)."""
    bm = params.benchmark
    tmarg: dict = {}
    amgm: dict = {}
    for r in sets.r:
        for i in sets.i:
            for rp in sets.rp if hasattr(sets, "rp") else sets.r:
                vtwr_total = (
                    sum(float(bm.vtwr.get((r, i, rp, mg), 0.0) or 0.0) for mg in sets.m)
                    if hasattr(bm, "vtwr")
                    else 0.0
                )
                vcif = (
                    float(bm.vcif.get((r, i, rp), 0.0) or 0.0)
                    if hasattr(bm, "vcif")
                    else 0.0
                )
                vfob = (
                    float(bm.vfob.get((r, i, rp), 0.0) or 0.0)
                    if hasattr(bm, "vfob")
                    else 0.0
                )
                xw_bench = (
                    float(bm.vxsb.get((r, i, rp), 0.0) or 0.0)
                    if hasattr(bm, "vxsb")
                    else 0.0
                )
                if xw_bench <= 0.0 and hasattr(bm, "vxmd"):
                    xw_bench = float(bm.vxmd.get((r, i, rp), 0.0) or 0.0)
                tmarg[(r, i, rp)] = (
                    max(vcif - vfob, 0.0) / max(xw_bench, 1e-12)
                    if xw_bench > 0.0
                    else 0.0
                )
                if vtwr_total > 0.0:
                    for mg in sets.m:
                        flow = (
                            float(bm.vtwr.get((r, i, rp, mg), 0.0) or 0.0)
                            if hasattr(bm, "vtwr")
                            else 0.0
                        )
                        amgm[(mg, r, i, rp)] = flow / vtwr_total
    return {"tmarg": (3, tmarg), "amgm": (4, amgm)}


def recompute_ifsub_report_vars(
    mp: MPArrays, params: Any, state: Any, period: str
) -> int:
    """``_recompute_ifsub_report_vars``: pfa / pfy / pp_rai / pwmg / pefob / pmcif / pm
    of ``period`` rewritten from the solved levels, as GAMS ``postsim`` recomputes them.

    Reads and writes as Pyomo does, key by key: a Var key carries the period, a cell or
    component the model lacks gives the default (reads) or is skipped (writes), and the
    Params ``tmarg`` / ``amgm`` are the multiperiod model's own (``mp_margin_params``).
    The formulas are the driver's (``pp_rai = p_rai``; the factor wedge from the
    benchmark), which are not the ``M_*`` macros of the closure. Returns the cells
    written."""
    from equilibria.templates.gtap.gtap_multiperiod_driver import (
        _benchmark_factor_wedge,
    )

    sets = state.sets
    R, A, I, F = (list(sets[k]) for k in ("r", "a", "i", "f"))
    M = list(sets["m"]) if "m" in sets else []
    mp_params = mp_margin_params(params, params.sets)
    tabs: dict[str, dict[tuple, int] | None] = {}

    def tab(comp: str):
        if comp not in tabs:
            v = mp.vars.get(comp)
            tabs[comp] = None if v is None else _table(v, period)
        return tabs[comp]

    def _pv(comp, key, default=0.0):
        if comp in mp_params:
            ndim, data = mp_params[comp]
            return data.get(key, 0.0) if len(key) == ndim else default
        t = tab(comp)
        if t is None or not key or key[-1] != period:
            return default
        i = t.get(key[:-1])
        return default if i is None else float(mp.vars[comp].value[i])

    def _param(d, key, default=0.0):
        v = d.get(key)
        if v is None and len(key) == 3:
            v = d.get((key[2], key[1], key[0]))  # transposed tolerance
        return float(v or default)

    n = 0

    def _force(comp, key, val) -> bool:
        nonlocal n
        t = tab(comp)
        i = None if t is None or key[-1] != period else t.get(key[:-1])
        if i is None:
            return False
        mp.vars[comp].value[i] = val
        n += 1
        return True

    _pfa_wedge = _benchmark_factor_wedge(params)

    # --- factor prices: pfa, pfy ---
    if tab("pf") is not None:
        for r in R:
            for f in F:
                for a in A:
                    pf = _pv("pf", (r, f, a, period), None)
                    if pf is None:
                        continue
                    kappaf = _pv("kappaf", (r, f, a, period))
                    _force("pfa", (r, f, a, period), pf * (1.0 + _pfa_wedge(r, f, a)))
                    _force("pfy", (r, f, a, period), pf * (1.0 - kappaf))

    # --- producer price: pp_rai ---
    if tab("p_rai") is not None and tab("pp_rai") is not None:
        for r in R:
            for a in A:
                for i in I:
                    p_rai = _pv("p_rai", (r, a, i, period), None)
                    if p_rai is None:
                        continue
                    _force("pp_rai", (r, a, i, period), p_rai)

    # --- trade prices: pwmg, pefob, pmcif, pm ---
    if tab("pe") is not None:
        bm, tx = params.benchmark, params.taxes
        for r in R:  # exporter
            for i in I:
                for rp in R:  # importer
                    pe = _pv("pe", (r, i, rp, period), None)
                    if pe is None:
                        continue
                    pwmg_val = 0.0
                    for mm in M:
                        amgm = (
                            _param(bm.amgm, (mm, r, i, rp))
                            if hasattr(bm, "amgm")
                            else _pv("amgm", (mm, r, i, rp))
                        )
                        ptmg = _pv("ptmg", (mm, period))
                        lam = _pv("lambdamg", (mm, r, i, rp, period), 1.0) or 1.0
                        pwmg_val += amgm * ptmg / (lam + 1e-12)
                    rtxs = _param(tx.rtxs, (r, i, rp)) if hasattr(tx, "rtxs") else 0.0
                    etax = _pv("etax", (r, i, period))
                    pefob_val = (1.0 + rtxs + etax) * pe
                    # tmarg is dim-3 (no period axis); mtax/chipm come from params.
                    tmarg = _pv("tmarg", (r, i, rp))
                    if not tmarg:
                        tmarg = _pv("tmarg", (r, i, rp, period))
                    pmcif_val = pefob_val + pwmg_val * tmarg
                    imptx = _pv("imptx", (r, i, rp, period))
                    mtax = _param(tx.mtax, (rp, i)) if hasattr(tx, "mtax") else 0.0
                    chipm = (
                        _param(bm.chipm, (r, i, rp), 1.0)
                        if hasattr(bm, "chipm")
                        else 1.0
                    )
                    chipm = chipm or 1.0
                    pm_val = (1.0 + imptx + mtax) * pmcif_val / (chipm + 1e-12)
                    for vn, val in (
                        ("pwmg", pwmg_val),
                        ("pefob", pefob_val),
                        ("pmcif", pmcif_val),
                        ("pm", pm_val),
                    ):
                        _force(vn, (r, i, rp, period), val)
    return n


# ── The endowment shock check ───────────────────────────────────────────────


def shocked_cells(mp: MPArrays, name: str) -> list[tuple]:
    """Keys (period last) of the ``shock`` cells of instrument ``name`` that differ from
    their ``check`` value (``shocks._shocked_cells``), endogenous ``@overwrite`` cells
    excluded."""
    from equilibria.templates.gtap.instruments import is_free_cell

    if name not in mp.exogenous or name not in mp.vars:
        return []
    v = mp.vars[name]
    libres = mp.endogenous_instrument_cells.get(name)
    chk = _table(v, "check")
    out = []
    for cell, i in _table(v, "shock").items():
        key = (*cell, "shock")
        if libres and is_free_cell(key, libres):
            continue
        j = chk.get(cell)
        if j is not None and float(v.value[i]) != float(v.value[j]):
            out.append(key)
    return out


def check_endowment_shock_entered(mp: MPArrays, backend: Any, state: Any) -> None:
    """After a converged solve, ``RuntimeError`` if an endowment shock (``aft``) did not
    reach the solution. Evaluates the ``xft = aft * (pft / pabs) ** etaf`` row of the
    shock (active or not) at the shocked ``aft``: with ``etaf = 0`` that is exactly "xft
    moved the shock's factor"."""
    if "eq_xfteq" not in mp.rows:
        return
    eq = _table(mp.rows["eq_xfteq"], "shock")
    recs = _records(backend, "eq_xfteq")
    lv = Levels(mp, backend, state, "shock")
    for k in shocked_cells(mp, "aft"):
        cell = k[:-1]
        if cell not in eq:
            continue
        resid = float(row_residual(recs[cell], lv))
        xft = float(mp.var("xft").value[mp.var("xft").pos(k)])
        if abs(resid) > 1e-6 * max(1.0, abs(xft)):
            aft = mp.var("aft")
            raise RuntimeError(
                f"aft{tuple(cell)} = {aft.value[aft.pos(k)]} but eq_xfteq{k} is off "
                f"by {resid:.3e} (xft = {xft}): the shock did not enter the solution"
            )


# ── The driver's sequence ───────────────────────────────────────────────────


def post_solve(
    mp: MPArrays,
    period: str,
    *,
    backend: Any,
    state: Any,
    params: Any,
    regions: list[str],
    if_sub: bool,
    sq: Squared | None = None,
    x: np.ndarray | None = None,
    code: int = 1,
) -> MPArrays:
    """The steps after the solve of ``period`` (``mp`` modified and returned).

    ``base`` is not solved (``skip_base_solve``): pass no ``sq`` / ``x``. Check and
    shock write the solution ``x`` of ``sq`` back first; the shock then recomputes the
    ifSUB report variables (``if_sub``), solves the welfare tail, and, when ``code`` is
    1, checks the endowment shock entered. Every period ends with ``freeze_period`` and
    the ``pabs`` pins."""
    if (sq is None) != (x is None):
        raise ValueError("sq and x go together")
    if period == "base":
        if sq is not None:
            raise ValueError("the base is not solved: no sq / x")
    elif sq is None:
        raise ValueError(f"{period}: the solution (sq, x) is required")
    if sq is not None and x is not None:
        writeback(mp, sq, x)
    if period == "shock" and if_sub:
        recompute_ifsub_report_vars(mp, params, state, "shock")
    solve_welfare_tail(mp, backend, state, period)
    if period == "shock" and code == 1:
        check_endowment_shock_entered(mp, backend, state)
    freeze_period(mp, period)
    if period != "shock":
        pin_pabs(mp, period, regions)
    return mp


__all__ = [
    "WELFARE_SOLVE_ORDER",
    "calculate_variable",
    "check_endowment_shock_entered",
    "freeze_period",
    "pin_pabs",
    "post_solve",
    "recompute_ifsub_report_vars",
    "solve_welfare_tail",
    "writeback",
]
