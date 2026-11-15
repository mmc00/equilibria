"""Closure, squaring and matching of one period on arrays: the Pyomo-free port of the
tail of ``path_capi._run_path_capi_nonlinear_full`` that runs before the solver.

Input: the multiperiod state after the period recipe (``MPArrays``). Output: the
``Squared`` system the solver receives (active rows and free columns in paired order,
and every cell of the active period). In order, as ``path_capi`` does (``gtap`` mode):

1. ``GTAPSolver._initialize_benchmark`` (a missing value becomes 1.0);
2. ``apply_closure`` (numeraire, and the closure flags), then ``apply_conditional_fixing``
   (flag fixing from the benchmark flows; Pyomo runs it twice, the second pass is a
   fixed point of the first and is not repeated);
3. ``apply_aggressive_fixing_for_mcp``: only the gap is counted; a gap > 0 (Pyomo
   would fix variables by matching) raises ``ValueError``;
4. ``apply_squareness_patches`` (``protect_xseq``): ``fix_sluggish_pft``,
   ``deactivate_fixed_quantity_eqs`` and ``deactivate_zero_unique_var_eqs``, which
   raises ``ValueError`` when rows outnumber free variables (Pyomo would drop rows by
   matching). ``deactivate_xfteq_for_fixed_mobile`` and the ``yi`` / ``pft`` blocks that
   follow are no-ops on the multiperiod model (it has no ``xftflag`` Param and ``yi`` /
   ``pft`` are indexed with the period, so their lookups miss);
5. the 1e-3 relative lower bounds (``_relative_floor_vars``, gtap set);
6. rows and free columns sorted by name, the GAMS pairs (``gams_pairs``), and the
   matching: hard pairs locked, soft pairs a starting point, a maximum matching on the
   rest of the incidence, leftover columns to the rows left unmatched.

The SOLVE_NLP deactivate / fix loop that ends the Pyomo code deactivates the active
rows and fixes the free variables that are not in ``constraints`` / ``free_variables``;
those lists are every active row and every free variable, so it does nothing.

Not ported: the investment log loop (log only), the optional env paths
(``EQUILIBRIA_GTAP_WARM_START``, ``..._DIRECT_DEGEN_FIX``, ``..._BTF_PROBE``) and the
altertax closure. A Var or Constraint component a step needs and the model lacks raises
``KeyError``; the GAMS pairs of components the model does not have are skipped, as
Pyomo does (``eq_pfaeq`` and ``eq_pfyeq`` are not in every model).
"""

from __future__ import annotations

import itertools
from typing import Any, Literal, overload

import numpy as np

from equilibria.family.cell_index import CellIndex, SquaredIds
from equilibria.family.mp_arrays import MPArrays, RowArr, Squared, VarArr
from equilibria.solver._matching import match_rows

_PRICE_LB_FACTOR = 1e-3
_GAMS_HARD_PAIR_EQS = ("eq_pfeq", "eq_pfyeq", "eq_pdeq")
_PERIODIC_SHOCK_EQS = ("eq_xweq_shock_ifsub", "eq_pmteq_shock_ifsub")

# (equation, variable) of the GAMS ``model`` statement (path_capi ``_gams_pairs``).
_GAMS_PAIRS = (
    ("eq_pfeq", "pf"),
    ("eq_xfteq", "xft"),
    ("eq_pfaeq", "pfa"),
    ("eq_pfyeq", "pfy"),
    ("eq_xfeq", "xf"),
    ("eq_xpeq", "xp"),
    ("eq_ppeq", "pp"),
    ("eq_pdeq", "pd"),
    ("eq_ndeq", "nd"),
    ("eq_vaeq", "va"),
    ("eq_pxeq", "px"),
    ("eq_pvaeq", "pva"),
    ("eq_pfact", "pfact"),
    ("eq_xapeq", "xaa"),
    ("eq_xeq", "x"),
    ("eq_pseq", "ps"),
    ("eq_pmteq", "pmt"),
    ("eq_pmeq", "pm"),
    ("eq_xweq", "xw"),
    ("eq_xweq_shock_ifsub", "xw"),
    ("eq_pmteq_shock_ifsub", "pmt"),
    ("eq_pwfact", "pwfact"),
)
# The degenerate CET block under omegax=inf: GAMS declares these pairs.
_SUPPLY_CYCLE = (
    ("eq_xs", "ps"),
    ("eq_xds", "xds"),
    ("eq_pdeq", "pd"),
    ("eq_xet", "xet"),
    ("eq_peteq", "pet"),
    ("eq_xseq", "xs"),
)
_FLOOR_PRICES = ("pmt", "pmcif", "pefob", "px", "pd", "pf", "pwmg")
_FLOOR_LEVELS = ("uh", "ug", "us")
_REQUIRED_VARS = (
    ("xw", "pe", "pm", "pmcif", "pefob", "imptx", "exptx", "xe", "xwmg", "pwmg")
    + ("xmgm", "xf", "pf", "pfa", "pfy", "xp", "va", "nd", "x", "p_rai", "pp_rai")
    + ("pp", "pabs", "pft")
    + _FLOOR_PRICES
    + _FLOOR_LEVELS
)
_REQUIRED_ROWS = ("eq_va", "eq_xfteq")


# ── Key lookups (cached on the arrays: keys never change) ───────────────────


def _grid(obj: VarArr | RowArr) -> np.ndarray:
    """The keys as a (cells, dims) string array."""
    c = obj.__dict__.setdefault("_kgrid", [])
    if not c:
        c.append(np.array(obj.keys, dtype=str).reshape(len(obj.keys), -1))
    return c[0]


def _codes(obj: VarArr | RowArr, j: int) -> tuple[np.ndarray, dict[str, int]]:
    """Integer code of each cell's j-th key element, and the element -> code map."""
    cache = obj.__dict__.setdefault("_kcodes", {})
    if j not in cache:
        vocab, inv = np.unique(_grid(obj)[:, j], return_inverse=True)
        cache[j] = (inv.astype(np.int64), {str(s): i for i, s in enumerate(vocab)})
    return cache[j]


def _in_set(obj: VarArr | RowArr, flags: set[tuple], ndim: int) -> np.ndarray:
    """Cells whose first ``ndim`` key elements are in ``flags`` (tuples of that size)."""
    out = np.zeros(len(obj.keys), dtype=bool)
    if not flags or _grid(obj).shape[1] < ndim:
        return out
    code = np.zeros(len(obj.keys), dtype=np.int64)
    maps = []
    stride = 1
    for j in range(ndim):
        inv, vocab = _codes(obj, j)
        code += inv * stride
        maps.append((vocab, stride))
        stride *= max(len(vocab), 1)
    want = []
    for f in flags:
        if all(f[j] in maps[j][0] for j in range(ndim)):
            want.append(sum(maps[j][0][f[j]] * maps[j][1] for j in range(ndim)))
    return np.isin(code, np.array(want, dtype=np.int64)) if want else out


def _pos_map(obj: VarArr | RowArr) -> dict[tuple, int]:
    cache = obj.__dict__.setdefault("_posmap", {})
    if not cache:
        cache.update({k: i for i, k in enumerate(obj.keys)})
    return cache


def _names(comp: str, keys: list[tuple]) -> list[str]:
    return [f"{comp}[{','.join(k)}]" if k else comp for k in keys]


# ── Fixing helpers ──────────────────────────────────────────────────────────


def _fix(v: VarArr, sel: np.ndarray, value: float | None = None) -> None:
    """``fix(value)`` (or ``fix()``, keeping the value, when ``value`` is None)."""
    if value is not None:
        v.value[sel] = value
    v.fixed[sel] = True


def _fix_zero(v: VarArr, sel: np.ndarray) -> None:
    """``_fix_to_zero_with_lb``: a positive lower bound drops to 0, then fix at 0."""
    v.lb[sel & (v.lb > 0.0)] = 0.0
    _fix(v, sel, 0.0)


# ── Parameters the Pyomo steps read from the model ──────────────────────────


def _positive(d: dict) -> set[tuple]:
    return {k for k, x in d.items() if x is not None and float(x) > 0}


def _trade_params(params: Any) -> tuple[set[tuple], set[tuple], set[tuple]]:
    """(xwFlag, tmarg > 0, amgm > 0): the time-invariant Params of
    ``GTAPMultiPeriodModel.build_sets`` and the route mask of ``apply_conditional_fixing``."""
    bm = params.benchmark
    xw_flag = _positive(getattr(bm, "vxsb", {}))
    if not xw_flag and hasattr(params, "shares"):
        xw_flag = _positive(getattr(params.shares, "p_amw", {}))

    sets = params.sets
    rps = sets.rp if hasattr(sets, "rp") else sets.r
    vtwr = getattr(bm, "vtwr", {})
    vcif, vfob = getattr(bm, "vcif", {}), getattr(bm, "vfob", {})
    vxsb, vxmd = getattr(bm, "vxsb", {}), getattr(bm, "vxmd", {})
    tmarg_pos, amgm_pos = set(), set()
    for r in sets.r:
        for i in sets.i:
            for rp in rps:
                total = sum(
                    float(vtwr.get((r, i, rp, mg), 0.0) or 0.0) for mg in sets.m
                )
                xw_bench = float(vxsb.get((r, i, rp), 0.0) or 0.0)
                if xw_bench <= 0.0 and hasattr(bm, "vxmd"):
                    xw_bench = float(vxmd.get((r, i, rp), 0.0) or 0.0)
                if xw_bench > 0.0:
                    gap = float(vcif.get((r, i, rp), 0.0) or 0.0) - float(
                        vfob.get((r, i, rp), 0.0) or 0.0
                    )
                    if max(gap, 0.0) / max(xw_bench, 1e-12) > 0.0:
                        tmarg_pos.add((r, i, rp))
                if total > 0.0:
                    for mg in sets.m:
                        flow = float(vtwr.get((r, i, rp, mg), 0.0) or 0.0)
                        if flow / total > 0.0:
                            amgm_pos.add((mg, r, i, rp))
    return xw_flag, tmarg_pos, amgm_pos


def _output_flags(params: Any) -> tuple[set[tuple], set[tuple]]:
    """(xFlag over (r, a), makb support over (r, a, i)); xFlag is keyed by activity
    (``gtap_parameters.activity_output_flag``)."""
    from equilibria.templates.gtap.gtap_parameters import activity_output_flag

    bm = params.benchmark
    x_rai = {
        k
        for k, val in getattr(bm, "makb", {}).items()
        if val is not None and abs(float(val)) > 1e-12
    }
    return activity_output_flag(bm, getattr(params.sets, "a", [])), x_rai


# ── Steps ───────────────────────────────────────────────────────────────────


def _initialize_benchmark(mp: MPArrays) -> None:
    for v in mp.vars.values():
        v.value[np.isnan(v.value)] = 1.0


def _apply_closure(mp: MPArrays, closure: Any, exogenous: frozenset[str]) -> None:
    """``GTAPSolver.apply_closure`` (``apply_flag_fixing`` is the conditional fixing
    below; ``close_mcp_gap`` is the gap check below)."""
    if closure.numeraire and closure.numeraire_mode == "fixed_benchmark":
        v = mp.var(closure.numeraire)
        keep = v.fixed if closure.numeraire in exogenous else np.zeros_like(v.fixed)
        _fix(v, ~keep, 1.0)
    if closure.fix_technology:
        for name in ("axp", "lambdaio", "lambdaf", "lambdaxm", "lambdam"):
            if name in mp.vars:  # a legacy list: not every model has all of them
                _fix(mp.vars[name], np.ones(len(mp.vars[name].keys), bool), 1.0)
    if closure.fix_taxes:
        for name in ("rto", "rtf", "rtms", "rtxs", "mtax", "etax", "dintx", "mintx"):
            if name in mp.vars:
                v = mp.vars[name]
                _fix(v, ~np.isnan(v.value))
        for name in ("eq_dintxeq", "eq_mintxeq"):
            if name in mp.rows:
                mp.rows[name].active[:] = False
    if getattr(closure, "gams_factor_pairing", False):
        mp.row("eq_pfteq").active[:] = False
    elif closure.fix_endowments:
        v = mp.var("xft")
        level = np.where(np.isnan(v.value), 1.0, v.value)
        v.value[:] = np.maximum(level, v.lb)
        _fix(v, np.ones(len(v.keys), bool))


class ConditionalMasks:
    """The cells ``apply_conditional_fixing`` fixes because they lie outside the data
    flags, per component, over every period of ``mp``.

    The flags depend on the data (``params``) only, so the masks are built once per run
    and reused by every ``square`` call (``mp``'s keys never change). The parts that
    depend on the current values (``imptx`` / ``exptx`` not NaN) stay in the call."""

    OUTSIDE_XW = ("xw", "pe", "pm", "pmcif", "pefob", "imptx", "exptx")
    OUTSIDE_TMARG = ("xwmg", "pwmg")
    OUTSIDE_XF = ("xf", "pf", "pfa", "pfy")
    OUTSIDE_X_FLAG = ("xp", "va", "nd")
    OUTSIDE_X_RAI = ("x", "p_rai", "pp_rai")

    def __init__(self, outside: dict[str, np.ndarray], keys: dict[str, list]):
        self.outside = outside
        self._keys = keys  # per component, the keys the mask was built for
        self._checked: dict[str, list] = {}  # key lists already verified equal

    def __getstate__(self):
        return {"outside": self.outside, "_keys": self._keys, "_checked": {}}

    @classmethod
    def from_params(cls, params: Any, mp: MPArrays) -> ConditionalMasks:
        xw_flag, tmarg_pos, amgm_pos = _trade_params(params)
        xf_flag = _positive(getattr(params.benchmark, "vfm", {}))
        x_flag, x_rai = _output_flags(params)
        regions = {(r,) for (r, _a) in x_flag}

        out: dict[str, np.ndarray] = {}
        keys: dict[str, list] = {}
        for names, flags, ndim in (
            (cls.OUTSIDE_XW, xw_flag, 3),
            (cls.OUTSIDE_TMARG, tmarg_pos, 3),
            (("xmgm",), amgm_pos, 4),
            (cls.OUTSIDE_XF, xf_flag, 3),
            (cls.OUTSIDE_X_FLAG, x_flag, 2),
            (cls.OUTSIDE_X_RAI, x_rai, 3),
            (("pabs",), regions, 1),
        ):
            for name in names:
                v = mp.var(name)
                out[name] = ~_in_set(v, flags, ndim)
                keys[name] = v.keys
        return cls(out, keys)

    def get(self, mp: MPArrays, name: str) -> np.ndarray:
        """The mask of ``name``; raises if ``mp`` has other keys (size or order).

        Identity first (``mp.copy()`` shares the key lists); an equal list that is a
        different object is compared once and remembered."""
        try:
            m = self.outside[name]
        except KeyError:
            raise KeyError(f"no conditional mask for {name!r}") from None
        keys = mp.var(name).keys
        if keys is not self._keys[name] and keys is not self._checked.get(name):
            if len(keys) != len(m) or keys != self._keys[name]:
                raise ValueError(
                    f"conditional mask of {name!r} was built for other keys"
                )
            self._checked[name] = keys
        return m


def _conditional_fixing(mp: MPArrays, masks: ConditionalMasks) -> None:
    """``GTAPSolver.apply_conditional_fixing`` over every cell of every period.

    The ``xft`` / ``pft`` fixing at the end needs the ``xftflag`` Param, which the
    multiperiod model does not have (the Pyomo step is skipped there too)."""

    def outside(name: str) -> np.ndarray:
        return masks.get(mp, name)

    _fix_zero(mp.var("xw"), outside("xw"))
    for name in ("pe", "pm", "pmcif", "pefob"):
        _fix(mp.var(name), outside(name), 1.0)
    for name in ("imptx", "exptx"):
        v = mp.var(name)
        _fix(v, outside(name) & ~np.isnan(v.value))
    xe = mp.var("xe")
    _fix(xe, np.ones(len(xe.keys), bool))
    _fix_zero(mp.var("xwmg"), outside("xwmg"))
    _fix_zero(mp.var("pwmg"), outside("pwmg"))
    _fix_zero(mp.var("xmgm"), outside("xmgm"))
    _fix_zero(mp.var("xf"), outside("xf"))
    for name in ("pf", "pfa", "pfy"):
        _fix(mp.var(name), outside(name), 1.0)
    for name in ("xp", "va", "nd"):
        _fix_zero(mp.var(name), outside(name))
    _fix_zero(mp.var("x"), outside("x"))
    for name in ("p_rai", "pp_rai"):
        _fix(mp.var(name), outside(name))
    pp = mp.var("pp")
    _fix(pp, np.ones(len(pp.keys), bool))
    _fix(mp.var("pabs"), outside("pabs"))


def _n_free(mp: MPArrays) -> int:
    return sum(int((~v.fixed).sum()) for v in mp.vars.values())


def _n_rows(mp: MPArrays) -> int:
    return sum(int(r.active.sum()) for r in mp.rows.values())


def _check_gap(mp: MPArrays) -> None:
    """``apply_aggressive_fixing_for_mcp``: Pyomo fixes variables when free vars
    outnumber active rows; the port only accepts a system that does not need it."""
    gap = _n_free(mp) - _n_rows(mp)
    if gap > 0:
        raise ValueError(
            f"MCP gap {gap} > 0 (free variables {_n_free(mp)}, active rows "
            f"{_n_rows(mp)}): Pyomo would fix variables by matching"
        )


def _fix_sluggish_pft(mp: MPArrays, params: Any) -> None:
    """Dangling ``pft``: free, not mobile, and no active ``eq_xfteq`` cell. (The
    ``xftflag`` guard of the Pyomo step does not exist on this model.)"""
    pft = mp.var("pft")
    eq = mp.row("eq_xfteq")
    active = {eq.keys[i] for i in np.flatnonzero(eq.active)}
    anchored = _in_set(pft, active, 3)
    mf = _in_set_col(pft, 1, set(getattr(params.sets, "mf", []) or []))
    _fix(pft, ~pft.fixed & ~mf & ~anchored)


def _in_set_col(obj: VarArr | RowArr, j: int, names: set[str]) -> np.ndarray:
    inv, vocab = _codes(obj, j)
    want = [vocab[n] for n in names if n in vocab]
    return np.isin(inv, np.array(want, dtype=np.int64))


def _deactivate_fixed_quantity_eqs(mp: MPArrays) -> None:
    """``eq_va`` off where both ``va`` and ``xp`` of the cell are fixed."""
    eq, va, xp = mp.row("eq_va"), mp.var("va"), mp.var("xp")
    pva, pxp = _pos_map(va), _pos_map(xp)
    for i in np.flatnonzero(eq.active):
        j, k = pva.get(eq.keys[i], -1), pxp.get(eq.keys[i], -1)
        if j >= 0 and k >= 0 and va.fixed[j] and xp.fixed[k]:
            eq.active[i] = False


def _check_overdetermined(mp: MPArrays) -> None:
    """``deactivate_zero_unique_var_eqs`` only fires when rows outnumber free
    variables, and then drops rows by matching: the port refuses that case, and a
    system left short of rows (``_check_gap`` runs before the patches drop rows)."""
    n, nv = _n_rows(mp), _n_free(mp)
    if n > nv:
        raise ValueError(
            f"over-determined: {n} active rows for {nv} free variables "
            "(Pyomo would deactivate rows by matching)"
        )
    if n < nv:
        raise ValueError(
            f"under-determined: {n} active rows for {nv} free variables after the "
            "squareness patches (rows dropped with their variables still free: are "
            "the data flags keyed by the right set?)"
        )


def _relative_floors(mp: MPArrays) -> None:
    """``pmt.lo = 1e-3 * pmt.l`` and friends, on the free cells with a positive value."""
    for name in _FLOOR_PRICES + _FLOOR_LEVELS:
        v = mp.var(name)
        with np.errstate(invalid="ignore"):
            want = _PRICE_LB_FACTOR * v.value
            sel = ~v.fixed & (v.value > 0) & (v.lb < want)
        v.lb[sel] = want[sel]


# ── GAMS pairs and matching ─────────────────────────────────────────────────


def _active_keys(row: RowArr) -> list[tuple]:
    return [row.keys[i] for i in np.flatnonzero(row.active)]


def gams_pairs(
    mp: MPArrays, *, params: Any, mode: str = "gtap"
) -> list[tuple[str, str, bool]]:
    """The ``(equation, variable, hard)`` pairs handed to the matching: the GAMS
    ``model`` statement over the active rows, and the CET supply cycle under
    omegax=inf. Hard pairs are locked (and may lie outside the row's incidence)."""
    if mode != "gtap":
        raise NotImplementedError(f"no array matching for mode {mode!r}")
    out: list[tuple[str, str, bool]] = []
    for eq, var in _GAMS_PAIRS:
        row = mp.rows.get(eq)
        if row is None or var not in mp.vars:
            continue
        hard = eq in _GAMS_HARD_PAIR_EQS
        suffix = ("shock",) if eq in _PERIODIC_SHOCK_EQS else ()
        keys = _active_keys(row)
        out += zip(
            _names(eq, keys),
            _names(var, [k + suffix for k in keys]),
            itertools.repeat(hard),
        )

    omegax = getattr(getattr(params, "elasticities", None), "omegax", {}) or {}
    xseq = mp.rows.get("eq_xseq")
    if xseq is None:
        return out
    cells = [
        k
        for k in _active_keys(xseq)
        if len(k) == 3
        and k[-1] in ("check", "shock")
        and omegax.get((k[0], k[1]), float("inf")) == float("inf")
    ]
    for eq, var in _SUPPLY_CYCLE:
        row = mp.rows.get(eq)
        if row is None or var not in mp.vars:
            continue
        live = set(_active_keys(row))
        keys = [k for k in cells if k in live]
        out += zip(_names(eq, keys), _names(var, keys), itertools.repeat(True))
    return out


def _concat(arrays: Any) -> np.ndarray:
    arrays = list(arrays)
    return np.concatenate(arrays) if arrays else np.zeros(0, dtype=bool)


class _LazyCells:
    """``_cells(mp, period)`` on the values captured when the system was squared."""

    def __init__(self, mp: MPArrays, index: CellIndex, period: str) -> None:
        self._snap = []
        for name, v in mp.vars.items():
            idx = index.period_ids("var", name, period, v)
            if len(idx):
                off = index.var_off[name]
                self._snap.append(
                    (
                        name,
                        index.col_names[off : off + len(v.keys)],
                        idx,
                        v.value[idx],
                        v.fixed[idx],
                        v.lb[idx],
                        v.ub[idx],
                    )
                )

    def __call__(self) -> dict[str, tuple[float, bool, float, float]]:
        out = {}
        for _, names, idx, value, fixed, lb, ub in self._snap:
            out.update(
                zip(
                    [names[i] for i in idx.tolist()],
                    zip(
                        value.tolist(),
                        fixed.tolist(),
                        lb.tolist(),
                        ub.tolist(),
                        strict=True,
                    ),
                    strict=True,
                )
            )
        return out


def _gams_pair_ids(
    mp: MPArrays,
    index: CellIndex,
    params: Any,
    row_pos: np.ndarray,
    col_pos: np.ndarray,
    mode: str,
) -> list[tuple[int, int, bool]]:
    """``gams_pairs`` as ``(row position, column position, hard)`` in the squared
    system, the pairs whose row or column is not in it dropped."""
    if mode != "gtap":
        raise NotImplementedError(f"no array matching for mode {mode!r}")
    out: list[tuple[int, int, bool]] = []

    def add(eq: str, var: str, rows: list[int], keys: list[tuple], hard: bool) -> None:
        vpos = index.var_pos(var, mp.vars[var])
        r0, c0 = index.row_off[eq], index.var_off[var]
        for i, k in zip(rows, keys, strict=True):
            j = vpos.get(k)
            if j is None:
                continue
            c = col_pos[c0 + j]
            if c >= 0:
                out.append((int(row_pos[r0 + i]), int(c), hard))

    for eq, var in _GAMS_PAIRS:
        row = mp.rows.get(eq)
        if row is None or var not in mp.vars:
            continue
        suffix = ("shock",) if eq in _PERIODIC_SHOCK_EQS else ()
        act = np.flatnonzero(row.active).tolist()
        add(
            eq,
            var,
            act,
            [row.keys[i] + suffix for i in act],
            eq in _GAMS_HARD_PAIR_EQS,
        )

    omegax = getattr(getattr(params, "elasticities", None), "omegax", {}) or {}
    xseq = mp.rows.get("eq_xseq")
    if xseq is None:
        return out
    cells = [
        k
        for k in _active_keys(xseq)
        if len(k) == 3
        and k[-1] in ("check", "shock")
        and omegax.get((k[0], k[1]), float("inf")) == float("inf")
    ]
    for eq, var in _SUPPLY_CYCLE:
        row = mp.rows.get(eq)
        if row is None or var not in mp.vars:
            continue
        live = {row.keys[i]: i for i in np.flatnonzero(row.active).tolist()}
        keys = [k for k in cells if k in live]
        add(eq, var, [live[k] for k in keys], keys, True)
    return out


def _adjacency(
    row_names: list[str], col_names: list[str], incidence: dict[str, Any]
) -> list[list[int]]:
    """Per row, the free columns of its incidence in the order given, duplicates once.

    Pyomo lists a row's variables in ``identify_variables`` order, skipping fixed ones
    and repeats; with the incidence in that order this is its adjacency."""
    col = {c: j for j, c in enumerate(col_names)}
    out: list[list[int]] = []
    for r in row_names:
        try:
            names = incidence[r]
        except KeyError:
            raise KeyError(f"no incidence for row {r!r}") from None
        seen: set[int] = set()
        cols: list[int] = []
        for name in names:
            j = col.get(name)
            if j is not None and j not in seen:
                seen.add(j)
                cols.append(j)
        out.append(cols)
    return out


# ── The system ──────────────────────────────────────────────────────────────


def _cells(mp: MPArrays, period: str) -> dict[str, tuple[float, bool, float, float]]:
    out = {}
    for name, v in mp.vars.items():
        idx = np.flatnonzero(v.period == period)
        if not len(idx):
            continue
        out.update(
            zip(
                _names(name, [v.keys[i] for i in idx]),
                zip(
                    v.value[idx].tolist(),
                    v.fixed[idx].tolist(),
                    v.lb[idx].tolist(),
                    v.ub[idx].tolist(),
                    strict=True,
                ),
                strict=True,
            )
        )
    return out


def _check_permutation(
    paired: list[int],
    rnames: list[str],
    cnames: list[str],
    pairs: list[tuple[int, int, bool]],
) -> None:
    """Raise ``ValueError`` unless ``paired`` gives every row its own column.

    Two forced pairs on one column (the later overwrites the column's owner, leaving
    the earlier row pointing at it) or a row left unmatched would otherwise yield a
    silently wrong system. The pairing itself is not altered.
    """
    owners: dict[int, list[int]] = {}
    for r, c in enumerate(paired):
        owners.setdefault(c, []).append(r)
    problems = []
    for r in owners.get(-1, []):
        problems.append(f"row {rnames[r]} has no column")
    for c, rows in owners.items():
        if c >= 0 and len(rows) > 1:
            forced = [
                f"{rnames[fr]} -> {cnames[fc]}{' (hard)' if h else ''}"
                for fr, fc, h in pairs
                if fc == c
            ]
            problems.append(
                f"rows {', '.join(rnames[r] for r in rows)} share column "
                f"{cnames[c]} (forced pairs on it: {'; '.join(forced) or 'none'})"
            )
    if problems:
        raise ValueError("the matching is not a permutation: " + " | ".join(problems))


def _paired(
    mp: MPArrays,
    index: CellIndex,
    incidence: Any,
    rid: np.ndarray,
    cid: np.ndarray,
    col_pos: np.ndarray,
    rnames: list[str],
    cnames: list[str],
    pairs: list[tuple[int, int, bool]],
) -> list[int]:
    """The column of every row (``match_rows`` on the adjacency, checked to be a
    permutation). With an ``ArrayIncidence`` it is kept per run (``index.cache``):
    the adjacency is a function of the incidence, the active rows ``rid`` and the free
    columns ``cid``, so the same rows, columns and forced pairs give the same pairing
    (the settle and the check, the shock and its path's sub-steps)."""
    keep = hasattr(incidence, "adjacency")
    if keep:
        key = ("paired", id(incidence))
        sig = (
            rid.tobytes(),
            cid.tobytes(),
            np.asarray(pairs, dtype=np.int64).tobytes(),
        )
        hit = index.cache.get(key)
        if hit is not None and hit[0] is incidence and hit[1] == sig:
            return list(hit[2])
        path = _pairing_path(getattr(incidence, "store", None), incidence.period, sig)
        paired = _load_pairing(path, rnames, cnames, pairs)
        if paired is not None:
            index.cache[key] = (incidence, sig, list(paired))
            return paired
        adjacency = incidence.adjacency(mp, index, rid, col_pos)
    else:
        adjacency = _adjacency(rnames, cnames, incidence)
    paired = match_rows(len(rnames), adjacency, pairs)
    _check_permutation(paired, rnames, cnames, pairs)
    if keep:
        index.cache[key] = (incidence, sig, list(paired))
        _save_pairing(path, paired)
    return paired


def _pairing_path(store: Any, period: str, sig: tuple[bytes, ...]) -> Any:
    """``store/pairing-<period>-<digest>.npy``, or ``None`` without a store.

    The store is the records folder, whose key covers the params, the closure, the
    residual region and ``equilibria/family``; the digest adds the rows, the columns,
    the forced pairs and the matcher's own code (``solver/_matching.py``)."""
    if store is None:
        return None
    import hashlib
    from pathlib import Path

    from equilibria.solver import _matching

    h = hashlib.sha256(Path(_matching.__file__).read_bytes())
    for part in sig:
        h.update(len(part).to_bytes(8, "little"))
        h.update(part)
    return Path(store) / f"pairing-{period}-{h.hexdigest()[:24]}.npy"


def _load_pairing(
    path: Any, rnames: list[str], cnames: list[str], pairs: list
) -> list[int] | None:
    """The saved pairing, or ``None`` when absent, unreadable or not a permutation of
    these rows (then it is recomputed and rewritten)."""
    if path is None or not path.exists():
        return None
    try:
        paired = np.load(path, allow_pickle=False).tolist()
    except (OSError, ValueError):
        return None
    if len(paired) != len(rnames):
        return None
    try:
        _check_permutation(paired, rnames, cnames, pairs)
    except ValueError:
        return None
    return paired


def _save_pairing(path: Any, paired: list[int]) -> None:
    """Write through a temporary file, so a reader never sees half a file."""
    if path is None:
        return
    import os

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.stem}.{os.getpid()}.tmp.npy")
    np.save(tmp, np.asarray(paired, dtype=np.int64))
    os.replace(tmp, path)


@overload
def square(
    mp: MPArrays,
    period: str,
    *,
    params: Any,
    closure: Any,
    incidence: Any,
    exogenous: frozenset[str],
    masks: ConditionalMasks,
    mode: str = "gtap",
    return_mp: Literal[False] = False,
    index: CellIndex | None = None,
) -> Squared: ...


@overload
def square(
    mp: MPArrays,
    period: str,
    *,
    params: Any,
    closure: Any,
    incidence: Any,
    exogenous: frozenset[str],
    masks: ConditionalMasks,
    mode: str = "gtap",
    return_mp: Literal[True],
    index: CellIndex | None = None,
) -> tuple[Squared, MPArrays]: ...


def square(
    mp: MPArrays,
    period: str,
    *,
    params: Any,
    closure: Any,
    incidence: Any,
    exogenous: frozenset[str],
    masks: ConditionalMasks,
    mode: str = "gtap",
    return_mp: bool = False,
    index: CellIndex | None = None,
) -> Squared | tuple[Squared, MPArrays]:
    """The squared system of ``period`` from the arrays after the recipe.

    ``incidence`` maps each active row's name to the names of the variables in its
    body, in Pyomo's ``identify_variables`` order (those not free are ignored, repeats
    count once): the pairs are those of ``structural_matching`` only if the incidence
    has Pyomo's columns and order. ``exogenous`` are the registered ShockBlock
    instruments (``SPRef.exogenous``), which ``apply_closure`` does not overwrite when
    fixed (the numeraire); no default, the model owns them. ``masks`` are the data-flag
    cell masks (``ConditionalMasks.from_params``), built once per run for all periods.
    ``mp`` is not modified. Raises ``ValueError`` where Pyomo would repair the system
    by matching (gap > 0, rows > free variables, not square) and ``KeyError`` for a
    missing component. ``incidence`` is the name dict above or an object with
    ``adjacency(mp, index, rid, col_pos)`` (``squared_system.ArrayIncidence``) giving the
    same lists on integer ids; ``index`` (``CellIndex``, built here if absent or not
    for these arrays) holds the names' sort ranks, built once per run. With ``return_mp`` the mutated copy comes back too (closure
    fixings, deactivated rows and floors of every period, which Pyomo leaves on ``m``)."""
    if mode != "gtap":
        raise NotImplementedError(f"no array squaring for mode {mode!r}")
    if closure is not None and getattr(closure, "name", None) == "altertax":
        raise NotImplementedError("no array squaring for the altertax closure")
    mp = mp.copy()
    for name in _REQUIRED_VARS:
        mp.var(name)
    for name in _REQUIRED_ROWS:
        mp.row(name)

    _initialize_benchmark(mp)
    if closure is not None:
        _apply_closure(mp, closure, exogenous)
    _conditional_fixing(mp, masks)
    _check_gap(mp)
    _fix_sluggish_pft(mp, params)
    _deactivate_fixed_quantity_eqs(mp)
    _check_overdetermined(mp)
    _relative_floors(mp)

    if index is None or not index.fits(mp):
        index = CellIndex(mp)
    row_active = _concat(r.active for r in mp.rows.values())
    col_free = ~_concat(v.fixed for v in mp.vars.values())
    rid = np.flatnonzero(row_active)
    cid = np.flatnonzero(col_free)
    if len(rid) != len(cid):
        # as before the names are only needed for the message
        raise ValueError(
            f"not square: {len(rid)} active rows, {len(cid)} free variables"
        )
    # the names' own string order: the rank of every name in the run's global sort
    rid = rid[np.argsort(index.row_rank[rid], kind="stable")]
    cid = cid[np.argsort(index.col_rank[cid], kind="stable")]
    rnames = [index.row_names[i] for i in rid.tolist()]
    cnames = [index.col_names[i] for i in cid.tolist()]
    row_pos = np.full(len(index.row_names), -1, dtype=np.int64)
    row_pos[rid] = np.arange(len(rid))
    col_pos = np.full(len(index.col_names), -1, dtype=np.int64)
    col_pos[cid] = np.arange(len(cid))

    pairs = _gams_pair_ids(mp, index, params, row_pos, col_pos, mode)
    paired = _paired(mp, index, incidence, rid, cid, col_pos, rnames, cnames, pairs)
    lazy = _LazyCells(mp, index, period)
    sq = Squared(rnames, [cnames[j] for j in paired], lazy)
    sq.ids = SquaredIds(
        index,
        rid,
        cid[np.asarray(paired, dtype=np.int64)],
        [(name, idx, value) for name, _, idx, value, *_ in lazy._snap],
    )
    return (sq, mp) if return_mp else sq
