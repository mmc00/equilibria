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
from typing import Any

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching

from equilibria.family.mp_arrays import MPArrays, RowArr, Squared, VarArr

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
    """(xFlag over (r, a), makb support over (r, a, i))."""
    bm = params.benchmark
    vom = getattr(bm, "vom", {}) or getattr(bm, "vom_i", {})
    if not vom:  # derive from the make flows without touching the benchmark
        vom = {}
        for (r, a, _i), val in getattr(bm, "makb", {}).items():
            if val is not None and abs(float(val)) > 1e-12:
                vom[(r, a)] = vom.get((r, a), 0.0) + float(val)
    x_rai = {
        k
        for k, val in getattr(bm, "makb", {}).items()
        if val is not None and abs(float(val)) > 1e-12
    }
    return _positive(vom), x_rai


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
    variables, and then drops rows by matching: the port refuses that case."""
    n, nv = _n_rows(mp), _n_free(mp)
    if n > nv:
        raise ValueError(
            f"over-determined: {n} active rows for {nv} free variables "
            "(Pyomo would deactivate rows by matching)"
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


def _edges(
    row_names: list[str], col_names: list[str], incidence: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray]:
    """(rows, cols) of the incidence restricted to the free columns."""
    try:
        lists = [incidence[r] for r in row_names]
    except KeyError as e:
        raise KeyError(f"no incidence for row {e.args[0]!r}") from None
    counts = np.fromiter(map(len, lists), dtype=np.int64, count=len(lists))
    flat = np.array(list(itertools.chain.from_iterable(lists)), dtype=str)
    rows = np.repeat(np.arange(len(lists)), counts)
    cols = np.array(col_names, dtype=str)  # sorted by name, like the rows
    if not len(flat) or not len(cols):
        return rows[:0], rows[:0]
    pos = np.minimum(np.searchsorted(cols, flat), len(cols) - 1)
    ok = cols[pos] == flat
    return rows[ok], pos[ok].astype(np.int64)


def _match(
    n: int,
    rows: np.ndarray,
    cols: np.ndarray,
    pair_rows: np.ndarray,
    pair_cols: np.ndarray,
    pair_hard: np.ndarray,
) -> np.ndarray:
    """Column of each row: hard pairs locked, soft pairs a starting point, a maximum
    matching on the rest, leftover columns to the rows left unmatched.

    The semantics of ``structural_matching``; not its exact pairs."""
    nc = n
    code = np.sort(rows * nc + cols)
    adjacent = np.isin(pair_rows * nc + pair_cols, code)
    valid = pair_hard | adjacent

    def first(r: np.ndarray, c: np.ndarray) -> np.ndarray:
        """The pairs that claim a still free row and a still free column, first wins."""
        keep = np.zeros(len(r), dtype=bool)
        if len(r):
            i = np.sort(np.unique(r, return_index=True)[1])
            keep[i[np.unique(c[i], return_index=True)[1]]] = True
        return keep

    left = np.full(n, -1, dtype=np.int64)
    taken = np.zeros(nc, dtype=bool)
    locked = np.zeros(n, dtype=bool)

    h = np.flatnonzero(valid & pair_hard)
    h = h[first(pair_rows[h], pair_cols[h])]
    left[pair_rows[h]] = pair_cols[h]
    taken[pair_cols[h]] = True
    locked[pair_rows[h]] = True

    s = np.flatnonzero(valid & ~pair_hard)
    s = s[~locked[pair_rows[s]] & ~taken[pair_cols[s]]]
    s = s[first(pair_rows[s], pair_cols[s])]
    seed = left.copy()
    seed[pair_rows[s]] = pair_cols[s]
    seeded_cols = taken.copy()
    seeded_cols[pair_cols[s]] = True

    def maximum(free_r: np.ndarray, free_c: np.ndarray) -> np.ndarray:
        """Maximum matching over the edges between the given free rows and columns."""
        sel = free_r[rows] & free_c[cols]
        g = csr_matrix((np.ones(int(sel.sum())), (rows[sel], cols[sel])), shape=(n, nc))
        return maximum_bipartite_matching(g, perm_type="column")

    # starting point: hard + soft pairs, completed on the rest
    rest = maximum(seed < 0, ~seeded_cols)
    start = np.where(seed >= 0, seed, rest)
    # the optimum of the unlocked sub-problem; the starting point stands if it reaches it
    best = maximum(~locked, ~taken)
    best = np.where(locked, left, best)
    left = start if (start >= 0).sum() >= (best >= 0).sum() else best

    free_cols = np.flatnonzero(np.bincount(left[left >= 0], minlength=nc) == 0)
    unmatched = np.flatnonzero(left < 0)
    left[unmatched] = free_cols[: len(unmatched)]
    return left


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


def square(
    mp: MPArrays,
    period: str,
    *,
    params: Any,
    closure: Any,
    incidence: dict[str, Any],
    exogenous: frozenset[str],
    masks: ConditionalMasks,
    mode: str = "gtap",
) -> Squared:
    """The squared system of ``period`` from the arrays after the recipe.

    ``incidence`` maps each active row's name to the names of the variables in its
    body (those not free are ignored). ``exogenous`` are the registered ShockBlock
    instruments (``SPRef.exogenous``), which ``apply_closure`` does not overwrite when
    fixed (the numeraire); no default, the model owns them. ``masks`` are the data-flag
    cell masks (``ConditionalMasks.from_params``), built once per run for all periods.
    ``mp`` is not modified. Raises ``ValueError`` where Pyomo would repair the system
    by matching (gap > 0, rows > free variables, not square) and ``KeyError`` for a
    missing component."""
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

    rnames, cnames = [], []
    for name, row in mp.rows.items():
        rnames += _names(name, _active_keys(row))
    for name, v in mp.vars.items():
        cnames += _names(name, [v.keys[i] for i in np.flatnonzero(~v.fixed)])
    rnames.sort()
    cnames.sort()
    if len(rnames) != len(cnames):
        raise ValueError(
            f"not square: {len(rnames)} active rows, {len(cnames)} free variables"
        )

    rows, cols = _edges(rnames, cnames, incidence)
    r_index = {r: i for i, r in enumerate(rnames)}
    c_index = {c: i for i, c in enumerate(cnames)}
    pairs = [
        (r_index[e], c_index[v], h)
        for e, v, h in gams_pairs(mp, params=params, mode=mode)
        if e in r_index and v in c_index
    ]
    pr, pc, ph = (
        np.array([p[k] for p in pairs], dtype=t)
        for k, t in ((0, np.int64), (1, np.int64), (2, bool))
    )
    paired = _match(len(rnames), rows, cols, pr, pc, ph)
    return Squared(rnames, [cnames[j] for j in paired], _cells(mp, period))
