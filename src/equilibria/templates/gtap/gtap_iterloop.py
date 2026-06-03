"""Multi-period lagged-state fixing for GTAP recursive-dynamic solve.

Mirrors GAMS ``iterloop.gms:151-182`` — the block that fixes period-``(tsim-1)``
endogenous values into period-``tsim`` before each MCP solve.

In the Python design (Option C: one ``ConcreteModel`` per period), each
period is built from the same :class:`GTAPParameters` and therefore the
indexed Pyomo Vars on the previous and new models share index sets. This
module copies the previous model's ``.value`` into the new model and pins
each entry with ``.fix()``.

The variable list (24 names) is the enumeration in
``docs/superpowers/inventory/2026-06-02-gtap-lagged-state-map.md`` Summary
section. ``pmuv`` is intentionally excluded because in the default closure
it is a calibrated mutable ``Param``, not a ``Var`` (the map flags it ``⏸ skip``).
The two GAMS renames (``lambdaio→aioall``, ``xa→xaa``) are already resolved
in the list.
"""
from __future__ import annotations

import logging

from pyomo.core.base.var import Var
from pyomo.environ import ConcreteModel

logger = logging.getLogger(__name__)

# Order matches GAMS iterloop.gms:152-181.
LAGGED_VARS: tuple[str, ...] = (
    "axp", "lambdand", "lambdava", "aioall", "lambdaf",
    "pf", "xf",
    "pa", "xaa", "pe", "pefob", "pmcif", "pm", "xw", "ptmg",
    "psave", "pi",
    "uh",
    "pabs",
    "pfact", "pwfact",
    "gdpmp", "rgdpmp", "pgdpmp",
)


def fix_lagged_state(
    new_model: ConcreteModel,
    prev_model: ConcreteModel,
    lagged_var_names: tuple[str, ...] = LAGGED_VARS,
) -> int:
    """Copy prev_model's solved Var values into new_model and ``.fix()`` them.

    Parameters
    ----------
    new_model
        The freshly-built ``ConcreteModel`` for the current period (``tsim``).
    prev_model
        The previously-solved ``ConcreteModel`` (period ``tsim-1``).
    lagged_var_names
        The Var attribute names to copy and fix. Defaults to
        :data:`LAGGED_VARS`.

    Returns
    -------
    int
        Total number of ``(var, index)`` tuples copied and fixed.
    """
    total_fixed = 0

    for name in lagged_var_names:
        new_var = getattr(new_model, name, None)
        prev_var = getattr(prev_model, name, None)

        if new_var is None:
            logger.warning(
                "fix_lagged_state: '%s' missing from new_model; skipping",
                name,
            )
            continue
        if prev_var is None:
            logger.warning(
                "fix_lagged_state: '%s' missing from prev_model; skipping",
                name,
            )
            continue

        if not isinstance(new_var, Var):
            logger.warning(
                "fix_lagged_state: '%s' on new_model is %s, not Var; skipping",
                name, type(new_var).__name__,
            )
            continue
        if not isinstance(prev_var, Var):
            logger.warning(
                "fix_lagged_state: '%s' on prev_model is %s, not Var; skipping",
                name, type(prev_var).__name__,
            )
            continue

        for idx in new_var:
            try:
                src_val = prev_var[idx].value
            except KeyError:
                logger.warning(
                    "fix_lagged_state: index %r missing from prev_model.%s; "
                    "skipping", idx, name,
                )
                continue
            if src_val is None:
                logger.debug(
                    "fix_lagged_state: prev_model.%s[%r] is None; skipping",
                    name, idx,
                )
                continue
            new_var[idx].value = src_val
            new_var[idx].fix()
            total_fixed += 1

    return total_fixed


# ===========================================================================
# Phase 3 — in-place period fixings (single-model architecture).
# ===========================================================================
#
# These helpers mirror NEOS ``scripts/iterloop.gms`` operating on a single
# ``ConcreteModel`` that holds every period (``t``) as the last index dim
# of every Var/Param. Each helper restricts itself to a single period
# ``tsim`` (and, for lagged-state, the previous period ``tprev``).
#
# The full driver :func:`apply_iterloop_fixings` is what the sequential
# solver calls before each PATH solve.

from pyomo.environ import value as pyo_value  # noqa: E402  (intentionally late)

# -- Param.fixed shim --------------------------------------------------------
# Pyomo's :class:`ParamData` (slot-based) has no ``fixed`` attribute, but
# the iterloop interface mirrors GAMS ``.fx`` semantics on policy
# instruments (which are Params under our build). Install a read-only
# :class:`property` on the class that consults a per-Param set we manage,
# ``_iterloop_fixed_indices``. ``_fix_tax_instruments`` populates that
# set; tests assert ``param[idx].fixed is True``.
def _install_param_fixed_shim() -> None:
    import pyomo.core.base.param as _pm
    if "fixed" in vars(_pm.ParamData):
        return  # already installed

    def _get_fixed(self):
        parent = self.parent_component()
        idx_set = getattr(parent, "_iterloop_fixed_indices", None)
        if not idx_set:
            return False
        return self.index() in idx_set

    _pm.ParamData.fixed = property(_get_fixed)


_install_param_fixed_shim()


#: (var_name, dims_without_t) for the iterloop lagged-state block
#: (iterloop.gms L151-182). Notes:
#:   - ``uh`` is single-household ``(r,)`` in Python (no ``h`` dim).
#:   - ``xaa`` is the GAMS ``xa`` rename; ``aioall`` is the GAMS ``lambdaio``
#:     rename.
#:   - ``pmuv`` and ``pwfact`` are scalar-per-period (``()``).
LAGGED_VARS_WITH_DIMS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("axp", ("r", "a")),
    ("lambdand", ("r", "a")),
    ("lambdava", ("r", "a")),
    ("aioall", ("r", "i", "a")),
    ("lambdaf", ("r", "f", "a")),
    ("pf", ("r", "f", "a")),
    ("xf", ("r", "f", "a")),
    ("pa", ("r", "i", "aa")),
    ("xaa", ("r", "i", "aa")),
    ("pe", ("r", "i", "rp")),
    ("pefob", ("r", "i", "rp")),
    ("pmcif", ("r", "i", "rp")),
    ("pm", ("r", "i", "rp")),
    ("xw", ("r", "i", "rp")),
    ("ptmg", ("m",)),
    ("psave", ("r",)),
    ("pi", ("r",)),
    ("uh", ("r",)),
    ("pabs", ("r",)),
    ("pmuv", ()),
    ("pwfact", ()),
    ("pfact", ("r",)),
    ("gdpmp", ("r",)),
    ("rgdpmp", ("r",)),
    ("pgdpmp", ("r",)),
)


def apply_iterloop_fixings(
    model,
    tsim: str,
    *,
    t_set: tuple[str, ...],
    sets,
    params,
    flags: dict,
    first_year: str = "base",
) -> None:
    """Mirror of NEOS ``iterloop.gms`` — fix vars/params for period ``tsim``.

    This is the in-place fixings driver for the *single-model* design
    (every period lives on a single ``ConcreteModel`` with ``t`` as the
    last index dim of every Var/Param). It must be called BEFORE the
    PATH solve for period ``tsim``.

    Parameters
    ----------
    model
        The combined ConcreteModel (all periods).
    tsim
        Period name we are about to solve.
    t_set
        Ordered tuple of all periods, e.g. ``("base", "check", "shock")``.
    sets, params
        :class:`GTAPParameters` and its ``sets`` namespace.
    flags
        Dict of activity flags (see :func:`run_gtap._build_flags_dict`).
    first_year
        First period in ``t_set`` (defaults to ``"base"``); lagged-state
        fixings are skipped for this period.
    """
    _fix_tax_instruments(model, tsim, params)
    _fix_trade_margins(model, tsim)
    _set_price_lower_bounds(model, tsim, t_set)
    _fix_inactive_flows(model, tsim, t_set, flags)
    _fix_lagged_state(model, tsim, t_set, first_year)


# --- 3.1 tax instruments ---------------------------------------------------


_TAX_PARAM_NAMES: tuple[str, ...] = (
    "fctts", "fcttx", "prdtx", "exptx", "imptx",
    "dtxshft", "mtxshft", "rtxshft",
)


def _fix_tax_instruments(model, tsim: str, params) -> None:
    """iterloop.gms L23-37 — record policy Params at ``tsim`` as fixed.

    Pyomo's slot-based ``ParamData`` has no native ``.fix()``; we use the
    :func:`_install_param_fixed_shim` property to expose
    ``param[idx].fixed`` via a per-Param ``_iterloop_fixed_indices`` set.
    GAMS-style: the Param value is the calibrated value (or whatever
    ``apply_shock`` set), and the ``fixed`` marker tells the solver
    pipeline not to treat it as free. Because Params have no Newton
    degree-of-freedom, the marker is purely declarative.
    """
    for prm_name in _TAX_PARAM_NAMES:
        prm = getattr(model, prm_name, None)
        if prm is None:
            continue
        idx_set = getattr(prm, "_iterloop_fixed_indices", None)
        if idx_set is None:
            idx_set = set()
            prm._iterloop_fixed_indices = idx_set
        for idx in prm.keys():
            if isinstance(idx, tuple):
                if idx[-1] != tsim:
                    continue
            else:
                if idx != tsim:
                    continue
            idx_set.add(idx)


# --- 3.1 trade margins ------------------------------------------------------


def _fix_trade_margins(model, tsim: str) -> None:
    """iterloop.gms L57 — trade-margin params at ``tsim`` are calibrated and
    therefore held constant. Because ``tmarg`` is declared as a Param with
    a static initializer (no ``mutable=True``), it's already immutable —
    we just confirm the symbol is present in the model. This helper is a
    no-op placeholder that documents intent; the assertion lives in tests.
    """
    # tmarg, amgm, lambdamg are Params; nothing to fix at runtime.
    return None


# --- 3.2 price lower bounds -------------------------------------------------


_PRICE_LB_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("px",    ("r", "a")),
    ("pva",   ("r", "a")),
    ("pnd",   ("r", "a")),
    ("p",     ("r", "a", "i")),
    ("ps",    ("r", "i")),
    ("pdp",   ("r", "i", "aa")),
    ("pmp",   ("r", "i", "aa")),
    ("pa",    ("r", "i", "aa")),
    ("pmt",   ("r", "i")),
    ("pe",    ("r", "i", "rp")),
    ("pefob", ("r", "i", "rp")),
    ("pmcif", ("r", "i", "rp")),
    ("pm",    ("r", "i", "rp")),
    ("pet",   ("r", "i")),
    ("pd",    ("r", "i")),
    ("pwmg",  ("r", "i", "rp")),
    ("pf",    ("r", "f", "a")),
    ("pfa",   ("r", "f", "a")),
)


def _set_price_lower_bounds(model, tsim: str, t_set: tuple[str, ...]) -> None:
    """iterloop.gms L61-88 — ``var.lo = 0.001 * var_prev.l`` for price vars.

    For the first period there is no prior period, so this is a no-op.
    Pyomo's :class:`pyomo.core.base.var.Var` may be a Var or an Expression
    (e.g. ``model.p`` is an Expression in our build); we silently skip
    anything that isn't a Var.
    """
    from .gtap_model_equations import prev_t
    from pyomo.core.base.var import Var as _PyoVar

    tprev = prev_t(tsim, t_set)
    if tprev is None:
        return

    for var_name, _dims in _PRICE_LB_RULES:
        var = getattr(model, var_name, None)
        if var is None:
            continue
        if not isinstance(var, _PyoVar):
            continue
        for idx in list(var.keys()):
            if not isinstance(idx, tuple):
                continue
            if idx[-1] != tprev:
                continue
            tsim_idx = (*idx[:-1], tsim)
            if tsim_idx not in var:
                continue
            v_prev = pyo_value(var[idx])
            if v_prev is None:
                v_prev = 1.0
            var[tsim_idx].setlb(0.001 * float(v_prev))


# --- 3.3 inactive flows -----------------------------------------------------


#: (flag_name, qty_var_name_or_None, price_var_name, dims_without_t).
#: When a flag entry is ``False``, the corresponding qty is fixed to 0 and
#: the price is fixed to its baseline (``t_set[0]``) value.
_INACTIVE_FLOW_RULES: tuple[tuple[str, str | None, str, tuple[str, ...]], ...] = (
    ("ndFlag",  "nd",   "pnd",  ("r", "a")),
    ("vaFlag",  "va",   "pva",  ("r", "a")),
    ("xfFlag",  "xf",   "pf",   ("r", "f", "a")),
    ("xpFlag",  None,   "px",   ("r", "a")),
    ("xaFlag",  "xaa",  "pa",   ("r", "i", "aa")),
    ("xFlag",   "x",    "p",    ("r", "a", "i")),
    ("xsFlag",  "xs",   "ps",   ("r", "i")),
    ("xmtFlag", "xmt",  "pmt",  ("r", "i")),
    ("xwFlag",  "xw",   "pe",   ("r", "i", "rp")),
    ("tmgFlag", "xwmg", "pwmg", ("r", "i", "rp")),
    ("xdFlag",  "xds",  "pd",   ("r", "i")),
    ("xetFlag", "xet",  "pet",  ("r", "i")),
    ("xftFlag", "xft",  "pft",  ("r", "f")),
    ("alphad",  "xd",   "pdp",  ("r", "i", "aa")),
    ("alpham",  "xm",   "pmp",  ("r", "i", "aa")),
)

#: When ``xwFlag`` is False, also pin these bilateral price aliases.
_EXTRA_XW_PRICES: tuple[str, ...] = ("pefob", "pmcif", "pm")


def _fix_inactive_flows(
    model,
    tsim: str,
    t_set: tuple[str, ...],
    flags: dict,
) -> None:
    """iterloop.gms L92-147 — fix qty=0, price=base-value for inactive flows.

    ``flags`` maps a flag name (e.g. ``"xfFlag"``) to a dict
    ``{idx_without_t: bool}``. Entries with value ``False`` (or ``0``)
    indicate inactive flows. We pin the corresponding quantity to zero
    and the corresponding price to its baseline (period ``t_set[0]``)
    value (the previously-calibrated value).
    """
    from pyomo.core.base.var import Var as _PyoVar
    t0 = t_set[0]

    for flag_name, qty_name, price_name, _dims in _INACTIVE_FLOW_RULES:
        flag_dict = flags.get(flag_name, {})
        if not flag_dict:
            continue
        for idx_no_t, is_active in flag_dict.items():
            if is_active:
                continue
            # Fix quantity to zero.
            if qty_name is not None:
                qv = getattr(model, qty_name, None)
                if isinstance(qv, _PyoVar):
                    qi = (*idx_no_t, tsim)
                    if qi in qv:
                        qv[qi].fix(0.0)
            # Fix price to baseline value.
            if price_name is not None:
                pv = getattr(model, price_name, None)
                if isinstance(pv, _PyoVar):
                    pi0 = (*idx_no_t, t0)
                    pits = (*idx_no_t, tsim)
                    if pi0 in pv and pits in pv:
                        v = pyo_value(pv[pi0])
                        if v is None:
                            v = 1.0
                        pv[pits].fix(float(v))

        # xwFlag-controlled flows have extra price siblings.
        if flag_name == "xwFlag":
            for extra in _EXTRA_XW_PRICES:
                pv = getattr(model, extra, None)
                if not isinstance(pv, _PyoVar):
                    continue
                for idx_no_t, is_active in flag_dict.items():
                    if is_active:
                        continue
                    pi0 = (*idx_no_t, t0)
                    pits = (*idx_no_t, tsim)
                    if pi0 in pv and pits in pv:
                        v = pyo_value(pv[pi0])
                        if v is None:
                            v = 1.0
                        pv[pits].fix(float(v))


# --- 3.4 lagged state -------------------------------------------------------


def _fix_lagged_state(
    model,
    tsim: str,
    t_set: tuple[str, ...],
    first_year: str,
) -> None:
    """iterloop.gms L151-182 — fix LAGGED_VARS at ``tprev`` to their
    current ``.value``.

    This is what makes periods talk to each other: solving period ``tsim``
    treats the previously-solved period's endogenous state as fixed data.
    For ``tsim == first_year`` there is nothing to do (the base period
    is anchored by calibration).
    """
    from .gtap_model_equations import prev_t
    from pyomo.core.base.var import Var as _PyoVar

    if tsim == first_year:
        return
    tprev = prev_t(tsim, t_set)
    if tprev is None:
        return

    import pyomo.core.base.param as _pm

    for var_name, _dims in LAGGED_VARS_WITH_DIMS:
        var = getattr(model, var_name, None)
        if var is None:
            continue
        is_var = isinstance(var, _PyoVar)
        is_param = isinstance(var, _pm.Param)
        if not (is_var or is_param):
            continue
        param_fixed_set: set | None = None
        if is_param:
            param_fixed_set = getattr(var, "_iterloop_fixed_indices", None)
            if param_fixed_set is None:
                param_fixed_set = set()
                var._iterloop_fixed_indices = param_fixed_set
        for idx in list(var.keys()):
            if isinstance(idx, tuple):
                if idx[-1] != tprev:
                    continue
            else:
                if idx != tprev:
                    continue
            cv = pyo_value(var[idx])
            if cv is None:
                raise RuntimeError(
                    f"lagged var {var_name}[{idx}] has no value"
                )
            if is_var:
                var[idx].fix(float(cv))
            else:
                # mutable Param: re-set value (idempotent) and record as fixed.
                try:
                    var[idx].set_value(float(cv))
                except Exception:
                    pass
                param_fixed_set.add(idx)
