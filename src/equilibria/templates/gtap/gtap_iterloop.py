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
