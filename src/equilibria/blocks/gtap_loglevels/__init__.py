"""GTAP log-levels blocks: OUR levels model (equilibria.blocks.gtap) re-expressed in
log form by wrapping each levels block. Distinct from gtap_logvalue (the mivanic Julia
port). Same economics/closure as the levels blocks (savf capital account, Fisher price
index, gy-indexed tax streams) — only the equation FORM is log(lhs)==log(rhs) where
both sides are positive.
"""

from equilibria.blocks.gtap import GTAP_BLOCK_ORDER as _LEVELS_ORDER

from ._logwrap import log_wrap_block

# Log-wrapped mirror of the levels block order (dependency order, leaf → closure).
GTAP_LOGLEVELS_BLOCK_ORDER = [log_wrap_block(cls) for cls in _LEVELS_ORDER]

__all__ = ["GTAP_LOGLEVELS_BLOCK_ORDER", "log_wrap_block"]
