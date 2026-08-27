"""GTAP6 symbolic Block units (F7 Task 6+). Blocks land incrementally.

Mirrors ``equilibria.blocks.gtap`` (GTAP7's Block-extraction package) but for
the v6.2 monolith (``scripts/gtap6/_v62_monolith_oracle.py``,
``GTAP6MonolithOracle``). Each block is form+domain gated against that
oracle at the benchmark point (``tests/blocks/gtap6/test_gtap6_blocks_form.py``).

This file is intentionally minimal until the composer (Task 10) exists — it
only re-exports the blocks landed so far. Extended per block task (6-9b).
"""

from __future__ import annotations

from equilibria.blocks.gtap6.trade_armington import TradeArmingtonBlock

__all__ = [
    "TradeArmingtonBlock",
]
