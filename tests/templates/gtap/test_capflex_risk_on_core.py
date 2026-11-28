"""``solve_block_model`` is gone (PR 3a of gtap-on-core). capFlex's ``risk`` no longer
solves a capFix twin: it comes from the data (``test_capflex_risk.py``, MAR-251)."""

from __future__ import annotations


def test_solve_block_model_is_gone() -> None:
    from equilibria.templates.gtap import gtap_block_model

    assert not hasattr(gtap_block_model, "solve_block_model")
