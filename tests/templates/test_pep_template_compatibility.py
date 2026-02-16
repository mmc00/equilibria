"""Compatibility checks for classic PEP templates."""

from __future__ import annotations

from equilibria.babel.sam import SAM
from equilibria.templates.pep_1r import PEP1R


def test_pep1r_load_sam_returns_legacy_accounts() -> None:
    """PEP1R should expose legacy SAM account labels used by classic blocks."""
    template = PEP1R()
    sam = template.load_sam()

    assert isinstance(sam, SAM)
    assert "USK" in sam.data.index
    assert "AGR" in sam.data.columns
    assert sam.data.shape[0] == sam.data.shape[1]


def test_pep1r_create_model_without_calibration() -> None:
    """Template assembly should still work independently of SAM calibration mode."""
    template = PEP1R()
    model = template.create_model(calibrate=False)

    assert len(model.blocks) > 0
    assert model.statistics.variables > 0
    assert model.statistics.equations > 0
