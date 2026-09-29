"""``lambdava_shock`` is rejected up front where it cannot be honoured.

Both checks must fire BEFORE any build or solve work: a run that would silently
drop the technology shock (altertax) or walk the wrong shock (the tariff
continuation) must fail in milliseconds, not after minutes of solving.
"""

from __future__ import annotations

import pytest

from equilibria.templates.gtap import gtap_multiperiod_driver as driver


def _explode():
    raise AssertionError("validation must run before loading the solver")


def test_rejected_in_altertax_mode(monkeypatch):
    monkeypatch.setattr(driver, "_load_run_gtap", _explode)
    with pytest.raises(ValueError, match="mode='gtap'"):
        driver.solve_multiperiod(
            None, None, None, mode="altertax", lambdava_shock={("USA", "SER"): 1.1}
        )


def test_rejected_with_tariff_continuation(monkeypatch):
    monkeypatch.setattr(driver, "_load_run_gtap", _explode)
    monkeypatch.setenv("EQUILIBRIA_GTAP_SHOCK_CONTINUATION", "0.5,1.0")
    with pytest.raises(ValueError, match="CONTINUATION"):
        driver.solve_multiperiod(
            None, None, None, mode="gtap", lambdava_shock={("USA", "SER"): 1.1}
        )


def test_rejects_a_non_positive_factor(monkeypatch):
    monkeypatch.setattr(driver, "_load_run_gtap", _explode)
    with pytest.raises(ValueError, match="must be > 0"):
        driver.solve_multiperiod(
            None, None, None, mode="gtap", lambdava_shock={("USA", "SER"): 0.0}
        )
