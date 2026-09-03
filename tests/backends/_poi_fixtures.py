"""Shared fixtures for the POI backend tests: real GTAP parameters, smallest dataset.

Fase 0 compares two backends over the same model, so the inputs have to be the real
ones. gtap7_3x3 is the smallest aggregation that still exercises every block.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def tiny_params():
    """``GTAPParameters`` loaded from gtap7_3x3, or skip if the dataset is absent.

    The four-file ``load_from_har`` call matches what the existing parity tests use
    (see tests/blocks/gtap/test_calibrate_no_double_solve.py).
    """
    import pytest

    d = ROOT / "datasets" / "gtap7_3x3"
    if not (d / "basedata.har").exists():
        pytest.skip(f"dataset gtap7_3x3 not present at {d}")

    from equilibria.templates.gtap import GTAPParameters

    params = GTAPParameters()
    params.load_from_har(
        basedata_path=d / "basedata.har",
        sets_path=d / "sets.har",
        default_path=d / "default.prm",
        baserate_path=d / "baserate.har",
    )
    return params


def load_ipopt_or_skip():
    """Ensure POI's Ipopt library is loaded; skip the test if it cannot be found.

    POI does not autoload it. The library comes from Homebrew locally and from the
    system package on Kaggle.
    """
    import pytest

    pytest.importorskip("pyoptinterface")
    from pyoptinterface import ipopt

    if ipopt.is_library_loaded():
        return

    for candidate in (
        "/opt/homebrew/lib/libipopt.dylib",
        "/usr/lib/x86_64-linux-gnu/libipopt.so",
        "libipopt.so",
    ):
        try:
            if ipopt.load_library(candidate):
                return
        except Exception:  # noqa: BLE001 - try the next candidate
            continue

    pytest.skip("Ipopt library not found; POI cannot build a model")
