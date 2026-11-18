"""The capFlex ``risk`` fallback (the capFix twin-solve, no GAMS GDX) runs on the
core driver and still gives main's numbers; ``solve_block_model`` is gone (PR 3a of
gtap-on-core). Oracle: ``tests/blocks/gtap/golden/calibrate_base_main.json``."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from equilibria.sources import has_dataset

ROOT = Path(__file__).resolve().parents[3]
GOLDEN = ROOT / "tests/blocks/gtap/golden/calibrate_base_main.json"


def _recorder():
    spec = importlib.util.spec_from_file_location(
        "record_calibrate_base", ROOT / "scripts/dev/record_calibrate_base.py"
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.skipif(not has_dataset("gtap7_3x3"), reason="gtap7_3x3 not present")
def test_capflex_twin_solve_equals_main(monkeypatch) -> None:
    monkeypatch.setenv("EQUILIBRIA_SEED_CACHE_DISABLE", "1")
    import equilibria.templates.gtap.gtap_multiperiod_driver as old

    def _refuse(*a, **k):
        raise AssertionError("the capFlex twin-solve reached solve_multiperiod")

    monkeypatch.setattr(old, "solve_multiperiod", _refuse)
    want = json.loads(GOLDEN.read_text(encoding="utf-8"))["capflex_risk"]
    got = _recorder()._risk("gtap7_3x3", 0)
    assert got == want["gtap7_3x3-ifsub0"]


def test_solve_block_model_is_gone() -> None:
    from equilibria.templates.gtap import gtap_block_model

    assert not hasattr(gtap_block_model, "solve_block_model")


@pytest.mark.skipif(not has_dataset("gtap7_3x3"), reason="gtap7_3x3 not present")
def test_a_failed_twin_solve_warns(monkeypatch, caplog) -> None:
    """The twin-solve's failure still falls back to risk = 1 (``{}``), as on main,
    but no longer silently: it is logged as a WARNING naming the error."""
    import logging

    from equilibria.core import exercise

    def _fail(self):
        raise RuntimeError("twin-solve broke")

    monkeypatch.setattr(exercise.Exercise, "run", _fail)
    rec = _recorder()
    p = rec._params("gtap7_3x3")
    from equilibria.templates.gtap.gtap_block_model import _calibrate_capflex_risk

    with caplog.at_level(logging.WARNING):
        risk = _calibrate_capflex_risk(
            p, p.sets, rec.closure(0, "capFlex"), list(p.sets.r)[-1], ref_gdx=None
        )
    assert risk == {}
    assert any(
        r.levelno == logging.WARNING and "twin-solve broke" in r.getMessage()
        for r in caplog.records
    )
