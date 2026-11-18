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
