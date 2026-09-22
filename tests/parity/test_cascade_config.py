import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "gtap"))

import pytest
from cascade_config import (
    family,
    resolve_periods,
    resolve_ref_gdx,
    scenario_for,
)

pytestmark = pytest.mark.needs_path


def test_family_split():
    assert family("gtap7_3x3") == "gtap7"
    assert family("9x10") == "bundle"
    assert family("nus333") == "bundle"


def test_scenario_for_gtap7():
    assert scenario_for("gtap7_3x3", "check") == "altertax_check"
    assert scenario_for("gtap7_3x3", "shock") == "altertax_shock"


def test_resolve_periods_drops_base_for_gtap7():
    available, dropped = resolve_periods("gtap7_3x3", ["base", "check", "shock"])
    assert available == ["check", "shock"]
    assert dropped == ["base"]


def test_resolve_periods_drops_check_for_bundle():
    available, dropped = resolve_periods("9x10", ["base", "check", "shock"])
    assert available == ["base", "shock"]
    assert dropped == ["check"]


def test_resolve_ref_gdx_durable_present():
    # El GDX "durable" es una referencia de GAMS que NO se versiona (vive en
    # EQUILIBRIA_REFS_DIR, por defecto ~/proyectos2/equilibria_refs). Sin el no
    # hay nada que resolver, asi que esto es un skip, no un fallo: mismo guard
    # que ya usa tests/parity/test_probe.py. El caso negativo lo cubre igual
    # test_resolve_ref_gdx_missing_is_not_usable, que corre siempre.
    probe = resolve_ref_gdx("gtap7_3x3")
    if not probe.usable:
        pytest.skip(f"reference GDX absent: {probe.note}")
    res = probe
    assert res.usable is True
    assert res.source == "durable"
    assert res.path is not None and res.path.exists()


def test_resolve_ref_gdx_missing_is_not_usable():
    res = resolve_ref_gdx("nus333")  # no durable *_altertax_cd ref
    assert res.usable is False
    assert res.source == "missing"
    assert res.path is None
    assert "NO usable ref" in res.note
