"""Unit test for the Python-side ``check`` phase of the GTAP .nl comparator.

The multi-period altertax CHECK period is the compStat betaCal→check
transition: altertax CD elasticities (esubva=1, esubd=esubm=0.95, etrae=1,
omegaf=1) + altertax closure, with the BASE period frozen as a snapshot and
NO tariff shock (imptx stays at base values).

This test spies on ``GTAPModelEquations`` construction during
``build_python_nls(phases=["base", "check"], ...)`` and asserts the check
build is:
  * counterfactual (``is_counterfactual=True``) with a ``t0_snapshot`` set,
  * built with altertax CD elasticities (esubva == 1.0, distinguishing it
    from the plain base GTAP build), and
  * does NOT apply the tariff shock (imptx unchanged from base).

It also confirms the base model is built exactly once and reused as the
check t0 snapshot (not rebuilt).

Runs in a few seconds; skips cleanly if the HAR dataset is absent.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
DATASETS_DIR = ROOT / "datasets"

sys.path.insert(0, str(ROOT / "scripts/gtap"))

DATASET = "gtap7_3x3"


@pytest.fixture
def har_dir() -> Path:
    d = DATASETS_DIR / DATASET
    if not (d / "basedata.har").exists():
        pytest.skip(f"HAR dataset {DATASET!r} not available at {d}")
    return d


def test_check_phase_builds_counterfactual_altertax(har_dir: Path, tmp_path: Path, monkeypatch) -> None:
    """The check phase builds a CD-altertax counterfactual with a base snapshot."""
    from equilibria.templates.gtap import gtap_model_equations as gme
    from equilibria.templates.gtap.gtap_contract import GTAPClosureConfig
    from nl_compare import build_python_nls

    real_init = gme.GTAPModelEquations.__init__
    calls: list[dict] = []

    def spy_init(self, sets, params, *args, **kwargs):
        # Capture the construction context per period, keyed by build order.
        # esubva is a {(a, r): value} dict on the elasticities container.
        esubva_vals = list(params.elasticities.esubva.values())
        # imptx tax rates (to confirm the check does NOT apply the shock).
        imptx_vals = {k: float(v or 0.0) for k, v in params.taxes.imptx.items()}
        calls.append(
            {
                "is_counterfactual": kwargs.get("is_counterfactual", False),
                "t0_snapshot": kwargs.get("t0_snapshot", None),
                "esubva_vals": esubva_vals,
                "imptx_vals": imptx_vals,
            }
        )
        return real_init(self, sets, params, *args, **kwargs)

    monkeypatch.setattr(gme.GTAPModelEquations, "__init__", spy_init)

    build_python_nls(
        phases=["base", "check"],
        out_dir=tmp_path,
        closure_config=GTAPClosureConfig(if_sub=False),
        har_dir=har_dir,
    )

    # The check phase must produce a python_check.nl.
    assert (tmp_path / "python_check.nl").exists(), "check phase did not write python_check.nl"

    # Base must be built exactly once (reused as the check t0 snapshot).
    assert len(calls) == 2, (
        f"expected exactly 2 GTAPModelEquations builds (base + check), got {len(calls)}"
    )
    base_call, check_call = calls[0], calls[1]

    # Base build: ordinary (not counterfactual), no snapshot.
    assert base_call["is_counterfactual"] is False
    assert base_call["t0_snapshot"] is None

    # Check build: counterfactual with a base snapshot.
    assert check_call["is_counterfactual"] is True, (
        "check phase must be is_counterfactual=True"
    )
    assert check_call["t0_snapshot"] is not None, (
        "check phase must pass the base model as t0_snapshot"
    )

    # Check build uses altertax CD elasticities — esubva collapses to 1.0,
    # which distinguishes it from the calibrated base GTAP value (≠ 1).
    assert check_call["esubva_vals"], "esubva should be non-empty"
    assert all(abs(v - 1.0) < 1e-9 for v in check_call["esubva_vals"]), (
        "check phase must apply altertax CD elasticities (esubva == 1.0); "
        f"got {check_call['esubva_vals'][:5]}"
    )
    # And the base build must NOT have CD esubva (else the test is vacuous).
    assert any(abs(v - 1.0) > 1e-6 for v in base_call["esubva_vals"]), (
        "base esubva should be the calibrated (non-CD) value, not all 1.0"
    )

    # Check build must NOT apply the 10% tariff shock — imptx == base imptx.
    assert check_call["imptx_vals"] == base_call["imptx_vals"], (
        "check phase must NOT shock imptx (it stays at base values)"
    )
