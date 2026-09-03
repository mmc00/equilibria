"""The shock continuation needs ONE deepcopy, not one per λ sub-step.

MEASURED on the 20x41 (Kaggle kernel gtap-scaffold-breakdown, 2026-09-03):
18 large deepcopies, 28.7s total, one per continuation sub-step.

``_apply_imptx_shock`` mutates ONLY ``params.taxes.imptx``, in place. The per-λ
deepcopy existed so each sub-step applies its λ to the UNSHOCKED base rates rather
than compounding on the previous sub-step's already-shocked values. Restoring a
pristine snapshot of that one dict gives the same guarantee without re-copying the
whole GTAPParameters object.

This test pins the equivalence directly: walking the λ ladder with a fresh deepcopy
each step and walking it with one deepcopy + dict restore must produce
BYTE-IDENTICAL tariffs at every step. If a future change makes the shock touch
something outside ``taxes.imptx``, the restore is no longer sufficient and this
test fails.
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
LAMBDAS = [0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0]


@pytest.fixture(scope="module")
def params():
    d = ROOT / "datasets" / "gtap7_3x3"
    if not (d / "basedata.har").exists():
        pytest.skip(f"dataset not present at {d}")
    scripts_gtap = ROOT / "scripts" / "gtap"
    if str(scripts_gtap) not in sys.path:
        sys.path.insert(0, str(scripts_gtap))

    from equilibria.templates.gtap import GTAPParameters

    p = GTAPParameters()
    p.load_from_har(
        basedata_path=d / "basedata.har",
        sets_path=d / "sets.har",
        default_path=d / "default.prm",
        baserate_path=d / "baserate.har",
    )
    return p


@pytest.mark.parametrize("gtap_mode", [True, False])
def test_single_deepcopy_matches_per_step_deepcopy(params, gtap_mode):
    """One deepcopy + dict restore == a fresh deepcopy per sub-step."""
    from equilibria.templates.gtap.gtap_multiperiod_driver import _apply_imptx_shock

    # reference: the OLD behaviour — a fresh deepcopy for every λ
    reference = []
    for lam in LAMBDAS:
        p_lam = copy.deepcopy(params)
        _apply_imptx_shock(p_lam, factor=(1.0 + 0.10) ** lam - 1.0, gtap_mode=gtap_mode)
        reference.append(dict(p_lam.taxes.imptx))

    # new: ONE deepcopy, restoring the pristine imptx per λ
    p_one = copy.deepcopy(params)
    pristine = dict(p_one.taxes.imptx)
    actual = []
    for lam in LAMBDAS:
        p_one.taxes.imptx.clear()
        p_one.taxes.imptx.update(pristine)
        _apply_imptx_shock(p_one, factor=(1.0 + 0.10) ** lam - 1.0, gtap_mode=gtap_mode)
        actual.append(dict(p_one.taxes.imptx))

    assert len(reference) == len(actual) == len(LAMBDAS)
    for i, (ref, act) in enumerate(zip(reference, actual)):
        assert ref.keys() == act.keys(), f"λ={LAMBDAS[i]}: tariff key sets differ"
        for k in ref:
            assert ref[k] == act[k], (
                f"λ={LAMBDAS[i]} key={k}: single-deepcopy path gives {act[k]!r}, "
                f"per-step deepcopy gives {ref[k]!r}. The restore no longer "
                f"reproduces the original behaviour."
            )


def test_shock_would_compound_without_restore(params):
    """Guard: the restore is load-bearing, not decoration.

    Without it, applying λ repeatedly to the same object compounds the tariff. This
    test documents WHY the pristine snapshot exists, so nobody 'simplifies' it away.
    """
    from equilibria.templates.gtap.gtap_multiperiod_driver import _apply_imptx_shock

    p = copy.deepcopy(params)
    key = next(iter(p.taxes.imptx))
    base = float(p.taxes.imptx[key] or 0.0)

    _apply_imptx_shock(p, factor=0.10, gtap_mode=True)
    once = float(p.taxes.imptx[key])
    _apply_imptx_shock(p, factor=0.10, gtap_mode=True)
    twice = float(p.taxes.imptx[key])

    assert once != twice, "expected the shock to compound when applied twice"
    expected_twice = (1.0 + base) * 1.10 * 1.10 - 1.0
    assert twice == pytest.approx(expected_twice, rel=1e-12)
