"""ifSUB is model condensation (van der Mensbrugghe Table D.1), not economics.

The two ifSUB modes are NOT expected to be bit-identical (GAMS itself differs on the
substituted-out margin block), but the PRIMARY quantity block (xw/xet/xp/... — the
explicitly-solved vars the study compares) must stay consistent across modes: ifSUB
changes representation, not the real quantities. This gate asserts that consistency;
per-mode GAMS fidelity is the coverage matrix's job.

LOCAL-only, gated on the shock GDX fixtures being present.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts/gtap"))
sys.path.insert(0, str(ROOT / "src"))

DATASETS = ["gtap7_3x3", "gtap7_3x4", "gtap7_5x5", "gtap7_10x7", "gtap7_15x10"]
# Primary-block agreement floor across modes. Measured ~99.3% on gtap7_3x3 (the
# residual is the margin-driven bilateral-trade cells that legitimately move +
# solver-tol noise); set conservatively below the measured value.
FLOOR = 0.90


@pytest.mark.parametrize("dataset", DATASETS)
def test_ifsub_primary_block_consistent(dataset):
    from verify_ifsub_equivalence import compare_primary_across_modes

    d = ROOT / "datasets" / dataset / "basedata.har"
    g1 = ROOT / f"tests/fixtures/gtap7/{dataset}/out_gtap_shock_ifsub1.gdx"
    g0 = ROOT / f"tests/fixtures/gtap7/{dataset}/out_gtap_shock_ifsub0.gdx"
    if not (d.exists() and g1.exists() and g0.exists()):
        pytest.skip(f"missing inputs for {dataset}")
    r = compare_primary_across_modes(dataset, tol_rel=1e-4)
    assert r["frac_agree"] >= FLOOR, (
        f"[{dataset}] primary quantity block agrees across ifSUB modes only "
        f"{r['frac_agree'] * 100:.2f}% < {FLOOR * 100:.0f}% — condensation is "
        f"corrupting the primary block. worst: {r['worst'][:5]}"
    )
