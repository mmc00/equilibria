"""ifSUB is model condensation (van der Mensbrugghe Table D.1), not economics.

The two ifSUB modes are NOT expected to be bit-identical (GAMS itself differs on the
substituted-out margin block), but the PRIMARY quantity block (xw/xet/xp/... — the
explicitly-solved vars the study compares) must stay consistent across modes: ifSUB
changes representation, not the real quantities. This gate asserts that consistency;
per-mode GAMS fidelity is the coverage matrix's job.

Gated on the GAMS MCP references (fbep=header) being recorded. A dataset whose
family (JAX) route matches GAMS at ifSUB 0 and 1 (test_family_gams_parity, floor >= 99)
is marked `pyomo_solve` and leaves PR CI; scripts/family/regen_photos.py runs it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from equilibria.sources import dataset_dir

# needs_gdxdump: estos tests leen el GDX de referencia con gdxdump; sin el
# binario fallaban con FileNotFoundError en vez de saltarse.
pytestmark = [pytest.mark.integration, pytest.mark.needs_gdxdump]
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts/gtap"))
sys.path.insert(0, str(ROOT / "src"))

from family_twins import pyomo_solve_ifsub_pair  # noqa: E402

DATASETS = ["gtap7_3x3", "gtap7_3x4", "gtap7_5x5", "gtap7_10x7", "gtap7_15x10"]
# Primary-block agreement floor across modes. Measured ~99.3% on gtap7_3x3 (the
# residual is the margin-driven bilateral-trade cells that legitimately move +
# solver-tol noise); set conservatively below the measured value.
FLOOR = 0.90


@pytest.mark.needs_path
@pytest.mark.parametrize(
    "dataset", [pytest.param(d, marks=pyomo_solve_ifsub_pair(d)) for d in DATASETS]
)
def test_ifsub_primary_block_consistent(dataset):
    from verify_ifsub_equivalence import compare_primary_across_modes, ref_gdx

    d = dataset_dir(dataset) / "basedata.har"
    if not (d.exists() and ref_gdx(dataset, 1) and ref_gdx(dataset, 0)):
        pytest.skip(f"missing inputs for {dataset}")
    r = compare_primary_across_modes(dataset, tol_rel=1e-4)
    assert r["frac_agree"] >= FLOOR, (
        f"[{dataset}] primary quantity block agrees across ifSUB modes only "
        f"{r['frac_agree'] * 100:.2f}% < {FLOOR * 100:.0f}% — condensation is "
        f"corrupting the primary block. worst: {r['worst'][:5]}"
    )
