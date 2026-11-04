"""Nothing is left under tests/**/fixtures/ (ADR-0010, phase 4).

Each phase-4 task moves one family of fixtures to data/ and removes its paths
from ``ALLOWED_UNTIL_PHASE_4``; phase 3b removes the GEMPACK ones.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# GEMPACK results and what is computed from them: regenerated on Windows (3a-2),
# moved in phase 3b.
ALLOWED_UNTIL_PHASE_3B_PREFIX = "tests/fixtures/gtap7_gempack/"
ALLOWED_UNTIL_PHASE_3B = frozenset({"tests/fixtures/nus333_bienestar_gempack.json.gz"})

ALLOWED_UNTIL_PHASE_4 = frozenset(
    {
        "tests/babel/gdx/fixtures/sam_v2_0_expected.csv",
        "tests/babel/gdx/fixtures/SAM-V2_0.gdx",
        "tests/compare/fixtures/eq_burfisher_tbl45a_levels.json.gz",
        "tests/compare/fixtures/eq_burfisher_tbl46a_levels.json.gz",
        "tests/compare/fixtures/eq_gtap7_3x3_levels.json.gz",
        "tests/compare/fixtures/gtap3x3_minimo.yaml",
        "tests/compare/fixtures/make_eq_burfisher_levels.py",
        "tests/compare/fixtures/make_eq_gtap7_levels.py",
        "tests/fixtures/bigskip_test.gdx",
        "tests/fixtures/gdxdump/pep_results.gdx.gdxdump.json.gz",
        "tests/fixtures/gdxdump/pep2_parameters.gdx.gdxdump.json.gz",
        "tests/fixtures/generate_5d_test.gms",
        "tests/fixtures/generate_6d_test.gms",
        "tests/fixtures/generate_compression_tests.gms",
        "tests/fixtures/generate_gdx_fixtures.gms",
        "tests/fixtures/generate_minimal_set.gms",
        "tests/fixtures/generate_multi_col_set.gms",
        "tests/fixtures/generate_multidim_test.gms",
        "tests/fixtures/generate_set_tests.gms",
        "tests/fixtures/gtap_logvalue/julia_3x3_calibrated.csv",
        "tests/fixtures/gtap7_10x7_settled_seed.json.gz",
        "tests/fixtures/gtap7_3x3_punto_sin_shock_3f3fe5f.json.gz",
        "tests/fixtures/gtap7_3x3_residuos_sin_shock_3f3fe5f.json.gz",
        "tests/fixtures/multidim_test.gdx",
        "tests/fixtures/nus333_bienestar_gams_levels.json.gz",
        "tests/fixtures/nus333_desempleo_gams_levels.json.gz",
        "tests/fixtures/nus333_tbl94_gams_levels.json.gz",
        "tests/fixtures/patterns_test.gdx",
        "tests/fixtures/period_prep_snapshot.json",
        "tests/fixtures/pyomo_sol/ipopt_3.14.19.sol",
        "tests/fixtures/pyomo_sol/path_4.7.03.sol",
        "tests/fixtures/set_2d_cartesian.gdx",
        "tests/fixtures/set_2d_empty.gdx",
        "tests/fixtures/set_2d_full.gdx",
        "tests/fixtures/set_2d_minimal.gdx",
        "tests/fixtures/set_2d_multi_col.gdx",
        "tests/fixtures/set_2d_sparse.gdx",
        "tests/fixtures/set_2d_with_param.gdx",
        "tests/fixtures/set_2d_with_text.gdx",
        "tests/fixtures/set_3d.gdx",
        "tests/fixtures/set_4d.gdx",
        "tests/fixtures/simple.gms",
        "tests/fixtures/simple_test.gdx",
        "tests/fixtures/skip_test.gdx",
        "tests/fixtures/sparse_test.gdx",
        "tests/fixtures/test_5d.gdx",
        "tests/fixtures/test_6d.gdx",
        "tests/fixtures/test_arithmetic_compressed.gdx",
        "tests/fixtures/test_arithmetic.gdx",
        "tests/fixtures/test_geometric_compressed.gdx",
        "tests/fixtures/test_geometric.gdx",
        "tests/fixtures/test_growth.gdx",
        "tests/fixtures/test_mixed.gdx",
        "tests/fixtures/test_small_arithmetic.gdx",
        "tests/fixtures/test_small_geometric.gdx",
        "tests/fixtures/test_sparse.gdx",
        "tests/fixtures/variables_equations_test.gdx",
        "tests/sam_tools/fixtures/simple_mip.xlsx",
    }
)


def _tracked_fixtures() -> set[str]:
    out = subprocess.run(
        ["git", "ls-files", "tests/**/fixtures/**", "tests/fixtures/**"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return {p for p in out.splitlines() if p}


def _allowed(path: str) -> bool:
    return (
        path.startswith(ALLOWED_UNTIL_PHASE_3B_PREFIX)
        or path in ALLOWED_UNTIL_PHASE_3B
        or path in ALLOWED_UNTIL_PHASE_4
    )


def test_no_files_under_tests_fixtures() -> None:
    extra = sorted(p for p in _tracked_fixtures() if not _allowed(p))
    assert not extra, f"move these to data/ (ADR-0010): {extra}"


def test_allowance_has_no_stale_entries() -> None:
    tracked = _tracked_fixtures()
    stale = sorted((ALLOWED_UNTIL_PHASE_4 | ALLOWED_UNTIL_PHASE_3B) - tracked)
    assert not stale, f"moved already; drop them from the allowance: {stale}"
