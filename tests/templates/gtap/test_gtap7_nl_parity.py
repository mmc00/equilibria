"""GTAP7 .nl structural parity gate.

For each dataset in tests/fixtures/gtap7/, regenerates the Python .nl and
compares its coefficients family-by-family against the GAMS reference .nl
fixture (generated once via NEOS and committed to the repo).

This test does NOT solve the model — it only builds the Pyomo NL and diffs
its coefficients. A change in any equation, parameter loading, or set
structure will surface here immediately.

Datasets covered (in git):
  gtap7_3x3, gtap7_3x4, gtap7_5x5, gtap7_10x7, gtap7_15x10

Large datasets (gtap7_20x41) are excluded from CI because the GAMS fixture
is too large for git; run manually with --dataset gtap7_20x41.

Run:
    uv run pytest tests/templates/gtap/test_gtap7_nl_parity.py -v
    uv run pytest tests/templates/gtap/test_gtap7_nl_parity.py -v -k gtap7_10x7
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
FIXTURES_DIR = ROOT / "tests/fixtures/gtap7"
DATASETS_DIR = ROOT / "datasets"

sys.path.insert(0, str(ROOT / "scripts/gtap"))

from coverage_matrix import nl_rows  # noqa: E402


def _available_datasets() -> list[str]:
    """Return dataset names that have both fixture NL files and HAR data.

    No longer feeds DATASETS — kept to document the on-disk contract and
    allow manual inspection.  The authoritative CI list comes from the
    coverage matrix (nl_rows()).
    """
    result = []
    for d in sorted(FIXTURES_DIR.iterdir()):
        if not d.is_dir():
            continue
        has_fixtures = (d / "gams_base.nl").exists() and (d / "gams_shock.nl").exists()
        has_data = (DATASETS_DIR / d.name / "basedata.har").exists()
        if has_fixtures and has_data:
            result.append(d.name)
    return result


# Datasets whose .nl gate runs in CI, per the coverage matrix.
# Restricted to gtap7_* names: nus333/9x10 are kind="gtap" ci_status="ci"
# but their parity is covered by dedicated tests, not this .nl fixture gate.
DATASETS = [
    r.dataset for r in nl_rows()
    if r.ci_status == "ci" and r.dataset.startswith("gtap7_")
]


@pytest.mark.parametrize("dataset", DATASETS)
def test_gtap7_nl_parity(dataset: str, tmp_path: Path) -> None:
    """Python .nl coefficients match GAMS fixture for base/check/shock phases.

    The "check" phase (multi-period altertax CD step) is only diffed when a
    gams_check.nl fixture is present for the dataset.
    """
    from nl_compare import build_python_nls, diff_nl_models, diff_bounds
    from _nl_parser import parse_nl
    from _parity_datasets import DATASETS as DS_REGISTRY
    from equilibria.templates.gtap.gtap_contract import GTAPClosureConfig

    ds = DS_REGISTRY.get(dataset)
    if ds is None:
        pytest.skip(f"Dataset {dataset!r} not in registry")

    fixture_dir = FIXTURES_DIR / dataset
    if not (fixture_dir / "gams_base.nl").exists():
        pytest.skip(f"no .nl fixtures for {dataset}")
    har_dir = DATASETS_DIR / dataset
    closure_config = GTAPClosureConfig(if_sub=False)

    # The "check" phase (multi-period altertax CD step) is opt-in per dataset:
    # it is compared only when a gams_check.nl fixture exists, so datasets that
    # only carry base/shock fixtures keep passing unchanged.
    phases = ["base", "shock"]
    if (fixture_dir / "gams_check.nl").exists():
        phases.insert(1, "check")

    build_python_nls(
        phases=phases,
        out_dir=tmp_path,
        closure_config=closure_config,
        har_dir=har_dir,
    )

    tol_rel = 1e-4
    for phase in phases:
        py_nl = parse_nl(tmp_path / f"python_{phase}.nl")
        gams_nl = parse_nl(fixture_dir / f"gams_{phase}.nl")

        result = diff_nl_models(py_nl, gams_nl, tol_rel=tol_rel, py_period=phase)
        b_diffs, _, _ = diff_bounds(py_nl, gams_nl, tol_rel=tol_rel)

        failures = [
            f"{fam}: {st['n_diff']} diffs (max_rel={st['max_rel']:.2e})"
            for fam, st in result["family_stats"].items()
            if st.get("n_diff", 0) > 0 and not st.get("_structural_fp")
        ]
        assert not failures, (
            f"[{dataset}/{phase}] Coefficient diffs vs GAMS fixture:\n"
            + "\n".join(f"  {f}" for f in failures)
        )
        assert len(b_diffs) == 0, (
            f"[{dataset}/{phase}] {len(b_diffs)} variables with different bounds"
        )
