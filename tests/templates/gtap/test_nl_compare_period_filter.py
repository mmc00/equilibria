"""The .nl comparator must read the GAMS rows of the phase it compares.

A GAMS ``gams_shock.nl`` holds BOTH periods (``pmeq(...,base)`` and
``pmeq(...,shock)``): CONVERT writes the whole two-period model. Stripping the
period from the names makes the two rows collide, and the comparator used to
keep the first one, i.e. the BASE row. The shock phase was then compared
against GAMS base rows, and a real shock difference was invisible.

The test feeds the comparator two GAMS files that differ only by the shock
(the 10% import tariff, ``imptx_shock = 1.10 * imptx_base``): the base .nl on
the "Python" side and the shock .nl on the GAMS side, at phase ``shock``.
``eq_pm`` carries ``1 + imptx`` as the ``pmcif`` coefficient, so the
comparator must report it, with the GAMS value equal to the shocked tariff.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from equilibria.sources import find_reference

ROOT = Path(__file__).resolve().parents[3]

sys.path.insert(0, str(ROOT / "scripts/gtap"))

DATASET = "gtap7_3x3"
FAMILY = "pmeq"  # both sides are GAMS files, so GAMS names
CELL = "pmeq[EU_28,Food,ROW]"


def test_shock_phase_reads_gams_shock_rows() -> None:
    from _nl_parser import parse_nl  # ty: ignore[unresolved-import]
    from nl_compare import diff_nl_models  # ty: ignore[unresolved-import]

    folder = find_reference(DATASET, "gams", form="nl", mode="altertax", ifsub=0)
    if folder is None:
        pytest.skip(f"no GAMS .nl recorded for {DATASET}")
        return  # pytest.skip raises; the return tells the type checker

    base = parse_nl(folder / "gams_base.nl")
    shock = parse_nl(folder / "gams_shock.nl")

    result = diff_nl_models(
        base, shock, tol_rel=1e-9, py_period="shock", cell_cap=10_000
    )

    assert result["family_stats"].get(FAMILY, {}).get("n_diff", 0) > 0, (
        "the shock tariff is invisible: the comparator read GAMS base rows"
    )
    cells = {c[0]: c for c in result["cell_diffs"] if c[0] == CELL}
    assert CELL in cells, f"{CELL} not reported"
    _, var, base_c, shock_c, _, _ = cells[CELL]
    assert var.startswith("pmcif[")
    # coefficient is -(1 + imptx): the GAMS side must hold the 10% shock
    assert shock_c == pytest.approx(-(1.0 + 1.10 * (-base_c - 1.0)), rel=1e-6)
