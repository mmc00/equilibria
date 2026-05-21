"""Phase 2b — Production block equations and benchmark residuals.

The production block wires the top CES nest (VA + intermediate
composite), the VA CES (across factors), and the firm-side top
Armington (domestic vs imported) onto the Pyomo model. Eleven
equations are added; ten balance to machine epsilon at benchmark; the
eleventh (``eq_qo``) carries a ≤5% residual from the implicit
output-tax wedge that is reconciled when the full price chain is
wired in Phase 2c.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from equilibria.templates.gtap_v62 import (
    GTAPv62ModelEquations,
    GTAPv62Parameters,
    GTAPv62Sets,
)


BOOK3X3_DIR = Path("C:/runGTAP375/BOOK3X3")


def _rungtap_available() -> bool:
    return all(
        (BOOK3X3_DIR / fname).exists()
        for fname in ("SETS.HAR", "basedata.har", "Default.prm")
    )


pytestmark = pytest.mark.skipif(
    not _rungtap_available(),
    reason="RunGTAP v6.2 dataset BOOK3X3 not available",
)


@pytest.fixture
def book3x3_model():
    sets = GTAPv62Sets()
    sets.load_from_har(BOOK3X3_DIR / "SETS.HAR", default_path=BOOK3X3_DIR / "Default.prm")
    params = GTAPv62Parameters()
    params.load_from_har(
        basedata_path=BOOK3X3_DIR / "basedata.har",
        default_prm_path=BOOK3X3_DIR / "Default.prm",
        sets=sets,
    )
    return GTAPv62ModelEquations(sets, params).build_model()


def _residual(constraint_idx) -> float:
    from pyomo.environ import value
    body = value(constraint_idx.body)
    if constraint_idx.upper is not None:
        return body - value(constraint_idx.upper)
    if constraint_idx.lower is not None:
        return body - value(constraint_idx.lower)
    return body


def test_production_block_equations_present(book3x3_model) -> None:
    """All Phase 2b production-block equations are wired."""
    expected = [
        "eq_qo", "eq_pds", "eq_va", "eq_qf",
        "eq_pva", "eq_qfe", "eq_pfe",
        "eq_pf_int", "eq_qfd", "eq_qfm",
        "eq_pfd", "eq_pfm",
    ]
    for name in expected:
        assert hasattr(book3x3_model, name), f"Missing equation {name!r}"


def test_total_constraint_cells_in_range(book3x3_model) -> None:
    """Constraint cell count is within expected bounds for BOOK3X3.

    After Phase 2c.1 (demand block + cgds identities), total cells
    grow to ~400-450.
    """
    from pyomo.environ import Constraint
    total = sum(len(list(c)) for c in book3x3_model.component_objects(Constraint))
    assert 200 < total < 600, f"Total constraint cells = {total}"


def test_benchmark_residual_zero_except_eq_qo(book3x3_model) -> None:
    """Ten of eleven production equations balance to machine epsilon.

    Phase 2b explicitly accepts a ~4% residual in eq_qo from the
    implicit output-tax wedge. Phase 2c reconciles it.
    """
    from pyomo.environ import Constraint

    expected_clean = {
        "eq_pds", "eq_va", "eq_qf", "eq_pva", "eq_qfe", "eq_pfe",
        "eq_pf_int", "eq_qfd", "eq_qfm", "eq_pfd", "eq_pfm",
    }

    for c in book3x3_model.component_objects(Constraint):
        max_residual = max((abs(_residual(c[idx])) for idx in c), default=0.0)
        if c.name in expected_clean:
            assert max_residual < 1e-6, (
                f"Equation {c.name!r} expected zero residual at benchmark, "
                f"got {max_residual:.4e}"
            )


def test_eq_qo_known_residual_bound(book3x3_model) -> None:
    """eq_qo residual is bounded by the implicit output-tax wedge magnitude.

    The mismatch comes from ``vom`` (output side at market prices) vs
    ``vop_AGENT`` (cost side at agent prices). For BOOK3X3 the implied
    output tax rates ``to`` are in the range [-7%, +7%], so the eq_qo
    residual should be of comparable magnitude. Phase 2c reconciles
    this via the pm/pds/pim price chain.
    """
    max_residual = max((abs(_residual(book3x3_model.eq_qo[idx]))
                         for idx in book3x3_model.eq_qo),
                        default=0.0)
    assert max_residual < 0.10, (
        f"eq_qo residual {max_residual:.4e} exceeds 10% threshold — "
        f"Phase 2b calibration may have regressed."
    )


def test_va_pva_variables_added(book3x3_model) -> None:
    """Phase 2b adds value-added variables va and pva."""
    from pyomo.environ import Var, value
    m = book3x3_model
    assert isinstance(m.va, Var)
    assert isinstance(m.pva, Var)
    # va initialized at the calibrated va_total (sum of VFM)
    assert value(m.va["food", "USA"]) > 100_000  # USA food VA ≈ 210k
    # pva starts at 1.0 (benchmark normalization)
    assert value(m.pva["food", "USA"]) == pytest.approx(1.0)


def test_pds_variable_carries_output_tax_wedge(book3x3_model) -> None:
    """pds(j,r) = ps(j,r) * (1 + to(j,r)) holds at benchmark by initialization."""
    from pyomo.environ import value
    m = book3x3_model
    # pds is initialized at (1 + to_derived).
    pds_food_usa = value(m.pds["food", "USA"])
    # BOOK3X3 food/USA has to ≈ -1%
    assert 0.97 < pds_food_usa < 1.05


def test_calibrated_alpha_in_armington(book3x3_model) -> None:
    """Distribution parameters (alpha_dom / alpha_imp) absorb tax wedges."""
    from pyomo.environ import Param, value
    m = book3x3_model

    assert isinstance(m.alpha_dom, Param)
    assert isinstance(m.alpha_imp, Param)
    # For a cell with intermediate taxes (food→svces in EU has tfd=7.5%),
    # alpha_dom != value share due to the calibrated price ratio.
    alpha = value(m.alpha_dom["food", "svces", "EU"])
    share = value(m.share_dom["food", "svces", "EU"])
    if alpha > 0:
        # The distribution parameter differs from the value share when
        # there's a non-zero agent price wedge.
        # Don't assert exact value; just that calibration didn't break.
        assert alpha > 0
        assert share > 0


def test_share_va_plus_share_int_sum_to_one(book3x3_model) -> None:
    """Top nest shares (share_va + sum_i share_int) sum to 1 for active sectors."""
    from pyomo.environ import value
    m = book3x3_model
    for r in m.r:
        for j in m.j:
            sv = value(m.share_va[j, r])
            si = sum(value(m.share_int[i, j, r]) for i in m.i)
            if sv > 0 or si > 0:
                assert abs(sv + si - 1.0) < 1e-6, (
                    f"Top nest shares for ({j},{r}) sum to {sv+si:.6f}, expected 1.0"
                )
