"""Phase 2c.1 — Household + government demand block, CGDS identities."""

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
    return all((BOOK3X3_DIR / f).exists()
               for f in ("SETS.HAR", "basedata.har", "Default.prm"))


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


def test_household_block_equations_present(book3x3_model) -> None:
    """Household demand equations are wired."""
    for name in [
        "eq_pp", "eq_qp", "eq_qpd", "eq_qpm",
        "eq_ppd", "eq_ppm", "eq_pcons", "eq_up",
    ]:
        assert hasattr(book3x3_model, name), f"Missing household equation {name!r}"


def test_government_block_equations_present(book3x3_model) -> None:
    """Government demand equations are wired."""
    for name in [
        "eq_pg", "eq_qg", "eq_qgd", "eq_qgm",
        "eq_pgd", "eq_pgm", "eq_pgov", "eq_ug",
    ]:
        assert hasattr(book3x3_model, name), f"Missing government equation {name!r}"


def test_investment_identities_present(book3x3_model) -> None:
    """CGDS identity equations link qcgds/pcgds to qo/ps."""
    assert hasattr(book3x3_model, "eq_qcgds")
    assert hasattr(book3x3_model, "eq_pcgds")


def test_clean_demand_equations_balance_at_benchmark(book3x3_model) -> None:
    """Most Phase 2c.1 demand-block equations balance to machine epsilon.

    Phase 2c.2 added pim_0 to the calibration cascade. With pim_0 ≠ 1,
    a few equations (eq_pp, eq_qpm, eq_pg, eq_qgm) carry small benchmark
    residuals from the market-value-share calibration approximation.
    Those are tested separately with a bounded tolerance below.
    """
    demand_eqs_clean = [
        "eq_qp", "eq_qpd", "eq_ppd", "eq_ppm",
        "eq_pcons", "eq_up",
        "eq_qg", "eq_qgd", "eq_pgd", "eq_pgm",
        "eq_pgov", "eq_ug",
        "eq_qcgds", "eq_pcgds",
    ]
    for name in demand_eqs_clean:
        eq = getattr(book3x3_model, name)
        max_residual = max((abs(_residual(eq[idx])) for idx in eq), default=0.0)
        assert max_residual < 1e-6, (
            f"Equation {name!r} expected zero residual at benchmark, "
            f"got {max_residual:.4e}"
        )


def test_armington_residuals_now_clean(book3x3_model) -> None:
    """After Phase 2d SAM-consistent trade chain calibration, the
    household and government Armington equations (eq_pp, eq_pg, eq_qpm,
    eq_qgm) balance to machine epsilon.

    Previously these had non-zero residuals because pim_0 was computed
    via a price chain that didn't match the v6.2 SAM identity
    VIWS = VXWD + sum_m VTWR. Phase 2d uses pmcif_0 = VIWS/VXWD
    directly, which makes pim_0 = sum_VIMS/sum_VXWD consistent
    throughout the calibration cascade.
    """
    for name in ["eq_pp", "eq_pg", "eq_qpm", "eq_qgm"]:
        eq = getattr(book3x3_model, name)
        max_residual = max((abs(_residual(eq[idx])) for idx in eq), default=0.0)
        assert max_residual < 1e-6, (
            f"{name!r} should balance after Phase 2d fix, got {max_residual:.4e}"
        )


def test_household_armington_variables_added(book3x3_model) -> None:
    """ppd, ppm are added as separate household agent prices."""
    from pyomo.environ import Var
    assert isinstance(book3x3_model.ppd, Var)
    assert isinstance(book3x3_model.ppm, Var)
    assert isinstance(book3x3_model.pcons, Var)


def test_government_armington_variables_added(book3x3_model) -> None:
    """pgd, pgm are added as separate gov agent prices."""
    from pyomo.environ import Var
    assert isinstance(book3x3_model.pgd, Var)
    assert isinstance(book3x3_model.pgm, Var)


def test_cd_shares_sum_to_one(book3x3_model) -> None:
    """Household and government CD budget shares sum to 1 per region."""
    from pyomo.environ import value
    m = book3x3_model
    for r in m.r:
        h_total = sum(value(m.share_hhd_cd[i, r]) for i in m.i)
        g_total = sum(value(m.share_gov_cd[i, r]) for i in m.i)
        assert abs(h_total - 1.0) < 1e-6, (
            f"Household CD shares in {r} sum to {h_total:.6f}, expected 1.0"
        )
        assert abs(g_total - 1.0) < 1e-6, (
            f"Government CD shares in {r} sum to {g_total:.6f}, expected 1.0"
        )


def test_yp_yg_initialized_from_benchmark(book3x3_model) -> None:
    """Household and government incomes start at total expenditure."""
    from pyomo.environ import value
    for r in book3x3_model.r:
        yp = value(book3x3_model.yp[r])
        yg = value(book3x3_model.yg[r])
        assert yp > 100_000, f"yp({r}) = {yp:.2f} seems too small for BOOK3X3"
        assert yg > 10_000, f"yg({r}) = {yg:.2f} seems too small"


def test_cgds_quantities_match_output(book3x3_model) -> None:
    """qcgds(cg,r) equals qo(cg,r) by identity."""
    from pyomo.environ import value
    for cg in book3x3_model.cgds:
        for r in book3x3_model.r:
            qcgds = value(book3x3_model.qcgds[cg, r])
            qo = value(book3x3_model.qo[cg, r])
            assert abs(qcgds - qo) < 1e-6


def test_total_constraint_cells_phase_2c1(book3x3_model) -> None:
    """Phase 2c.1+2c.2 brings constraint families to ~41 with ~536 cells."""
    from pyomo.environ import Constraint
    n_families = sum(1 for _ in book3x3_model.component_objects(Constraint))
    n_cells = sum(len(list(c)) for c in book3x3_model.component_objects(Constraint))
    assert 25 <= n_families <= 50, f"Got {n_families} equation families"
    assert 350 <= n_cells <= 600, f"Got {n_cells} constraint cells"
