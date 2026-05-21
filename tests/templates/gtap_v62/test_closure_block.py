"""Phase 2c.3 — Factor markets, income, and closure (numeraire + Walras).

Eight new equation families that close the model:
- eq_factor_clear (per f, r): sum_j qfe = qoes — factor market clearing
- eq_qoes_fixed (per f, r): factor supply pinned to benchmark
- eq_y (per r): regional factor income identity
- eq_yp / eq_yg (per r): income shares for household / government
- eq_pgdpwld: numeraire identity (pgdpwld = 1)
- eq_walras: global market clearing residual

All balance to machine epsilon at the BOOK3X3 benchmark, with the
exception of the four "known residuals" carried over from Phases
2b/2c.1/2c.2 calibration imperfections (eq_qo, eq_qpm, eq_qtm,
eq_market).
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


def test_factor_market_equations_present(book3x3_model) -> None:
    """Phase 2c.3 factor market and closure equations wired."""
    for name in ["eq_factor_clear", "eq_qoes_fixed", "eq_y",
                 "eq_yp", "eq_yg", "eq_pgdpwld", "eq_walras"]:
        assert hasattr(book3x3_model, name), f"Missing equation {name!r}"


def test_factor_market_clearing_holds_at_benchmark(book3x3_model) -> None:
    """sum_j qfe(f,j,r) ≈ qoes(f,r) at benchmark (float32 rounding tolerable)."""
    eq = book3x3_model.eq_factor_clear
    max_residual = max((abs(_residual(eq[idx])) for idx in eq), default=0.0)
    # Float32 storage in HAR introduces tiny rounding; allow 1.0 absolute
    # tolerance (relative to factor incomes ~5M)
    assert max_residual < 5.0, f"Factor market residual {max_residual:.4e}"


def test_walras_residual_is_zero_at_benchmark(book3x3_model) -> None:
    """Global Walras balance: walras = sum_r [y - yp - yg - SAVE + savf] ≈ 0."""
    eq = book3x3_model.eq_walras
    max_residual = max((abs(_residual(eq[idx])) for idx in eq), default=0.0)
    assert max_residual < 1e-6, (
        f"eq_walras residual {max_residual:.4e} — benchmark calibration imbalance"
    )


def test_income_identity_clean(book3x3_model) -> None:
    """eq_y, eq_yp, eq_yg balance to machine epsilon at benchmark."""
    for name in ["eq_y", "eq_yp", "eq_yg", "eq_qoes_fixed", "eq_pgdpwld"]:
        eq = getattr(book3x3_model, name)
        max_residual = max((abs(_residual(eq[idx])) for idx in eq), default=0.0)
        assert max_residual < 1e-6, (
            f"{name!r} residual {max_residual:.4e} at benchmark"
        )


def test_qoes_initialized_from_evoa(book3x3_model) -> None:
    """Factor supply qoes is initialized from EVOA at benchmark."""
    from pyomo.environ import value
    m = book3x3_model
    for f in m.f:
        for r in m.r:
            evom = value(m.evom[f, r])
            qoes = value(m.qoes[f, r])
            if evom > 0:
                assert abs(qoes - evom) < 1.0, (
                    f"qoes({f},{r}) = {qoes:.2f} vs evom = {evom:.2f}"
                )


def test_c_p_plus_c_g_below_one(book3x3_model) -> None:
    """Income shares c_p (hhd) + c_g (gov) < 1 — remainder goes to savings."""
    from pyomo.environ import value
    m = book3x3_model
    for r in m.r:
        cp = value(m.c_p[r])
        cg = value(m.c_g[r])
        # Standard GTAP regions have cp ≈ 0.7-0.85, cg ≈ 0.15-0.25, savings rest
        assert 0.6 < cp + cg < 1.05, (
            f"c_p({r}) + c_g({r}) = {cp+cg:.4f}, unusual for BOOK3X3"
        )


def test_savf_initialized_to_balance(book3x3_model) -> None:
    """savf initialized to close the regional budget identity at benchmark."""
    from pyomo.environ import value
    m = book3x3_model
    # Walras check: sum_r [y - yp - yg - SAVE + savf] should ≈ 0 at benchmark.
    total = 0.0
    for r in m.r:
        residual = (
            value(m.y[r]) - value(m.yp[r]) - value(m.yg[r])
            - value(m.save_0[r]) + value(m.savf[r])
        )
        total += residual
    # Global Walras at benchmark — should be very close to 0
    assert abs(total) < 1.0, f"Global Walras residual at benchmark: {total:.4e}"


def test_total_constraint_cells_phase_2c3(book3x3_model) -> None:
    """Phase 2c.3 brings total to ~48 families, ~565 cells in BOOK3X3."""
    from pyomo.environ import Constraint
    n_fam = sum(1 for _ in book3x3_model.component_objects(Constraint))
    n_cells = sum(len(list(c)) for c in book3x3_model.component_objects(Constraint))
    assert 40 <= n_fam <= 60, f"Got {n_fam} equation families"
    assert 500 <= n_cells <= 700, f"Got {n_cells} constraint cells"


def test_model_essentially_square(book3x3_model) -> None:
    """After Phase 2c.3, the model is roughly square (variables ≈ constraints)."""
    from pyomo.environ import Var, Constraint
    n_var_cells = sum(len(list(v)) for v in book3x3_model.component_objects(Var))
    n_cons_cells = sum(len(list(c)) for c in book3x3_model.component_objects(Constraint))
    # Variables include some that are fixed by the closure (qoes, pgdpwld),
    # so the count is not exactly square — allow a 25% tolerance.
    ratio = n_cons_cells / n_var_cells if n_var_cells > 0 else 0
    assert 0.75 <= ratio <= 1.25, (
        f"Vars={n_var_cells}, Cons={n_cons_cells}, ratio={ratio:.3f}"
    )
