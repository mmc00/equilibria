"""Tests for GTAPv62ModelEquations.build_model() — Phase 2a skeleton."""

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
    """Build a full BOOK3X3 v6.2 Pyomo model skeleton."""
    sets = GTAPv62Sets()
    sets.load_from_har(BOOK3X3_DIR / "SETS.HAR", default_path=BOOK3X3_DIR / "Default.prm")
    params = GTAPv62Parameters()
    params.load_from_har(
        basedata_path=BOOK3X3_DIR / "basedata.har",
        default_prm_path=BOOK3X3_DIR / "Default.prm",
        sets=sets,
    )
    return GTAPv62ModelEquations(sets, params).build_model()


def test_model_builds_without_errors(book3x3_model) -> None:
    """build_model() succeeds and returns a ConcreteModel."""
    from pyomo.environ import ConcreteModel
    assert isinstance(book3x3_model, ConcreteModel)
    # Pyomo wraps single-token names in quotes when displaying; strip
    # them to compare the underlying value.
    assert book3x3_model.name.strip("'\"") == "GTAP_v6.2_Model"


def test_model_has_v62_sets(book3x3_model) -> None:
    """All v6.2 sets are declared on the model."""
    m = book3x3_model
    # Core sets
    assert list(m.r) == ["USA", "EU", "ROW"]
    assert list(m.i) == ["food", "mnfcs", "svces"]
    assert list(m.f) == ["Land", "Labor", "Capital"]
    assert list(m.mf) == ["Labor", "Capital"]
    assert list(m.sf) == ["Land"]
    assert list(m.cgds) == ["CGDS"]
    assert list(m.j) == ["food", "mnfcs", "svces", "CGDS"]  # PROD_COMM
    # v7-only sets must NOT exist
    assert not hasattr(m, "acts")


def test_model_has_core_variables(book3x3_model) -> None:
    """All v6.2 core variables are declared with correct dimensions."""
    m = book3x3_model
    from pyomo.environ import Var

    expected_vars_dim = {
        # Output & prices
        "qo": 2, "ps": 2, "pm": 2,
        # Factors
        "qfe": 3, "pfe": 3, "pf": 2, "qoes": 2,
        # Intermediates
        "qfd": 3, "qfm": 3, "qf": 3, "pfd": 3, "pfm": 3, "pf_int": 3,
        # Household
        "qpd": 2, "qpm": 2, "qp": 2, "pp": 2, "up": 1,
        # Government
        "qgd": 2, "qgm": 2, "qg": 2, "pg": 2, "pgov": 1, "ug": 1,
        # Investment
        "qcgds": 2, "pcgds": 2,
        # Trade
        "qxs": 3, "pms": 3, "pmcif": 3, "pe": 3, "qim": 2, "pim": 2, "qds": 2,
        # Margins
        "qst": 2, "pst": 2, "qtm": 1, "ptmg": 1, "pwmg": 3,
        # Income & capital
        "y": 1, "yp": 1, "yg": 1, "psave": 1, "savf": 1, "kb": 1, "ke": 1,
        # Numeraire & GDP & Walras
        "pgdpwld": 0, "gdpmp": 1, "rgdpmp": 1, "pgdpmp": 1, "walras": 0,
    }
    for name, dim in expected_vars_dim.items():
        var = getattr(m, name, None)
        assert var is not None, f"Missing variable {name!r}"
        assert isinstance(var, Var), f"{name} is not a Var"
        assert var.dim() == dim, f"{name} has dim {var.dim()}, expected {dim}"


def test_model_has_no_v7_specific_variables(book3x3_model) -> None:
    """v7-only variables (intermediate bundle, MAKE, factor-real prices) are absent."""
    m = book3x3_model
    v7_only = [
        "nd", "pnd",          # intermediate bundle (§3)
        "qint", "aint", "pint",
        "qca", "pca", "qc",   # MAKE transformation (§5)
        "tinc", "peb",        # activity-level factor taxes (§12)
        "pefactreal", "pebfactreal",  # post-2017 additions
        "qfa", "qpa", "qga", "qia",  # 'a' suffix v7-only Armington variants
        "ppa", "pga", "pia", "pfa",
        "qms", "pmds",        # v7 trade renames
    ]
    leaked = [name for name in v7_only if hasattr(m, name)]
    assert not leaked, f"v7-specific variables leaked into v6.2 model: {leaked}"


def test_model_has_constraints(book3x3_model) -> None:
    """Phase 2c.1 wires production + demand blocks (~30 equation families)."""
    from pyomo.environ import Constraint

    n = sum(1 for _ in book3x3_model.component_objects(Constraint))
    # Phase 2b: 12, Phase 2c.1: +18 = ~30. Phase 2c.2/2d will add more.
    assert 20 <= n <= 60, (
        f"Expected ~30 constraint families at Phase 2c.1, found {n}."
    )


def test_variable_initial_values_from_benchmark(book3x3_model) -> None:
    """qo, qfe, qpd, etc. are initialized from the benchmark SAM values."""
    from pyomo.environ import value

    m = book3x3_model
    # qo (output) should be initialized at the VOM value
    qo_food_usa = value(m.qo["food", "USA"])
    assert qo_food_usa > 100_000, (
        f"qo(food, USA) = {qo_food_usa:.2f} — expected large positive (VOM-scale)"
    )

    # qfe (factor demand) should match benchmark VFM
    qfe_labor_food_usa = value(m.qfe["Labor", "food", "USA"])
    assert qfe_labor_food_usa > 0

    # Prices initialize at 1.0 (benchmark normalization)
    assert value(m.ps["food", "USA"]) == pytest.approx(1.0)
    assert value(m.pgdpwld) == pytest.approx(1.0)


def test_model_has_calibrated_tax_rates_as_params(book3x3_model) -> None:
    """Derived tax rates are exposed as Pyomo Params, not Vars."""
    from pyomo.environ import Param, value
    m = book3x3_model

    for name in ["tfd", "tfi", "tpd", "tpi", "tgd", "tgi", "tf", "tms", "txs"]:
        param = getattr(m, name, None)
        assert param is not None, f"Missing tax-rate Param {name}"
        assert isinstance(param, Param), f"{name} should be a Param, not a Var"

    # tms[food, USA, EU] = ~37% (the headline BOOK3X3 tariff)
    rate = value(m.tms["food", "USA", "EU"])
    assert 0.30 < rate < 0.45, f"tms(food,USA,EU) = {rate}"


def test_model_variable_count_reasonable(book3x3_model) -> None:
    """Total variable cells in BOOK3X3 (3r × 3i × 3f) is within expected range."""
    from pyomo.environ import Var

    total = sum(len(list(v)) for v in book3x3_model.component_objects(Var))
    # Empirically ~650 cells for BOOK3X3 at Phase 2a.
    # Range allows future variable additions during Phase 2b/2c without
    # being too tight.
    assert 500 < total < 1000, f"Total variable cells = {total}"
