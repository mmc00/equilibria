"""Tests for GTAPv62Contract — closure and equation configurations."""

from __future__ import annotations

import pytest

from equilibria.templates.gtap_v62 import (
    GTAPv62ClosureConfig,
    GTAPv62Contract,
    GTAPv62EquationConfig,
    build_gtap_v62_contract,
    default_gtap_v62_contract,
)


def test_default_contract_basic_fields() -> None:
    """Standard v6.2 closure has pgdpwld numeraire and CNS type."""
    c = default_gtap_v62_contract()
    assert c.closure.name == "gtap_v62_standard"
    assert c.closure.numeraire == "pgdpwld"
    assert c.closure.closure_type == "CNS"
    assert c.closure.if_sub is False
    assert c.closure.rordelta is True


def test_default_contract_fixed_endogenous_disjoint() -> None:
    """No variable appears in both fixed and endogenous lists."""
    c = default_gtap_v62_contract()
    overlap = set(c.closure.fixed) & set(c.closure.endogenous)
    assert not overlap, f"Overlap between fixed and endogenous: {overlap}"


def test_default_contract_includes_v62_equations() -> None:
    """The contract registers v6.2-specific equations and excludes v7-only ones."""
    c = default_gtap_v62_contract()
    eqs = set(c.equations.include)

    # v6.2 core equations must be present
    must_have = {"e_qo", "e_ps", "e_qfe", "e_pfe", "e_qpd", "e_qpm",
                 "e_qgd", "e_pg", "e_qxs", "e_pms", "e_walras", "e_pgdpwld"}
    missing = must_have - eqs
    assert not missing, f"Missing v6.2 equations: {missing}"

    # v7-only equations must NOT be present
    forbidden = {"e_qint", "e_pint", "e_qca", "e_pca",
                 "e_pefactreal", "e_pebfactreal"}
    leaked = forbidden & eqs
    assert not leaked, f"v7-only equations leaked into v6.2 contract: {leaked}"


def test_trade_policy_closure_relaxes_tariffs() -> None:
    """The trade_policy closure makes import/export taxes endogenous."""
    c = build_gtap_v62_contract("trade_policy")
    assert c.closure.fix_taxes is False
    assert "tm" not in c.closure.fixed
    assert "tms" not in c.closure.fixed
    assert "tx" not in c.closure.fixed
    assert "txs" not in c.closure.fixed
    # Other taxes should remain fixed
    assert "to" in c.closure.fixed


def test_altertax_closure_relaxes_all_taxes() -> None:
    """The altertax closure relaxes all tax flags so the SAM rebalances."""
    c = build_gtap_v62_contract("altertax")
    assert c.closure.name == "altertax"
    assert c.closure.fix_taxes is False


def test_unknown_closure_raises() -> None:
    """Unknown closure names raise ValueError."""
    with pytest.raises(ValueError, match="Unknown v6.2 closure"):
        build_gtap_v62_contract("not_a_real_closure")


def test_closure_is_frozen() -> None:
    """Pydantic models are frozen — direct mutation must fail."""
    c = GTAPv62ClosureConfig()
    with pytest.raises(Exception):  # pydantic frozen_instance error
        c.fix_taxes = False  # type: ignore[misc]
