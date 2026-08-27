"""GTAP6Contract builds a valid standard closure."""

from __future__ import annotations

from equilibria.templates.gtap6.gtap6_contract import (
    build_gtap6_contract,
    default_gtap6_contract,
)


def test_default_contract_is_standard_closure():
    contract = default_gtap6_contract()
    assert contract.closure.name == "gtap6_standard"
    assert contract.closure.numeraire == "pgdpwld"
    assert contract.closure.if_sub is False


def test_full_equation_ids_include_production_and_closure():
    contract = build_gtap6_contract("gtap6_standard")
    ids = set(contract.equations.include)
    assert "e_qo" in ids
    assert "e_walras" in ids
    assert "e_pgdpwld" in ids


def test_trade_policy_closure_frees_tariffs():
    contract = build_gtap6_contract("trade_policy")
    assert "tm" not in contract.closure.fixed
    assert "tms" not in contract.closure.fixed
