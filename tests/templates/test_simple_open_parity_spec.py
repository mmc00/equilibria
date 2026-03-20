from __future__ import annotations

import pytest

from equilibria.templates import build_simple_open_contract
from equilibria.templates.simple_open_parity_spec import (
    build_simple_open_parity_spec,
    default_simple_open_benchmark_parameters,
)


def test_simple_open_parity_spec_default_is_canonical_contract() -> None:
    spec = build_simple_open_parity_spec("simple_open_v1")

    assert spec.contract_name == "simple_open_v1"
    assert spec.closure_name == "simple_open_default"
    assert spec.equation_names == ("EQ_VA", "EQ_INT", "EQ_CET")
    assert spec.variable_names == ("VA", "INT", "X", "D", "E", "ER", "PFX", "CAB", "FSAV")
    assert spec.benchmark_levels["PFX"] == pytest.approx(1.0)
    assert spec.benchmark_levels["ER"] == pytest.approx(1.0)
    assert spec.benchmark_levels["CAB"] == pytest.approx(1.0)
    assert spec.benchmark_levels["FSAV"] == pytest.approx(1.0)


def test_simple_open_parity_spec_follows_flexible_external_balance_closure() -> None:
    contract = build_simple_open_contract({"closure": {"name": "flexible_external_balance"}})
    spec = build_simple_open_parity_spec(contract)

    assert spec.closure_name == "flexible_external_balance"
    assert spec.benchmark_parameters.CAB == pytest.approx(0.82)
    assert spec.benchmark_parameters.FSAV == pytest.approx(0.82)
    assert spec.benchmark_levels["ER"] == pytest.approx(1.08)
    assert spec.benchmark_levels["D"] == pytest.approx(1.04)
    assert spec.benchmark_levels["E"] == pytest.approx(0.93)


def test_simple_open_benchmark_parameters_reject_unknown_closure() -> None:
    with pytest.raises(ValueError, match="Unsupported simple-open parity closure"):
        default_simple_open_benchmark_parameters("bad_closure")
