"""Canonical parity specification for the simple-open template."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from equilibria.templates.simple_open_contract import SimpleOpenContract, build_simple_open_contract


def _canonical_variable_names() -> tuple[str, ...]:
    return ("VA", "INT", "X", "D", "E", "ER", "PFX", "CAB", "FSAV")


class SimpleOpenBenchmarkParameters(BaseModel):
    """Benchmark parameters and exogenous levels for one simple-open closure."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    alpha_va: float
    rho_va: float
    a_int: float
    b_ext: float
    theta_cet: float
    phi_cet: float
    ER: float
    PFX: float
    D: float
    E: float
    CAB: float
    FSAV: float


class SimpleOpenParitySpec(BaseModel):
    """Canonical parity specification for `simple_open_v1`."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    contract_name: str = "simple_open_v1"
    closure_name: str
    equation_names: tuple[str, ...]
    variable_names: tuple[str, ...] = Field(default_factory=_canonical_variable_names)
    benchmark_parameters: SimpleOpenBenchmarkParameters
    benchmark_levels: dict[str, float]


def _va_target(*, alpha: float, rho: float, er: float, pfx: float) -> float:
    inner = (alpha * (er**rho)) + ((1.0 - alpha) * (pfx**rho))
    return float(inner ** (1.0 / rho))


def _cet_target(*, theta: float, phi: float, d: float, e: float, er: float, pfx: float) -> float:
    trade_term = e * er / pfx
    inner = (theta * (d**phi)) + ((1.0 - theta) * (trade_term**phi))
    return float(inner ** (1.0 / phi))


def default_simple_open_benchmark_parameters(
    closure_name: str,
) -> SimpleOpenBenchmarkParameters:
    """Return the canonical benchmark parameter block for one closure."""

    name = str(closure_name).strip().lower()
    if name == "flexible_external_balance":
        return SimpleOpenBenchmarkParameters(
            alpha_va=0.45,
            rho_va=0.70,
            a_int=0.55,
            b_ext=0.08,
            theta_cet=0.58,
            phi_cet=1.25,
            ER=1.08,
            PFX=1.00,
            D=1.04,
            E=0.93,
            CAB=0.82,
            FSAV=0.82,
        )
    if name == "simple_open_default":
        return SimpleOpenBenchmarkParameters(
            alpha_va=0.40,
            rho_va=0.75,
            a_int=0.50,
            b_ext=0.10,
            theta_cet=0.60,
            phi_cet=1.20,
            ER=1.00,
            PFX=1.00,
            D=1.00,
            E=1.00,
            CAB=1.00,
            FSAV=1.00,
        )
    raise ValueError(f"Unsupported simple-open parity closure: {closure_name!r}")


def _benchmark_levels_from_parameters(
    params: SimpleOpenBenchmarkParameters,
) -> dict[str, float]:
    va = _va_target(alpha=params.alpha_va, rho=params.rho_va, er=params.ER, pfx=params.PFX)
    x = _cet_target(
        theta=params.theta_cet,
        phi=params.phi_cet,
        d=params.D,
        e=params.E,
        er=params.ER,
        pfx=params.PFX,
    )
    int_val = (params.a_int * x) + (params.b_ext * (params.CAB - params.FSAV))
    return {
        "VA": va,
        "INT": float(int_val),
        "X": x,
        "D": float(params.D),
        "E": float(params.E),
        "ER": float(params.ER),
        "PFX": float(params.PFX),
        "CAB": float(params.CAB),
        "FSAV": float(params.FSAV),
    }


def build_simple_open_parity_spec(
    contract: str | Mapping[str, Any] | SimpleOpenContract | None = None,
) -> SimpleOpenParitySpec:
    """Build the canonical parity spec for the resolved simple-open contract."""

    resolved_contract = build_simple_open_contract(contract)
    params = default_simple_open_benchmark_parameters(resolved_contract.closure.name)
    return SimpleOpenParitySpec(
        contract_name=resolved_contract.name,
        closure_name=resolved_contract.closure.name,
        equation_names=resolved_contract.equations.include,
        benchmark_parameters=params,
        benchmark_levels=_benchmark_levels_from_parameters(params),
    )
