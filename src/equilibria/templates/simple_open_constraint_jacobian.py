"""Simple-open benchmark Jacobian harness built on the generic base."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from equilibria.solver import ConstraintJacobianHarness
from equilibria.templates.simple_open_contract import SimpleOpenContract
from equilibria.templates.simple_open_parity_spec import (
    SimpleOpenParitySpec,
    build_simple_open_parity_spec,
)


@dataclass(frozen=True)
class SimpleOpenJacobianContext:
    """Evaluation context for the simple-open benchmark system."""

    VA: float
    INT: float
    X: float
    D: float
    E: float
    ER: float
    PFX: float
    CAB: float
    FSAV: float
    alpha_va: float
    rho_va: float
    a_int: float
    b_ext: float
    theta_cet: float
    phi_cet: float


class SimpleOpenConstraintJacobianHarness(ConstraintJacobianHarness):
    """Analytic/numeric Jacobian harness for a small simple-open benchmark."""

    variable_names_default = ("VA", "INT", "X", "D", "E", "ER", "PFX", "CAB", "FSAV")

    def __init__(
        self,
        *,
        contract: SimpleOpenContract,
        jacobian_mode: str = "analytic",
    ) -> None:
        self.contract = contract
        self.spec: SimpleOpenParitySpec = build_simple_open_parity_spec(contract)
        self._params = self.spec.benchmark_parameters.model_dump(mode="python")
        self._benchmark_x = np.array(
            [self.spec.benchmark_levels[name] for name in self.spec.variable_names],
            dtype=float,
        )
        super().__init__(
            n_variables=len(self.spec.variable_names),
            constraint_names=self.spec.equation_names,
            variable_names=self.spec.variable_names,
            sparsity_reference_x=self._benchmark_x,
            jacobian_mode=jacobian_mode,
        )

    @property
    def benchmark_point(self) -> np.ndarray:
        """Return the benchmark point used for the harness reference."""

        return np.array(self._benchmark_x, dtype=float)

    @staticmethod
    def _va_target(*, alpha: float, rho: float, er: float, pfx: float) -> float:
        inner = (alpha * (er**rho)) + ((1.0 - alpha) * (pfx**rho))
        return float(inner ** (1.0 / rho))

    @staticmethod
    def _cet_target(*, theta: float, phi: float, d: float, e: float, er: float, pfx: float) -> float:
        trade_term = e * er / pfx
        inner = (theta * (d**phi)) + ((1.0 - theta) * (trade_term**phi))
        return float(inner ** (1.0 / phi))

    def _build_context(self, x: np.ndarray) -> SimpleOpenJacobianContext:
        values = np.array(x, dtype=float)
        return SimpleOpenJacobianContext(
            VA=float(values[0]),
            INT=float(values[1]),
            X=float(values[2]),
            D=float(values[3]),
            E=float(values[4]),
            ER=float(values[5]),
            PFX=float(values[6]),
            CAB=float(values[7]),
            FSAV=float(values[8]),
            alpha_va=float(self._params["alpha_va"]),
            rho_va=float(self._params["rho_va"]),
            a_int=float(self._params["a_int"]),
            b_ext=float(self._params["b_ext"]),
            theta_cet=float(self._params["theta_cet"]),
            phi_cet=float(self._params["phi_cet"]),
        )

    def _calculate_constraint_residual_dict(
        self,
        context: SimpleOpenJacobianContext,
    ) -> dict[str, float]:
        va_target = self._va_target(
            alpha=context.alpha_va,
            rho=context.rho_va,
            er=context.ER,
            pfx=context.PFX,
        )
        cet_target = self._cet_target(
            theta=context.theta_cet,
            phi=context.phi_cet,
            d=context.D,
            e=context.E,
            er=context.ER,
            pfx=context.PFX,
        )
        return {
            "EQ_VA": float(context.VA - va_target),
            "EQ_INT": float(context.INT - ((context.a_int * context.X) + (context.b_ext * (context.CAB - context.FSAV)))),
            "EQ_CET": float(context.X - cet_target),
        }

    def _analytic_constraint_derivatives(
        self,
        constraint_name: str,
        context: SimpleOpenJacobianContext,
    ) -> dict[int, float] | None:
        if constraint_name == "EQ_VA":
            inner = (context.alpha_va * (context.ER**context.rho_va)) + (
                (1.0 - context.alpha_va) * (context.PFX**context.rho_va)
            )
            common = inner ** ((1.0 / context.rho_va) - 1.0)
            return {
                self._var_idx("VA"): 1.0,
                self._var_idx("ER"): -context.alpha_va * common * (context.ER ** (context.rho_va - 1.0)),
                self._var_idx("PFX"): -(1.0 - context.alpha_va)
                * common
                * (context.PFX ** (context.rho_va - 1.0)),
            }

        if constraint_name == "EQ_INT":
            return {
                self._var_idx("INT"): 1.0,
                self._var_idx("X"): -context.a_int,
                self._var_idx("CAB"): -context.b_ext,
                self._var_idx("FSAV"): context.b_ext,
            }

        if constraint_name == "EQ_CET":
            trade_term = context.E * context.ER / context.PFX
            inner = (context.theta_cet * (context.D**context.phi_cet)) + (
                (1.0 - context.theta_cet) * (trade_term**context.phi_cet)
            )
            common = inner ** ((1.0 / context.phi_cet) - 1.0)
            d_h_d_d = context.theta_cet * common * (context.D ** (context.phi_cet - 1.0))
            d_h_d_trade = (1.0 - context.theta_cet) * common * (trade_term ** (context.phi_cet - 1.0))
            return {
                self._var_idx("X"): 1.0,
                self._var_idx("D"): -d_h_d_d,
                self._var_idx("E"): -(d_h_d_trade * (context.ER / context.PFX)),
                self._var_idx("ER"): -(d_h_d_trade * (context.E / context.PFX)),
                self._var_idx("PFX"): -(
                    d_h_d_trade * (-(context.E * context.ER) / (context.PFX**2))
                ),
            }

        return None
