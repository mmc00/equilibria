"""Constraint and Jacobian harness for PEP IPOPT solves."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from equilibria.solver.transforms import pep_array_to_variables
from equilibria.templates.pep_model_equations import PEPModelEquations


class PEPConstraintJacobianHarness:
    """Encapsulate hard-constraint values, Jacobian values, and structure.

    This keeps IPOPT-facing constraint logic separate from the feasibility NLP
    wrapper so later epics can swap finite differences for analytic derivatives
    without reopening the whole `CGEProblem` class.
    """

    def __init__(
        self,
        *,
        equations: PEPModelEquations,
        sets: dict[str, list[str]],
        n_variables: int,
        hard_constraints: Sequence[str] | None = None,
        variable_names: Sequence[str] | None = None,
        sparsity_reference_x: np.ndarray | None = None,
        sparsity_tol: float = 1e-12,
    ) -> None:
        self.equations = equations
        self.sets = sets
        self.n_variables = int(n_variables)
        self.constraint_names = tuple(hard_constraints or ())
        self.variable_names = tuple(variable_names or ())
        self.variable_index = {name: idx for idx, name in enumerate(self.variable_names)}
        self.sparsity_reference_x = None if sparsity_reference_x is None else np.array(sparsity_reference_x, dtype=float)
        self.sparsity_tol = float(sparsity_tol)
        self._constraint_scale: np.ndarray | None = None
        self._rows: np.ndarray | None = None
        self._cols: np.ndarray | None = None

    def evaluate_constraints(self, x: np.ndarray) -> np.ndarray:
        """Return scaled hard-constraint residuals in declared order."""
        if not self.constraint_names:
            return np.array([], dtype=float)

        residual_dict = self.equations.calculate_all_residuals(pep_array_to_variables(x, self.sets))
        residuals = np.array(
            [residual_dict.get(eq, 0.0) for eq in self.constraint_names],
            dtype=float,
        )
        if self._constraint_scale is None:
            self._constraint_scale = np.maximum(np.abs(residuals), 1.0)
        return residuals / self._constraint_scale

    def evaluate_jacobian_values(self, x: np.ndarray) -> np.ndarray:
        """Return Jacobian values aligned with the cached sparse structure."""
        rows, cols = self.jacobian_structure()
        m = len(self.constraint_names)
        n = len(x)
        if m == 0 or len(rows) == 0:
            return np.array([], dtype=float)

        base = self.evaluate_constraints(x)
        vars = pep_array_to_variables(x, self.sets)
        jac = np.zeros((m, n), dtype=float)
        analytic_rows: set[int] = set()
        for row, name in enumerate(self.constraint_names):
            analytic = self._analytic_constraint_derivatives(name, vars)
            if analytic is None:
                continue
            analytic_rows.add(row)
            for col, value in analytic.items():
                jac[row, col] = value

        fd_eps = np.sqrt(np.finfo(float).eps)
        for i in range(n):
            step = max(1e-8, fd_eps * max(abs(float(x[i])), 1.0))
            x_plus = x.copy()
            x_plus[i] += step
            c_plus = self.evaluate_constraints(x_plus)
            delta = (c_plus - base) / step
            if analytic_rows:
                for row in range(m):
                    if row in analytic_rows:
                        continue
                    jac[row, i] = delta[row]
            else:
                jac[:, i] = delta
        return jac[rows, cols]

    def jacobian_structure(self) -> tuple[np.ndarray, np.ndarray]:
        """Return sparse row/column indices for the current constraint set."""
        if self._rows is not None and self._cols is not None:
            return self._rows, self._cols

        m = len(self.constraint_names)
        n = self.n_variables
        if m == 0:
            return np.array([], dtype=int), np.array([], dtype=int)

        if self.sparsity_reference_x is None:
            rows = np.repeat(np.arange(m, dtype=int), n)
            cols = np.tile(np.arange(n, dtype=int), m)
            self._rows, self._cols = rows, cols
            return rows, cols

        base = self.evaluate_constraints(self.sparsity_reference_x)
        vars = pep_array_to_variables(self.sparsity_reference_x, self.sets)
        fd_eps = np.sqrt(np.finfo(float).eps)
        pairs: set[tuple[int, int]] = set()
        analytic_rows: set[int] = set()
        for row, name in enumerate(self.constraint_names):
            analytic = self._analytic_constraint_derivatives(name, vars)
            if analytic is None:
                continue
            analytic_rows.add(row)
            for col in analytic:
                pairs.add((int(row), int(col)))

        for col in range(n):
            step = max(1e-8, fd_eps * max(abs(float(self.sparsity_reference_x[col])), 1.0))
            x_plus = self.sparsity_reference_x.copy()
            x_plus[col] += step
            c_plus = self.evaluate_constraints(x_plus)
            delta = (c_plus - base) / step
            hit_rows = np.flatnonzero(np.abs(delta) > self.sparsity_tol)
            for row in hit_rows.tolist():
                if row in analytic_rows:
                    continue
                pairs.add((int(row), int(col)))

        if not pairs:
            rows = np.repeat(np.arange(m, dtype=int), n)
            cols = np.tile(np.arange(n, dtype=int), m)
            self._rows, self._cols = rows, cols
            return rows, cols

        ordered_pairs = sorted(pairs)
        rows = np.array([row for row, _ in ordered_pairs], dtype=int)
        cols = np.array([col for _, col in ordered_pairs], dtype=int)
        self._rows, self._cols = rows, cols
        return rows, cols

    def _var_idx(self, name: str) -> int | None:
        return self.variable_index.get(name)

    def _analytic_constraint_derivatives(
        self,
        constraint_name: str,
        vars,
    ) -> dict[int, float] | None:
        if not self.variable_index:
            return None

        result: dict[int, float] = {}

        def add(name: str, value: float) -> None:
            idx = self._var_idx(name)
            if idx is None or abs(value) <= 0.0:
                return
            result[idx] = result.get(idx, 0.0) + float(value)

        if constraint_name.startswith("EQ66_"):
            j = constraint_name.split("_", 1)[1]
            ttip = self.equations.params.get("ttip", {}).get(j, 0.0)
            add(f"PT[{j}]", 1.0)
            add(f"PP[{j}]", -(1.0 + ttip))
            return result

        if constraint_name.startswith("EQ70_"):
            _, l, j = constraint_name.split("_", 2)
            ttiw = self.equations.params.get("ttiw", {}).get((l, j), 0.0)
            add(f"WTI[{l},{j}]", 1.0)
            add(f"W[{l}]", -(1.0 + ttiw))
            return result

        if constraint_name.startswith("EQ72_"):
            _, k, j = constraint_name.split("_", 2)
            ttik = self.equations.params.get("ttik", {}).get((k, j), 0.0)
            add(f"RTI[{k},{j}]", 1.0)
            add(f"R[{k},{j}]", -(1.0 + ttik))
            return result

        if constraint_name.startswith("EQ73_"):
            _, k, j = constraint_name.split("_", 2)
            add(f"R[{k},{j}]", 1.0)
            add(f"RK[{k}]", -1.0)
            return result

        if constraint_name.startswith("EQ74_"):
            _, j, i = constraint_name.split("_", 2)
            add(f"P[{j},{i}]", 1.0)
            add(f"PT[{j}]", -1.0)
            return result

        if constraint_name == "EQ81":
            co0 = self.equations.params.get("CO0", {})
            pco0 = self.equations.params.get("PCO0", {})
            den = 0.0
            weights: dict[str, float] = {}
            for i in self.sets.get("I", []):
                weight_i = sum(co0.get((i, h), 0.0) for h in self.sets.get("H", []))
                weights[i] = weight_i
                den += pco0.get(i, 1.0) * weight_i
            if den <= 1e-12:
                return result
            add("PIXCON", 1.0)
            for i, weight_i in weights.items():
                if abs(weight_i) <= 1e-12:
                    continue
                add(f"PC[{i}]", -(weight_i / den))
            return result

        if constraint_name == "EQ82":
            gamma_inv = self.equations.params.get("gamma_INV", {})
            pco0 = self.equations.params.get("PCO0", {})
            log_pixinv = 0.0
            active: list[tuple[str, float, float]] = []
            for i in self.sets.get("I", []):
                g = gamma_inv.get(i, 0.0)
                if g <= 0:
                    continue
                pc_i = vars.PC.get(i, 1.0)
                pco_i = pco0.get(i, 1.0)
                if pc_i > 0 and pco_i > 0:
                    active.append((i, g, pc_i))
                    log_pixinv += g * np.log(pc_i / pco_i)
            expected = float(np.exp(log_pixinv))
            add("PIXINV", 1.0)
            for i, g, pc_i in active:
                add(f"PC[{i}]", -(expected * g / pc_i))
            return result

        if constraint_name == "EQ83":
            gamma_gvt = self.equations.params.get("gamma_GVT", {})
            pco0 = self.equations.params.get("PCO0", {})
            log_pixgvt = 0.0
            active: list[tuple[str, float, float]] = []
            for i in self.sets.get("I", []):
                g = gamma_gvt.get(i, 0.0)
                if g <= 0:
                    continue
                pc_i = vars.PC.get(i, 1.0)
                pco_i = pco0.get(i, 1.0)
                if pc_i > 0 and pco_i > 0:
                    active.append((i, g, pc_i))
                    log_pixgvt += g * np.log(pc_i / pco_i)
            expected = float(np.exp(log_pixgvt))
            add("PIXGVT", 1.0)
            for i, g, pc_i in active:
                add(f"PC[{i}]", -(expected * g / pc_i))
            return result

        if constraint_name == "EQ90":
            add("GDP_BP", 1.0)
            add("TIPT", -1.0)
            for j in self.sets.get("J", []):
                add(f"PVA[{j}]", -vars.VA.get(j, 0.0))
                add(f"VA[{j}]", -vars.PVA.get(j, 0.0))
            return result

        if constraint_name == "EQ91":
            add("GDP_MP", 1.0)
            add("GDP_BP", -1.0)
            add("TPRCTS", -1.0)
            return result

        if constraint_name.startswith("EQ94_"):
            h = constraint_name.split("_", 1)[1]
            pixcon = vars.PIXCON
            if abs(pixcon) <= 1e-12:
                return result
            cth = vars.CTH.get(h, 0.0)
            add(f"CTH_REAL[{h}]", 1.0)
            add(f"CTH[{h}]", -(1.0 / pixcon))
            add("PIXCON", cth / (pixcon ** 2))
            return result

        if constraint_name == "EQ95":
            pixgvt = vars.PIXGVT
            add("G_REAL", 1.0)
            if abs(pixgvt) > 1e-12:
                add("G", -(1.0 / pixgvt))
                add("PIXGVT", vars.G / (pixgvt ** 2))
            return result

        if constraint_name == "EQ96":
            pixgdp = vars.PIXGDP
            add("GDP_BP_REAL", 1.0)
            if abs(pixgdp) > 1e-12:
                add("GDP_BP", -(1.0 / pixgdp))
                add("PIXGDP", vars.GDP_BP / (pixgdp ** 2))
            return result

        if constraint_name == "EQ97":
            pixcon = vars.PIXCON
            add("GDP_MP_REAL", 1.0)
            if abs(pixcon) > 1e-12:
                add("GDP_MP", -(1.0 / pixcon))
                add("PIXCON", vars.GDP_MP / (pixcon ** 2))
            return result

        if constraint_name == "EQ98":
            add("GFCF_REAL", 1.0)
            if abs(vars.PIXINV) > 1e-12:
                add("GFCF", -(1.0 / vars.PIXINV))
                add("PIXINV", vars.GFCF / (vars.PIXINV ** 2))
            return result

        return None
