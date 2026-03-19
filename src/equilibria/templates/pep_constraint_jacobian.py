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
            scale = float(self._constraint_scale[row]) if self._constraint_scale is not None else 1.0
            for col, value in analytic.items():
                jac[row, col] = value / scale

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

        if constraint_name.startswith("EQ1_"):
            j = constraint_name.split("_", 1)[1]
            v_j = self.equations.params.get("v", {}).get(j, 0.0)
            add(f"VA[{j}]", 1.0)
            add(f"XST[{j}]", -v_j)
            return result

        if constraint_name.startswith("EQ2_"):
            j = constraint_name.split("_", 1)[1]
            io_j = self.equations.params.get("io", {}).get(j, 0.0)
            add(f"CI[{j}]", 1.0)
            add(f"XST[{j}]", -io_j)
            return result

        if constraint_name.startswith("EQ3_"):
            j = constraint_name.split("_", 1)[1]
            rho_va = self.equations.params.get("rho_VA", {}).get(j, -1.0)
            beta_va = self.equations.params.get("beta_VA", {}).get(j, 0.5)
            b_va = self.equations.params.get("B_VA", {}).get(j, 1.0)
            ldc_j = vars.LDC.get(j, 0.0)
            kdc_j = vars.KDC.get(j, 0.0)
            if rho_va == 0 or b_va <= 0 or ldc_j <= 0 or kdc_j <= 0:
                return None
            term = beta_va * (ldc_j ** (-rho_va)) + (1.0 - beta_va) * (kdc_j ** (-rho_va))
            if term <= 0:
                return None
            coeff = b_va * (term ** ((-1.0 / rho_va) - 1.0))
            add(f"VA[{j}]", 1.0)
            add(f"LDC[{j}]", -(coeff * beta_va * (ldc_j ** (-rho_va - 1.0))))
            add(f"KDC[{j}]", -(coeff * (1.0 - beta_va) * (kdc_j ** (-rho_va - 1.0))))
            return result

        if constraint_name.startswith("EQ4_"):
            j = constraint_name.split("_", 1)[1]
            sigma_va = self.equations.params.get("sigma_VA", {}).get(j, 1.0)
            beta_va = self.equations.params.get("beta_VA", {}).get(j, 0.5)
            rc_j = vars.RC.get(j, 0.0)
            wc_j = vars.WC.get(j, 0.0)
            kdc_j = vars.KDC.get(j, 0.0)
            if not (0.0 < beta_va < 1.0):
                return result
            if rc_j <= 0 or wc_j <= 0:
                return None
            ratio = ((beta_va / (1.0 - beta_va)) * (rc_j / wc_j)) ** sigma_va
            expected_ldc = ratio * kdc_j
            add(f"LDC[{j}]", 1.0)
            add(f"KDC[{j}]", -ratio)
            add(f"RC[{j}]", -(expected_ldc * sigma_va / rc_j))
            add(f"WC[{j}]", expected_ldc * sigma_va / wc_j)
            return result

        if constraint_name.startswith("EQ5_"):
            j = constraint_name.split("_", 1)[1]
            rho_ld = self.equations.params.get("rho_LD", {}).get(j, 0.0)
            b_ld = self.equations.params.get("B_LD", {}).get(j, 1.0)
            if rho_ld == 0 or b_ld <= 0:
                return None
            term = 0.0
            active: list[tuple[str, float, float]] = []
            for l in self.sets.get("L", []):
                beta_ld = self.equations.params.get("beta_LD", {}).get((l, j), 0.0)
                ld_lj = vars.LD.get((l, j), 0.0)
                if ld_lj <= 0 or beta_ld <= 0:
                    continue
                active.append((l, beta_ld, ld_lj))
                term += beta_ld * (ld_lj ** (-rho_ld))
            if term <= 0:
                return None
            coeff = b_ld * (term ** ((-1.0 / rho_ld) - 1.0))
            add(f"LDC[{j}]", 1.0)
            for l, beta_ld, ld_lj in active:
                add(f"LD[{l},{j}]", -(coeff * beta_ld * (ld_lj ** (-rho_ld - 1.0))))
            return result

        if constraint_name.startswith("EQ6_"):
            _, l, j = constraint_name.split("_", 2)
            sigma_ld = self.equations.params.get("sigma_LD", {}).get(j, 1.0)
            beta_ld = self.equations.params.get("beta_LD", {}).get((l, j), 0.0)
            b_ld = self.equations.params.get("B_LD", {}).get(j, 1.0)
            wti_lj = vars.WTI.get((l, j), 0.0)
            wc_j = vars.WC.get(j, 0.0)
            ldc_j = vars.LDC.get(j, 0.0)
            if b_ld <= 0 or wti_lj <= 0 or wc_j <= 0 or beta_ld <= 0:
                return None
            alloc = (beta_ld * wc_j / wti_lj) ** sigma_ld * (b_ld ** (sigma_ld - 1.0))
            expected_ld = alloc * ldc_j
            add(f"LD[{l},{j}]", 1.0)
            add(f"LDC[{j}]", -alloc)
            add(f"WC[{j}]", -(expected_ld * sigma_ld / wc_j))
            add(f"WTI[{l},{j}]", expected_ld * sigma_ld / wti_lj)
            return result

        if constraint_name.startswith("EQ7_"):
            j = constraint_name.split("_", 1)[1]
            rho_kd = self.equations.params.get("rho_KD", {}).get(j, 0.0)
            b_kd = self.equations.params.get("B_KD", {}).get(j, 1.0)
            if rho_kd == 0 or b_kd <= 0:
                return None
            term = 0.0
            active: list[tuple[str, float, float]] = []
            for k in self.sets.get("K", []):
                if self.equations.params.get("KDO0", {}).get((k, j), 0.0) == 0:
                    continue
                beta_kd = self.equations.params.get("beta_KD", {}).get((k, j), 0.0)
                kd_kj = vars.KD.get((k, j), 0.0)
                if kd_kj <= 0 or beta_kd <= 0:
                    continue
                active.append((k, beta_kd, kd_kj))
                term += beta_kd * (kd_kj ** (-rho_kd))
            if term <= 0:
                return None
            coeff = b_kd * (term ** ((-1.0 / rho_kd) - 1.0))
            add(f"KDC[{j}]", 1.0)
            for k, beta_kd, kd_kj in active:
                add(f"KD[{k},{j}]", -(coeff * beta_kd * (kd_kj ** (-rho_kd - 1.0))))
            return result

        if constraint_name.startswith("EQ8_"):
            _, k, j = constraint_name.split("_", 2)
            sigma_kd = self.equations.params.get("sigma_KD", {}).get(j, 1.0)
            beta_kd = self.equations.params.get("beta_KD", {}).get((k, j), 0.0)
            b_kd = self.equations.params.get("B_KD", {}).get(j, 1.0)
            rti_kj = vars.RTI.get((k, j), 0.0)
            rc_j = vars.RC.get(j, 0.0)
            kdc_j = vars.KDC.get(j, 0.0)
            if b_kd <= 0 or rti_kj <= 0 or rc_j <= 0 or beta_kd <= 0:
                return None
            alloc = (beta_kd * rc_j / rti_kj) ** sigma_kd * (b_kd ** (sigma_kd - 1.0))
            expected_kd = alloc * kdc_j
            add(f"KD[{k},{j}]", 1.0)
            add(f"KDC[{j}]", -alloc)
            add(f"RC[{j}]", -(expected_kd * sigma_kd / rc_j))
            add(f"RTI[{k},{j}]", expected_kd * sigma_kd / rti_kj)
            return result

        if constraint_name.startswith("EQ9_"):
            _, i, j = constraint_name.split("_", 2)
            aij = self.equations.params.get("aij", {}).get((i, j), 0.0)
            add(f"DI[{i},{j}]", 1.0)
            add(f"CI[{j}]", -aij)
            return result

        if constraint_name.startswith("EQ10_"):
            h = constraint_name.split("_", 1)[1]
            add(f"YH[{h}]", 1.0)
            add(f"YHL[{h}]", -1.0)
            add(f"YHK[{h}]", -1.0)
            add(f"YHTR[{h}]", -1.0)
            return result

        if constraint_name.startswith("EQ11_"):
            h = constraint_name.split("_", 1)[1]
            add(f"YHL[{h}]", 1.0)
            for l in self.sets.get("L", []):
                lam = self.equations.params.get("lambda_WL", {}).get((h, l), 0.0)
                if lam == 0:
                    continue
                ld_sum = 0.0
                for j in self.sets.get("J", []):
                    if abs(self.equations.params.get("LDO0", {}).get((l, j), 0.0)) <= 1e-12:
                        continue
                    ld_sum += vars.LD.get((l, j), 0.0)
                    add(f"LD[{l},{j}]", -(lam * vars.W.get(l, 0.0)))
                add(f"W[{l}]", -(lam * ld_sum))
            return result

        if constraint_name.startswith("EQ12_"):
            h = constraint_name.split("_", 1)[1]
            add(f"YHK[{h}]", 1.0)
            for k in self.sets.get("K", []):
                lam = self.equations.params.get("lambda_RK", {}).get((h, k), 0.0)
                if lam == 0:
                    continue
                for j in self.sets.get("J", []):
                    if abs(self.equations.params.get("KDO0", {}).get((k, j), 0.0)) <= 1e-12:
                        continue
                    add(f"R[{k},{j}]", -(lam * vars.KD.get((k, j), 0.0)))
                    add(f"KD[{k},{j}]", -(lam * vars.R.get((k, j), 0.0)))
            return result

        if constraint_name.startswith("EQ13_"):
            h = constraint_name.split("_", 1)[1]
            add(f"YHTR[{h}]", 1.0)
            for ag in self.sets.get("AG", []):
                add(f"TR[{h},{ag}]", -1.0)
            return result

        if constraint_name.startswith("EQ14_"):
            h = constraint_name.split("_", 1)[1]
            add(f"YDH[{h}]", 1.0)
            add(f"YH[{h}]", -1.0)
            add(f"TDH[{h}]", 1.0)
            add(f"TR[gvt,{h}]", 1.0)
            return result

        if constraint_name.startswith("EQ15_"):
            h = constraint_name.split("_", 1)[1]
            add(f"CTH[{h}]", 1.0)
            add(f"YDH[{h}]", -1.0)
            add(f"SH[{h}]", 1.0)
            for agng in self.sets.get("AGNG", []):
                add(f"TR[{agng},{h}]", 1.0)
            return result

        if constraint_name.startswith("EQ16_"):
            h = constraint_name.split("_", 1)[1]
            eta = self.equations.params.get("eta", 0.0)
            sh0 = self.equations.params.get("sh0", {}).get(h, 0.0)
            sh1 = self.equations.params.get("sh1", {}).get(h, 0.0)
            add(f"SH[{h}]", 1.0)
            add(f"YDH[{h}]", -sh1)
            if eta != 0 and vars.PIXCON > 0 and sh0 != 0:
                add("PIXCON", -(eta * sh0 * (vars.PIXCON ** (eta - 1.0))))
            return result

        if constraint_name.startswith("EQ17_"):
            f = constraint_name.split("_", 1)[1]
            add(f"YF[{f}]", 1.0)
            add(f"YFK[{f}]", -1.0)
            add(f"YFTR[{f}]", -1.0)
            return result

        if constraint_name.startswith("EQ18_"):
            f = constraint_name.split("_", 1)[1]
            add(f"YFK[{f}]", 1.0)
            for k in self.sets.get("K", []):
                lam = self.equations.params.get("lambda_RK", {}).get((f, k), 0.0)
                if lam == 0:
                    continue
                for j in self.sets.get("J", []):
                    if abs(self.equations.params.get("KDO0", {}).get((k, j), 0.0)) <= 1e-12:
                        continue
                    add(f"R[{k},{j}]", -(lam * vars.KD.get((k, j), 0.0)))
                    add(f"KD[{k},{j}]", -(lam * vars.R.get((k, j), 0.0)))
            return result

        if constraint_name.startswith("EQ19_"):
            f = constraint_name.split("_", 1)[1]
            add(f"YFTR[{f}]", 1.0)
            for ag in self.sets.get("AG", []):
                add(f"TR[{f},{ag}]", -1.0)
            return result

        if constraint_name.startswith("EQ20_"):
            f = constraint_name.split("_", 1)[1]
            add(f"YDF[{f}]", 1.0)
            add(f"YF[{f}]", -1.0)
            add(f"TDF[{f}]", 1.0)
            return result

        if constraint_name.startswith("EQ21_"):
            f = constraint_name.split("_", 1)[1]
            add(f"SF[{f}]", 1.0)
            add(f"YDF[{f}]", -1.0)
            for ag in self.sets.get("AG", []):
                add(f"TR[{ag},{f}]", 1.0)
            return result

        if constraint_name == "EQ22":
            add("YG", 1.0)
            add("YGK", -1.0)
            add("TDHT", -1.0)
            add("TDFT", -1.0)
            add("TPRODN", -1.0)
            add("TPRCTS", -1.0)
            add("YGTR", -1.0)
            return result

        if constraint_name == "EQ23":
            add("YGK", 1.0)
            for k in self.sets.get("K", []):
                lam = self.equations.params.get("lambda_RK", {}).get(("gvt", k), 0.0)
                if lam == 0:
                    continue
                for j in self.sets.get("J", []):
                    if abs(self.equations.params.get("KDO0", {}).get((k, j), 0.0)) <= 1e-12:
                        continue
                    add(f"R[{k},{j}]", -(lam * vars.KD.get((k, j), 0.0)))
                    add(f"KD[{k},{j}]", -(lam * vars.R.get((k, j), 0.0)))
            return result

        if constraint_name == "EQ24":
            add("TDHT", 1.0)
            for h in self.sets.get("H", []):
                add(f"TDH[{h}]", -1.0)
            return result

        if constraint_name == "EQ25":
            add("TDFT", 1.0)
            for f in self.sets.get("F", []):
                add(f"TDF[{f}]", -1.0)
            return result

        if constraint_name == "EQ26":
            add("TPRODN", 1.0)
            add("TIWT", -1.0)
            add("TIKT", -1.0)
            add("TIPT", -1.0)
            return result

        if constraint_name == "EQ27":
            add("TIWT", 1.0)
            for l in self.sets.get("L", []):
                for j in self.sets.get("J", []):
                    if abs(self.equations.params.get("LDO0", {}).get((l, j), 0.0)) > 1e-12:
                        add(f"TIW[{l},{j}]", -1.0)
            return result

        if constraint_name == "EQ28":
            add("TIKT", 1.0)
            for k in self.sets.get("K", []):
                for j in self.sets.get("J", []):
                    if abs(self.equations.params.get("KDO0", {}).get((k, j), 0.0)) > 1e-12:
                        add(f"TIK[{k},{j}]", -1.0)
            return result

        if constraint_name == "EQ29":
            add("TIPT", 1.0)
            for j in self.sets.get("J", []):
                add(f"TIP[{j}]", -1.0)
            return result

        if constraint_name == "EQ30":
            add("TPRCTS", 1.0)
            add("TICT", -1.0)
            add("TIMT", -1.0)
            add("TIXT", -1.0)
            return result

        if constraint_name == "EQ31":
            add("TICT", 1.0)
            for i in self.sets.get("I", []):
                add(f"TIC[{i}]", -1.0)
            return result

        if constraint_name == "EQ32":
            add("TIMT", 1.0)
            for i in self.sets.get("I", []):
                if abs(self.equations.params.get("IMO0", {}).get(i, 0.0)) > 1e-12:
                    add(f"TIM[{i}]", -1.0)
            return result

        if constraint_name == "EQ33":
            add("TIXT", 1.0)
            for i in self.sets.get("I", []):
                if abs(self.equations.params.get("EXDO0", {}).get(i, 0.0)) > 1e-12:
                    add(f"TIX[{i}]", -1.0)
            return result

        if constraint_name == "EQ34":
            add("YGTR", 1.0)
            for agng in self.sets.get("AGNG", []):
                add(f"TR[gvt,{agng}]", -1.0)
            return result

        if constraint_name.startswith("EQ35_"):
            h = constraint_name.split("_", 1)[1]
            eta = self.equations.params.get("eta", 0.0)
            ttdh0 = self.equations.params.get("ttdh0", {}).get(h, 0.0)
            ttdh1 = self.equations.params.get("ttdh1", {}).get(h, 0.0)
            add(f"TDH[{h}]", 1.0)
            add(f"YH[{h}]", -ttdh1)
            if eta != 0 and vars.PIXCON > 0 and ttdh0 != 0:
                add("PIXCON", -(eta * ttdh0 * (vars.PIXCON ** (eta - 1.0))))
            return result

        if constraint_name.startswith("EQ36_"):
            f = constraint_name.split("_", 1)[1]
            eta = self.equations.params.get("eta", 0.0)
            ttdf0 = self.equations.params.get("ttdf0", {}).get(f, 0.0)
            ttdf1 = self.equations.params.get("ttdf1", {}).get(f, 0.0)
            add(f"TDF[{f}]", 1.0)
            add(f"YFK[{f}]", -ttdf1)
            if eta != 0 and vars.PIXCON > 0 and ttdf0 != 0:
                add("PIXCON", -(eta * ttdf0 * (vars.PIXCON ** (eta - 1.0))))
            return result

        if constraint_name.startswith("EQ37_"):
            _, l, j = constraint_name.split("_", 2)
            ttiw = self.equations.params.get("ttiw", {}).get((l, j), 0.0)
            add(f"TIW[{l},{j}]", 1.0)
            add(f"W[{l}]", -(ttiw * vars.LD.get((l, j), 0.0)))
            add(f"LD[{l},{j}]", -(ttiw * vars.W.get(l, 0.0)))
            return result

        if constraint_name.startswith("EQ38_"):
            _, k, j = constraint_name.split("_", 2)
            ttik = self.equations.params.get("ttik", {}).get((k, j), 0.0)
            add(f"TIK[{k},{j}]", 1.0)
            add(f"R[{k},{j}]", -(ttik * vars.KD.get((k, j), 0.0)))
            add(f"KD[{k},{j}]", -(ttik * vars.R.get((k, j), 0.0)))
            return result

        if constraint_name.startswith("EQ39_"):
            j = constraint_name.split("_", 1)[1]
            ttip = self.equations.params.get("ttip", {}).get(j, 0.0)
            add(f"TIP[{j}]", 1.0)
            add(f"PP[{j}]", -(ttip * vars.XST.get(j, 0.0)))
            add(f"XST[{j}]", -(ttip * vars.PP.get(j, 0.0)))
            return result

        if constraint_name.startswith("EQ40_"):
            i = constraint_name.split("_", 1)[1]
            ttic = self.equations.params.get("ttic", {}).get(i, 0.0)
            denom = 1.0 + ttic
            add(f"TIC[{i}]", 1.0)
            if abs(denom) <= 1e-12:
                return result
            coeff = ttic / denom
            if abs(self.equations.params.get("DDO0", {}).get(i, 0.0)) > 1e-12:
                add(f"PD[{i}]", -(coeff * vars.DD.get(i, 0.0)))
                add(f"DD[{i}]", -(coeff * vars.PD.get(i, 0.0)))
            if abs(self.equations.params.get("IMO0", {}).get(i, 0.0)) > 1e-12:
                add(f"PM[{i}]", -(coeff * vars.IM.get(i, 0.0)))
                add(f"IM[{i}]", -(coeff * vars.PM.get(i, 0.0)))
            return result

        if constraint_name.startswith("EQ41_"):
            i = constraint_name.split("_", 1)[1]
            ttim = self.equations.params.get("ttim", {}).get(i, 0.0)
            add(f"TIM[{i}]", 1.0)
            add("e", -(ttim * vars.PWM.get(i, 0.0) * vars.IM.get(i, 0.0)))
            add(f"PWM[{i}]", -(ttim * vars.e * vars.IM.get(i, 0.0)))
            add(f"IM[{i}]", -(ttim * vars.e * vars.PWM.get(i, 0.0)))
            return result

        if constraint_name.startswith("EQ42_"):
            i = constraint_name.split("_", 1)[1]
            ttix = self.equations.params.get("ttix", {}).get(i, 0.0)
            margin_sum = sum(
                vars.PC.get(ij, 1.0) * self.equations.params.get("tmrg_X", {}).get((ij, i), 0.0)
                for ij in self.sets.get("I", [])
            )
            add(f"TIX[{i}]", 1.0)
            add(f"PE[{i}]", -(ttix * vars.EXD.get(i, 0.0)))
            add(f"EXD[{i}]", -(ttix * (vars.PE.get(i, 0.0) + margin_sum)))
            for ij in self.sets.get("I", []):
                add(f"PC[{ij}]", -(ttix * self.equations.params.get("tmrg_X", {}).get((ij, i), 0.0) * vars.EXD.get(i, 0.0)))
            return result

        if constraint_name == "EQ43":
            add("SG", 1.0)
            add("YG", -1.0)
            add("G", 1.0)
            for agng in self.sets.get("AGNG", []):
                add(f"TR[{agng},gvt]", 1.0)
            return result

        if constraint_name == "EQ44":
            add("YROW", 1.0)
            import_term = 0.0
            for i in self.sets.get("I", []):
                if abs(self.equations.params.get("IMO0", {}).get(i, 0.0)) <= 1e-12:
                    continue
                import_term += vars.PWM.get(i, 0.0) * vars.IM.get(i, 0.0)
                add(f"PWM[{i}]", -(vars.e * vars.IM.get(i, 0.0)))
                add(f"IM[{i}]", -(vars.e * vars.PWM.get(i, 0.0)))
            add("e", -import_term)
            for k in self.sets.get("K", []):
                lam = self.equations.params.get("lambda_RK", {}).get(("row", k), 0.0)
                if lam == 0:
                    continue
                for j in self.sets.get("J", []):
                    if abs(self.equations.params.get("KDO0", {}).get((k, j), 0.0)) <= 1e-12:
                        continue
                    add(f"R[{k},{j}]", -(lam * vars.KD.get((k, j), 0.0)))
                    add(f"KD[{k},{j}]", -(lam * vars.R.get((k, j), 0.0)))
            for agd in self.sets.get("AGD", []):
                add(f"TR[row,{agd}]", -1.0)
            return result

        if constraint_name == "EQ45":
            add("SROW", 1.0)
            add("YROW", -1.0)
            for i in self.sets.get("I", []):
                if abs(self.equations.params.get("EXDO0", {}).get(i, 0.0)) <= 1e-12:
                    continue
                add(f"PE_FOB[{i}]", vars.EXD.get(i, 0.0))
                add(f"EXD[{i}]", vars.PE_FOB.get(i, 0.0))
            for agd in self.sets.get("AGD", []):
                add(f"TR[{agd},row]", 1.0)
            return result

        if constraint_name == "EQ46":
            add("SROW", 1.0)
            add("CAB", 1.0)
            return result

        if constraint_name.startswith("EQ47_"):
            _, agng, h = constraint_name.split("_", 2)
            lam = self.equations.params.get("lambda_TR_households", {}).get((agng, h), 0.0)
            add(f"TR[{agng},{h}]", 1.0)
            add(f"YDH[{h}]", -lam)
            return result

        if constraint_name.startswith("EQ48_"):
            h = constraint_name.split("_", 1)[1]
            eta = self.equations.params.get("eta", 0.0)
            tr0 = self.equations.params.get("tr0", {}).get(h, 0.0)
            tr1 = self.equations.params.get("tr1", {}).get(h, 0.0)
            add(f"TR[gvt,{h}]", 1.0)
            add(f"YH[{h}]", -tr1)
            if eta != 0 and vars.PIXCON > 0 and tr0 != 0:
                add("PIXCON", -(eta * tr0 * (vars.PIXCON ** (eta - 1.0))))
            return result

        if constraint_name.startswith("EQ49_"):
            _, ag, f = constraint_name.split("_", 2)
            lam = self.equations.params.get("lambda_TR_firms", {}).get((ag, f), 0.0)
            add(f"TR[{ag},{f}]", 1.0)
            add(f"YDF[{f}]", -lam)
            return result

        if constraint_name.startswith("EQ50_"):
            agng = constraint_name.split("_", 1)[1]
            eta = self.equations.params.get("eta", 0.0)
            tro = self.equations.params.get("TRO", {}).get((agng, "gvt"), 0.0)
            add(f"TR[{agng},gvt]", 1.0)
            if eta != 0 and vars.PIXCON > 0 and tro != 0:
                add("PIXCON", -(eta * tro * (vars.PIXCON ** (eta - 1.0))))
            return result

        if constraint_name.startswith("EQ51_"):
            agd = constraint_name.split("_", 1)[1]
            eta = self.equations.params.get("eta", 0.0)
            tro = self.equations.params.get("TRO", {}).get((agd, "row"), 0.0)
            add(f"TR[{agd},row]", 1.0)
            if eta != 0 and vars.PIXCON > 0 and tro != 0:
                add("PIXCON", -(eta * tro * (vars.PIXCON ** (eta - 1.0))))
            return result

        if constraint_name == "EQ53":
            add("GFCF", 1.0)
            add("IT", -1.0)
            for i in self.sets.get("I", []):
                add(f"PC[{i}]", vars.VSTK.get(i, 0.0))
                add(f"VSTK[{i}]", vars.PC.get(i, 0.0))
            return result

        if constraint_name.startswith("EQ54_"):
            i = constraint_name.split("_", 1)[1]
            gamma_inv = self.equations.params.get("gamma_INV", {}).get(i, 0.0)
            add(f"PC[{i}]", vars.INV.get(i, 0.0))
            add(f"INV[{i}]", vars.PC.get(i, 0.0))
            add("GFCF", -gamma_inv)
            return result

        if constraint_name.startswith("EQ55_"):
            i = constraint_name.split("_", 1)[1]
            gamma_gvt = self.equations.params.get("gamma_GVT", {}).get(i, 0.0)
            add(f"PC[{i}]", vars.CG.get(i, 0.0))
            add(f"CG[{i}]", vars.PC.get(i, 0.0))
            add("G", -gamma_gvt)
            return result

        if constraint_name.startswith("EQ56_"):
            i = constraint_name.split("_", 1)[1]
            add(f"DIT[{i}]", 1.0)
            for j in self.sets.get("J", []):
                add(f"DI[{i},{j}]", -1.0)
            return result

        if constraint_name.startswith("EQ57_"):
            i = constraint_name.split("_", 1)[1]
            add(f"MRGN[{i}]", 1.0)
            for ij in self.sets.get("I", []):
                tmrg = self.equations.params.get("tmrg", {}).get((i, ij), 0.0)
                if abs(self.equations.params.get("DDO0", {}).get(ij, 0.0)) > 1e-12:
                    add(f"DD[{ij}]", -tmrg)
                if abs(self.equations.params.get("IMO0", {}).get(ij, 0.0)) > 1e-12:
                    add(f"IM[{ij}]", -tmrg)
                if abs(self.equations.params.get("EXDO0", {}).get(ij, 0.0)) > 1e-12:
                    tmrg_x = self.equations.params.get("tmrg_X", {}).get((i, ij), 0.0)
                    add(f"EXD[{ij}]", -tmrg_x)
            return result

        if constraint_name.startswith("EQ58_"):
            j = constraint_name.split("_", 1)[1]
            rho_xt = self.equations.params.get("rho_XT", {}).get(j, 1.0)
            b_xt = self.equations.params.get("B_XT", {}).get(j, 1.0)
            if rho_xt == 0:
                return None
            term = 0.0
            active: list[tuple[str, float, float]] = []
            for i in self.sets.get("I", []):
                if abs(self.equations.params.get("XSO0", {}).get((j, i), 0.0)) <= 1e-12:
                    continue
                beta_xt = self.equations.params.get("beta_XT", {}).get((j, i), 0.0)
                xs_ji = vars.XS.get((j, i), 0.0)
                if beta_xt <= 0 or xs_ji <= 0:
                    return None
                active.append((i, beta_xt, xs_ji))
                term += beta_xt * (xs_ji ** rho_xt)
            if term <= 0 or b_xt <= 0:
                return None
            coeff = b_xt * (term ** ((1.0 / rho_xt) - 1.0))
            add(f"XST[{j}]", 1.0)
            for i, beta_xt, xs_ji in active:
                add(f"XS[{j},{i}]", -(coeff * beta_xt * (xs_ji ** (rho_xt - 1.0))))
            return result

        if constraint_name.startswith("EQ59_"):
            _, j, i = constraint_name.split("_", 2)
            sigma_xt = self.equations.params.get("sigma_XT", {}).get(j, 2.0)
            b_xt = self.equations.params.get("B_XT", {}).get(j, 1.0)
            beta_xt = self.equations.params.get("beta_XT", {}).get((j, i), 0.0)
            pt_j = vars.PT.get(j, 0.0)
            p_ji = vars.P.get((j, i), 0.0)
            xst_j = vars.XST.get(j, 0.0)
            if b_xt <= 0 or beta_xt <= 0 or pt_j <= 0 or p_ji <= 0:
                return None
            ratio = p_ji / (beta_xt * pt_j)
            scale_factor = (ratio ** sigma_xt) / (b_xt ** (1.0 + sigma_xt))
            expected_xs = xst_j * scale_factor
            add(f"XS[{j},{i}]", 1.0)
            add(f"XST[{j}]", -scale_factor)
            add(f"P[{j},{i}]", -(expected_xs * sigma_xt / p_ji))
            add(f"PT[{j}]", expected_xs * sigma_xt / pt_j)
            return result

        if constraint_name.startswith("EQ61_"):
            _, j, i = constraint_name.split("_", 2)
            beta_x = self.equations.params.get("beta_X", {}).get((j, i), 0.0)
            sigma_x = self.equations.params.get("sigma_X", {}).get((j, i), 2.0)
            pe_i = vars.PE.get(i, 0.0)
            pl_i = vars.PL.get(i, 0.0)
            ds_ji = vars.DS.get((j, i), 0.0)
            if beta_x <= 0 or beta_x >= 1 or pe_i <= 0 or pl_i <= 0:
                return None
            ratio_factor = ((1.0 - beta_x) / beta_x) * (pe_i / pl_i)
            if ratio_factor <= 0:
                return None
            alloc = ratio_factor ** sigma_x
            expected_ex = alloc * ds_ji
            add(f"EX[{j},{i}]", 1.0)
            add(f"DS[{j},{i}]", -alloc)
            add(f"PE[{i}]", -(expected_ex * sigma_x / pe_i))
            add(f"PL[{i}]", expected_ex * sigma_x / pl_i)
            return result

        if constraint_name.startswith("EQ60_"):
            _, j, i = constraint_name.split("_", 2)
            rho_x = self.equations.params.get("rho_X", {}).get((j, i), 1.0)
            b_x = self.equations.params.get("B_X", {}).get((j, i), 1.0)
            beta_x = self.equations.params.get("beta_X", {}).get((j, i), 0.5)
            if rho_x == 0 or b_x <= 0 or (j, i) not in self.equations.params.get("beta_X", {}):
                return None
            term = 0.0
            ex_ji = vars.EX.get((j, i), 0.0)
            ds_ji = vars.DS.get((j, i), 0.0)
            active_ex = abs(self.equations.params.get("EXO0", {}).get((j, i), 0.0)) > 1e-12
            active_ds = abs(self.equations.params.get("DSO0", {}).get((j, i), 0.0)) > 1e-12
            if active_ex:
                if beta_x <= 0 or ex_ji <= 0:
                    return None
                term += beta_x * (ex_ji ** rho_x)
            if active_ds:
                if beta_x >= 1 or ds_ji <= 0:
                    return None
                term += (1.0 - beta_x) * (ds_ji ** rho_x)
            if term <= 0:
                return None
            coeff = b_x * (term ** ((1.0 / rho_x) - 1.0))
            add(f"XS[{j},{i}]", 1.0)
            if active_ex:
                add(f"EX[{j},{i}]", -(coeff * beta_x * (ex_ji ** (rho_x - 1.0))))
            if active_ds:
                add(f"DS[{j},{i}]", -(coeff * (1.0 - beta_x) * (ds_ji ** (rho_x - 1.0))))
            return result

        if constraint_name.startswith("EQ62_"):
            i = constraint_name.split("_", 1)[1]
            sigma_xd = self.equations.params.get("sigma_XD", {}).get(i, 1.0)
            exdo = self.equations.params.get("EXDO", {}).get(i, 0.0)
            pwx_i = vars.PWX.get(i, self.equations.params.get("PWX", {}).get(i, vars.PWM.get(i, 1.0)))
            pe_fob_i = vars.PE_FOB.get(i, 0.0)
            if abs(exdo) <= 1e-12 or pwx_i <= 0 or pe_fob_i <= 0 or vars.e <= 0:
                return None
            expected_exd = exdo * ((vars.e * pwx_i) / pe_fob_i) ** sigma_xd
            add(f"EXD[{i}]", 1.0)
            add("e", -(expected_exd * sigma_xd / vars.e))
            add(f"PWX[{i}]", -(expected_exd * sigma_xd / pwx_i))
            add(f"PE_FOB[{i}]", expected_exd * sigma_xd / pe_fob_i)
            return result

        if constraint_name.startswith("EQ63_"):
            i = constraint_name.split("_", 1)[1]
            rho_m = self.equations.params.get("rho_M", {}).get(i, -0.5)
            b_m = self.equations.params.get("B_M", {}).get(i, 1.0)
            beta_m = self.equations.params.get("beta_M", {}).get(i, 0.5)
            if rho_m == 0 or b_m <= 0:
                return None
            term = 0.0
            im_i = vars.IM.get(i, 0.0)
            dd_i = vars.DD.get(i, 0.0)
            active_im = abs(self.equations.params.get("IMO0", {}).get(i, 0.0)) > 1e-12
            active_dd = abs(self.equations.params.get("DDO0", {}).get(i, 0.0)) > 1e-12
            if active_im:
                if beta_m <= 0 or im_i <= 0:
                    return None
                term += beta_m * (im_i ** (-rho_m))
            if active_dd:
                if beta_m >= 1 or dd_i <= 0:
                    return None
                term += (1.0 - beta_m) * (dd_i ** (-rho_m))
            if term <= 0:
                return None
            coeff = b_m * (term ** ((-1.0 / rho_m) - 1.0))
            add(f"Q[{i}]", 1.0)
            if active_im:
                add(f"IM[{i}]", -(coeff * beta_m * (im_i ** (-rho_m - 1.0))))
            if active_dd:
                add(f"DD[{i}]", -(coeff * (1.0 - beta_m) * (dd_i ** (-rho_m - 1.0))))
            return result

        if constraint_name.startswith("EQ64_"):
            i = constraint_name.split("_", 1)[1]
            beta_m = self.equations.params.get("beta_M", {}).get(i, 0.0)
            sigma_m = self.equations.params.get("sigma_M", {}).get(i, 2.0)
            pd_i = vars.PD.get(i, 0.0)
            pm_i = vars.PM.get(i, 0.0)
            dd_i = vars.DD.get(i, 0.0)
            if beta_m <= 0 or beta_m >= 1 or pd_i <= 0 or pm_i <= 0:
                return None
            ratio_factor = (beta_m / (1.0 - beta_m)) * (pd_i / pm_i)
            if ratio_factor <= 0:
                return None
            alloc = ratio_factor ** sigma_m
            expected_im = alloc * dd_i
            add(f"IM[{i}]", 1.0)
            add(f"DD[{i}]", -alloc)
            add(f"PD[{i}]", -(expected_im * sigma_m / pd_i))
            add(f"PM[{i}]", expected_im * sigma_m / pm_i)
            return result

        if constraint_name.startswith("EQ66_"):
            j = constraint_name.split("_", 1)[1]
            ttip = self.equations.params.get("ttip", {}).get(j, 0.0)
            add(f"PT[{j}]", 1.0)
            add(f"PP[{j}]", -(1.0 + ttip))
            return result

        if constraint_name.startswith("EQ65_"):
            j = constraint_name.split("_", 1)[1]
            add(f"PP[{j}]", vars.XST.get(j, 0.0))
            add(f"XST[{j}]", vars.PP.get(j, 0.0))
            add(f"PVA[{j}]", -vars.VA.get(j, 0.0))
            add(f"VA[{j}]", -vars.PVA.get(j, 0.0))
            add(f"PCI[{j}]", -vars.CI.get(j, 0.0))
            add(f"CI[{j}]", -vars.PCI.get(j, 0.0))
            return result

        if constraint_name.startswith("EQ67_"):
            j = constraint_name.split("_", 1)[1]
            add(f"PCI[{j}]", vars.CI.get(j, 0.0))
            add(f"CI[{j}]", vars.PCI.get(j, 0.0))
            for i in self.sets.get("I", []):
                add(f"PC[{i}]", -vars.DI.get((i, j), 0.0))
                add(f"DI[{i},{j}]", -vars.PC.get(i, 0.0))
            return result

        if constraint_name.startswith("EQ68_"):
            j = constraint_name.split("_", 1)[1]
            add(f"PVA[{j}]", vars.VA.get(j, 0.0))
            add(f"VA[{j}]", vars.PVA.get(j, 0.0))
            if any(abs(self.equations.params.get("LDO0", {}).get((l, j), 0.0)) > 1e-12 for l in self.sets.get("L", [])):
                add(f"WC[{j}]", -vars.LDC.get(j, 0.0))
                add(f"LDC[{j}]", -vars.WC.get(j, 0.0))
            if any(abs(self.equations.params.get("KDO0", {}).get((k, j), 0.0)) > 1e-12 for k in self.sets.get("K", [])):
                add(f"RC[{j}]", -vars.KDC.get(j, 0.0))
                add(f"KDC[{j}]", -vars.RC.get(j, 0.0))
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

        if constraint_name.startswith("EQ75_"):
            _, j, i = constraint_name.split("_", 2)
            add(f"P[{j},{i}]", vars.XS.get((j, i), 0.0))
            add(f"XS[{j},{i}]", vars.P.get((j, i), 0.0))
            if abs(self.equations.params.get("EXO0", {}).get((j, i), 0.0)) > 1e-12:
                add(f"PE[{i}]", -vars.EX.get((j, i), 0.0))
                add(f"EX[{j},{i}]", -vars.PE.get(i, 0.0))
            if abs(self.equations.params.get("DSO0", {}).get((j, i), 0.0)) > 1e-12:
                add(f"PL[{i}]", -vars.DS.get((j, i), 0.0))
                add(f"DS[{j},{i}]", -vars.PL.get(i, 0.0))
            return result

        if constraint_name.startswith("EQ76_"):
            i = constraint_name.split("_", 1)[1]
            ttix = self.equations.params.get("ttix", {}).get(i, 0.0)
            add(f"PE[{i}]", 1.0)
            add(f"PE_FOB[{i}]", -(1.0 / (1.0 + ttix)))
            for ij in self.sets.get("I", []):
                add(f"PC[{ij}]", self.equations.params.get("tmrg_X", {}).get((ij, i), 0.0))
            return result

        if constraint_name.startswith("EQ77_"):
            i = constraint_name.split("_", 1)[1]
            ttic = self.equations.params.get("ttic", {}).get(i, 0.0)
            factor = 1.0 + ttic
            add(f"PD[{i}]", 1.0)
            add(f"PL[{i}]", -factor)
            for ij in self.sets.get("I", []):
                add(f"PC[{ij}]", -(factor * self.equations.params.get("tmrg", {}).get((ij, i), 0.0)))
            return result

        if constraint_name.startswith("EQ78_"):
            i = constraint_name.split("_", 1)[1]
            ttim = self.equations.params.get("ttim", {}).get(i, 0.0)
            ttic = self.equations.params.get("ttic", {}).get(i, 0.0)
            factor = (1.0 + ttic) * (1.0 + ttim)
            add(f"PM[{i}]", 1.0)
            add("e", -(factor * vars.PWM.get(i, 0.0)))
            add(f"PWM[{i}]", -(factor * vars.e))
            for ij in self.sets.get("I", []):
                add(f"PC[{ij}]", -((1.0 + ttic) * self.equations.params.get("tmrg", {}).get((ij, i), 0.0)))
            return result

        if constraint_name.startswith("EQ79_"):
            i = constraint_name.split("_", 1)[1]
            add(f"PC[{i}]", vars.Q.get(i, 0.0))
            add(f"Q[{i}]", vars.PC.get(i, 0.0))
            if abs(self.equations.params.get("IMO0", {}).get(i, 0.0)) > 1e-12:
                add(f"PM[{i}]", -vars.IM.get(i, 0.0))
                add(f"IM[{i}]", -vars.PM.get(i, 0.0))
            if abs(self.equations.params.get("DDO0", {}).get(i, 0.0)) > 1e-12:
                add(f"PD[{i}]", -vars.DD.get(i, 0.0))
                add(f"DD[{i}]", -vars.PD.get(i, 0.0))
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

        if constraint_name.startswith("EQ84_"):
            i = constraint_name.split("_", 1)[1]
            add(f"Q[{i}]", 1.0)
            for h in self.sets.get("H", []):
                add(f"C[{i},{h}]", -1.0)
            add(f"CG[{i}]", -1.0)
            add(f"INV[{i}]", -1.0)
            add(f"VSTK[{i}]", -1.0)
            add(f"DIT[{i}]", -1.0)
            add(f"MRGN[{i}]", -1.0)
            return result

        if constraint_name.startswith("EQ85_"):
            l = constraint_name.split("_", 1)[1]
            add(f"LS[{l}]", 1.0)
            for j in self.sets.get("J", []):
                add(f"LD[{l},{j}]", -1.0)
            return result

        if constraint_name.startswith("EQ86_"):
            k = constraint_name.split("_", 1)[1]
            add(f"KS[{k}]", 1.0)
            for j in self.sets.get("J", []):
                add(f"KD[{k},{j}]", -1.0)
            return result

        if constraint_name == "EQ87":
            add("IT", 1.0)
            for h in self.sets.get("H", []):
                add(f"SH[{h}]", -1.0)
            for f in self.sets.get("F", []):
                add(f"SF[{f}]", -1.0)
            add("SG", -1.0)
            add("SROW", -1.0)
            return result

        if constraint_name.startswith("EQ88_"):
            i = constraint_name.split("_", 1)[1]
            for j in self.sets.get("J", []):
                if abs(self.equations.params.get("DSO0", {}).get((j, i), 0.0)) > 1e-12:
                    add(f"DS[{j},{i}]", 1.0)
            add(f"DD[{i}]", -1.0)
            return result

        if constraint_name.startswith("EQ89_"):
            i = constraint_name.split("_", 1)[1]
            for j in self.sets.get("J", []):
                if abs(self.equations.params.get("EXO0", {}).get((j, i), 0.0)) > 1e-12:
                    add(f"EX[{j},{i}]", 1.0)
            add(f"EXD[{i}]", -1.0)
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

        if constraint_name == "EQ92":
            add("GDP_IB", 1.0)
            add("TPRODN", -1.0)
            add("TPRCTS", -1.0)
            for l in self.sets.get("L", []):
                for j in self.sets.get("J", []):
                    if abs(self.equations.params.get("LDO0", {}).get((l, j), 0.0)) <= 1e-12:
                        continue
                    add(f"W[{l}]", -vars.LD.get((l, j), 0.0))
                    add(f"LD[{l},{j}]", -vars.W.get(l, 0.0))
            for k in self.sets.get("K", []):
                for j in self.sets.get("J", []):
                    if abs(self.equations.params.get("KDO0", {}).get((k, j), 0.0)) <= 1e-12:
                        continue
                    add(f"R[{k},{j}]", -vars.KD.get((k, j), 0.0))
                    add(f"KD[{k},{j}]", -vars.R.get((k, j), 0.0))
            return result

        if constraint_name == "EQ93":
            add("GDP_FD", 1.0)
            for i in self.sets.get("I", []):
                total_absorption = (
                    sum(vars.C.get((i, h), 0.0) for h in self.sets.get("H", []))
                    + vars.CG.get(i, 0.0)
                    + vars.INV.get(i, 0.0)
                    + vars.VSTK.get(i, 0.0)
                )
                add(f"PC[{i}]", -total_absorption)
                for h in self.sets.get("H", []):
                    add(f"C[{i},{h}]", -vars.PC.get(i, 0.0))
                add(f"CG[{i}]", -vars.PC.get(i, 0.0))
                add(f"INV[{i}]", -vars.PC.get(i, 0.0))
                add(f"VSTK[{i}]", -vars.PC.get(i, 0.0))
                if abs(self.equations.params.get("EXDO0", {}).get(i, 0.0)) > 1e-12:
                    add(f"PE_FOB[{i}]", -vars.EXD.get(i, 0.0))
                    add(f"EXD[{i}]", -vars.PE_FOB.get(i, 0.0))
                if abs(self.equations.params.get("IMO0", {}).get(i, 0.0)) > 1e-12:
                    add(f"PWM[{i}]", vars.e * vars.IM.get(i, 0.0))
                    add("e", vars.PWM.get(i, 0.0) * vars.IM.get(i, 0.0))
                    add(f"IM[{i}]", vars.PWM.get(i, 0.0) * vars.e)
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

        if constraint_name == "WALRAS":
            walras_i = "agr" if "agr" in self.sets.get("I", []) else (self.sets.get("I", [None])[0])
            if walras_i is None:
                return result
            add("LEON", 1.0)
            add(f"Q[{walras_i}]", -1.0)
            for h in self.sets.get("H", []):
                add(f"C[{walras_i},{h}]", 1.0)
            add(f"CG[{walras_i}]", 1.0)
            add(f"INV[{walras_i}]", 1.0)
            add(f"VSTK[{walras_i}]", 1.0)
            add(f"DIT[{walras_i}]", 1.0)
            add(f"MRGN[{walras_i}]", 1.0)
            return result

        return None
