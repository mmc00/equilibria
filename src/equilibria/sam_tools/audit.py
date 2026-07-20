"""Auditoría universal de SAMs — checks SCN-2008 aplicables a cualquier país.

Uso básico:
    from equilibria.sam_tools.audit import audit_sam, AnchorSpec
    import pandas as pd

    df = pd.read_csv("mi_sam.csv", index_col=0)
    result = audit_sam(df)
    print(result.summary())

Con anchors país-específicos:
    anchors = [
        AnchorSpec("X_total", [("Agri","ROW"),("Ind","ROW"),("Serv","ROW")],
                   expected_min=11_000, expected_max=13_000, source="BoP MBP6"),
    ]
    result = audit_sam(df, anchors=anchors)
    result.to_markdown("audit_report.md")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

# ---------------------------------------------------------------------------
# Tipos de datos
# ---------------------------------------------------------------------------


@dataclass
class AnchorSpec:
    """Flujo conocido con rango publicado — para validación país-específica."""

    name: str
    cells: list[tuple[str, str]]  # [(row, col), ...]
    expected_min: float
    expected_max: float
    source: str = ""  # ej. "BoP MBP6 2023"


@dataclass
class CheckResult:
    name: str
    passed: bool
    value: Any
    detail: str = ""


@dataclass
class AuditResult:
    """Resultado completo de la auditoría."""

    checks: list[CheckResult] = field(default_factory=list)
    anchor_checks: list[CheckResult] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.checks + self.anchor_checks)

    @property
    def n_passed(self) -> int:
        return sum(c.passed for c in self.checks + self.anchor_checks)

    @property
    def n_total(self) -> int:
        return len(self.checks) + len(self.anchor_checks)

    def summary(self) -> str:
        lines = [f"Auditoría SAM — {self.n_passed}/{self.n_total} checks OK\n"]
        for c in self.checks + self.anchor_checks:
            icon = "✅" if c.passed else "❌"
            lines.append(f"  {icon} {c.name}: {c.detail}")
        return "\n".join(lines)

    def to_markdown(self, path: str | Path | None = None) -> str:
        lines = [
            "# Auditoría SAM\n",
            f"**Resultado:** {self.n_passed}/{self.n_total} checks OK  ",
            f"**Estado:** {'PASSED ✅' if self.passed else 'FAILED ❌'}\n",
            "## Checks universales SCN-2008\n",
            "| Check | Estado | Valor | Detalle |",
            "|-------|--------|-------|---------|",
        ]
        for c in self.checks:
            icon = "✅" if c.passed else "❌"
            val = f"{c.value:.4f}" if isinstance(c.value, float) else str(c.value)
            lines.append(f"| {c.name} | {icon} | {val} | {c.detail} |")

        if self.anchor_checks:
            lines += [
                "\n## Anchors país-específicos\n",
                "| Anchor | Estado | Valor | Rango esperado | Fuente |",
                "|--------|--------|-------|---------------|--------|",
            ]
            for c in self.anchor_checks:
                icon = "✅" if c.passed else "❌"
                val = f"{c.value:,.2f}" if isinstance(c.value, float) else str(c.value)
                lines.append(f"| {c.name} | {icon} | {val} | {c.detail} |")

        md = "\n".join(lines)
        if path is not None:
            Path(path).write_text(md, encoding="utf-8")
        return md

    def failing(self) -> list[CheckResult]:
        return [c for c in self.checks + self.anchor_checks if not c.passed]


# ---------------------------------------------------------------------------
# Checks universales
# ---------------------------------------------------------------------------


def _check_balance(df: pd.DataFrame, tol: float = 1e-4) -> CheckResult:
    """Fila = columna para cada cuenta (condición SAM fundamental)."""
    residuals = df.sum(axis=1) - df.sum(axis=0)
    max_res = float(residuals.abs().max())
    worst = residuals.abs().idxmax()
    passed = max_res <= tol
    return CheckResult(
        name="Balance fila=col",
        passed=passed,
        value=max_res,
        detail=(
            f"max residual={max_res:.2e} en '{worst}' (tol={tol:.0e})"
            if not passed
            else f"max residual={max_res:.2e}"
        ),
    )


def _check_non_negative(df: pd.DataFrame) -> CheckResult:
    """Todos los valores ≥ 0 (convención SAM estándar)."""
    neg = df[df < -1e-9]
    n_neg = int((neg < -1e-9).sum().sum())
    passed = n_neg == 0
    if not passed:
        worst_val = float(neg.min().min())
        idx = neg.stack().idxmin()
        detail = f"{n_neg} celdas negativas; peor: {idx}={worst_val:.4f}"
    else:
        detail = "todas las celdas ≥ 0"
    return CheckResult(name="No negativos", passed=passed, value=n_neg, detail=detail)


def _check_square(df: pd.DataFrame) -> CheckResult:
    """Matriz cuadrada con mismas cuentas en filas y columnas."""
    rows, cols = df.shape
    square = rows == cols
    same_accounts = set(df.index.tolist()) == set(df.columns.tolist())
    passed = square and same_accounts
    detail = (
        f"{rows}×{cols}, cuentas filas==cols: {same_accounts}"
        if not passed
        else f"OK ({rows}×{rows})"
    )
    return CheckResult(
        name="Cuadrada y simétrica", passed=passed, value=rows, detail=detail
    )


def _check_pib_expenditure(
    df: pd.DataFrame,
    activity_accounts: list[str],
    factor_accounts: list[str],
    household_account: str | None,
    government_account: str | None,
    investment_account: str | None,
    row_account: str | None,
    tprod_account: str | None = None,
    tol_pct: float = 2.0,
) -> CheckResult:
    """PIB(P) ≈ PIB(G) dentro de tol_pct%.

    PIB(P) = Σ VA + T_prod_neto
    PIB(G) = C_h + C_g + I + X − M
    """
    accounts = list(df.index)

    # PIB por el lado de la producción
    va_total = 0.0
    for fac in factor_accounts:
        if fac in accounts:
            for act in activity_accounts:
                if act in accounts:
                    va_total += float(df.loc[fac, act])
    t_prod = 0.0
    if tprod_account and tprod_account in accounts:
        for act in activity_accounts:
            if act in accounts:
                t_prod += float(df.loc[tprod_account, act])
    pib_p = va_total + t_prod

    # PIB por el lado del gasto
    pib_g = 0.0
    for act in activity_accounts:
        if act not in accounts:
            continue
        if household_account and household_account in accounts:
            pib_g += float(df.loc[act, household_account])
        if government_account and government_account in accounts:
            pib_g += float(df.loc[act, government_account])
        if investment_account and investment_account in accounts:
            pib_g += float(df.loc[act, investment_account])
        if row_account and row_account in accounts:
            pib_g += float(df.loc[act, row_account])  # X
            pib_g -= float(df.loc[row_account, act])  # − M int

    if row_account and row_account in accounts:
        for acc in [household_account, government_account, investment_account]:
            if acc and acc in accounts:
                pib_g -= float(df.loc[row_account, acc])  # − M finales

    if abs(pib_p) < 1e-9:
        return CheckResult("PIB(P)≈PIB(G)", False, 0.0, "PIB(P)=0, revisa cuentas")

    gap_pct = abs(pib_p - pib_g) / abs(pib_p) * 100
    passed = gap_pct <= tol_pct
    detail = (
        f"PIB(P)={pib_p:,.2f}, PIB(G)={pib_g:,.2f}, gap={gap_pct:.2f}% (tol={tol_pct}%)"
    )
    return CheckResult(
        name="PIB(P)≈PIB(G)", passed=passed, value=gap_pct, detail=detail
    )


def _check_s_minus_i_plus_b9(
    df: pd.DataFrame,
    capital_account: str | None,
    row_account: str | None,
    tol: float = 1.0,
) -> CheckResult:
    """S − I + B9_ROW ≈ 0 (identidad ahorro-inversión externa).

    En la SAM: la cuenta de capital absorbe el ahorro (columna) y financia
    la inversión (fila). B9_ROW = (CC, ROW) − (ROW, CC).
    """
    if not capital_account or not row_account:
        return CheckResult("S−I+B9=0", True, 0.0, "omitido (cuentas no especificadas)")
    accounts = list(df.index)
    if capital_account not in accounts or row_account not in accounts:
        return CheckResult(
            "S−I+B9=0",
            False,
            0.0,
            f"cuenta '{capital_account}' o '{row_account}' no encontrada",
        )
    cc_row = float(df.loc[capital_account, row_account])
    row_cc = float(df.loc[row_account, capital_account])
    b9_row = cc_row - row_cc
    # S = Σ col de CC (ahorro institucional), I = Σ fila de CC (inversión)
    s = float(df[capital_account].sum()) - row_cc
    i = float(df.loc[capital_account].sum()) - cc_row
    gap = abs(s - i + b9_row)
    passed = gap <= tol
    detail = f"S={s:,.2f}, I={i:,.2f}, B9={b9_row:,.2f}, |S−I+B9|={gap:.4f}"
    return CheckResult(name="S−I+B9=0", passed=passed, value=gap, detail=detail)


def _check_diagonal(df: pd.DataFrame, tol: float = 1e-9) -> CheckResult:
    """Diagonal nula (flujo de una cuenta a sí misma es raro; advertencia si >5% del total)."""
    n = len(df)
    diag_sum = sum(float(df.iloc[i, i]) for i in range(n))
    total = float(df.values.sum())
    pct = diag_sum / total * 100 if total > 0 else 0.0
    # Advertencia blanda: diagonal puede existir (ej. Hogares→Hogares por D7)
    passed = pct < 10.0
    detail = f"diagonal={diag_sum:,.2f} ({pct:.1f}% del total)"
    return CheckResult(
        name="Diagonal <10% total", passed=passed, value=pct, detail=detail
    )


def _check_anchors(df: pd.DataFrame, anchors: list[AnchorSpec]) -> list[CheckResult]:
    results = []
    for anc in anchors:
        total = 0.0
        for r, c in anc.cells:
            if r in df.index and c in df.columns:
                total += float(df.loc[r, c])
        passed = anc.expected_min <= total <= anc.expected_max
        detail = (
            f"[{anc.expected_min:,.0f}, {anc.expected_max:,.0f}]  fuente: {anc.source}"
        )
        results.append(
            CheckResult(
                name=anc.name,
                passed=passed,
                value=total,
                detail=detail,
            )
        )
    return results


# ---------------------------------------------------------------------------
# Función principal
# ---------------------------------------------------------------------------


def audit_sam(
    df: pd.DataFrame,
    *,
    activity_accounts: list[str] | None = None,
    factor_accounts: list[str] | None = None,
    household_account: str | None = None,
    government_account: str | None = None,
    investment_account: str | None = None,
    capital_account: str | None = None,
    row_account: str | None = None,
    tprod_account: str | None = None,
    anchors: list[AnchorSpec] | None = None,
    balance_tol: float = 1e-4,
    pib_tol_pct: float = 2.0,
) -> AuditResult:
    """Audita una SAM DataFrame contra criterios universales SCN-2008.

    Parámetros:
        df: SAM cuadrada (pd.DataFrame, index=filas=receptores, columns=pagadores).
        activity_accounts: cuentas de actividad/sector productivo (ej. ["Agri","Ind","Serv"]).
        factor_accounts: cuentas de factor (ej. ["Trabajo","Capital"]).
        household_account: cuenta de hogares (ej. "Hogares").
        government_account: cuenta de gobierno (ej. "Gobierno").
        investment_account: cuenta de inversión (ej. "Inversion").
        capital_account: cuenta de capital/ahorro (ej. "Cuenta_Capital").
        row_account: cuenta del resto del mundo (ej. "ROW").
        tprod_account: cuenta de impuestos a la producción (ej. "T_prod").
        anchors: lista de AnchorSpec con flujos conocidos país-específicos.
        balance_tol: tolerancia para fila=columna (default 1e-4).
        pib_tol_pct: tolerancia % para PIB(P)≈PIB(G) (default 2%).

    Devuelve AuditResult con todos los checks y métodos summary()/to_markdown().
    """
    result = AuditResult()

    # Checks universales siempre
    result.checks.append(_check_square(df))
    result.checks.append(_check_non_negative(df))
    result.checks.append(_check_balance(df, tol=balance_tol))
    result.checks.append(_check_diagonal(df))

    # Checks macro (requieren especificar cuentas)
    if activity_accounts and factor_accounts:
        result.checks.append(
            _check_pib_expenditure(
                df,
                activity_accounts=activity_accounts,
                factor_accounts=factor_accounts,
                household_account=household_account,
                government_account=government_account,
                investment_account=investment_account,
                row_account=row_account,
                tprod_account=tprod_account,
                tol_pct=pib_tol_pct,
            )
        )

    if capital_account and row_account:
        result.checks.append(_check_s_minus_i_plus_b9(df, capital_account, row_account))

    # Anchors país-específicos
    if anchors:
        result.anchor_checks.extend(_check_anchors(df, anchors))

    return result
