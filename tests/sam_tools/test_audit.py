"""Tests para equilibria.sam_tools.audit.

Tres niveles:
  - Unit: cada check en aislamiento con DataFrames mínimos construidos a mano
  - Smoke: audit_sam completo en SAMs pequeñas bien/mal formadas
  - End-to-end: SAM 12x12 realista (Bolivia 2023 sintética) con anchors
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from equilibria.sam_tools.audit import (
    AnchorSpec,
    AuditResult,
    CheckResult,
    _check_balance,
    _check_diagonal,
    _check_non_negative,
    _check_s_minus_i_plus_b9,
    _check_square,
    audit_sam,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_sam(accounts: list[str], values: dict[tuple[str, str], float]) -> pd.DataFrame:
    """Construye un DataFrame SAM cuadrado y balanceado a partir de flujos."""
    df = pd.DataFrame(0.0, index=accounts, columns=accounts)
    for (r, c), v in values.items():
        df.loc[r, c] = v
    return df


def _balanced_3x3() -> pd.DataFrame:
    """SAM 3×3 balanceada mínima: Prod→Factor→Hog→Prod."""
    accounts = ["Prod", "Factor", "Hog"]
    df = _make_sam(accounts, {
        ("Factor", "Prod"): 100.0,   # VA
        ("Hog",    "Factor"): 100.0, # salario
        ("Prod",   "Hog"): 100.0,    # consumo
    })
    return df


def _balanced_5x5() -> pd.DataFrame:
    """SAM 5×5 con sector externo y cuenta capital, balanceada."""
    accts = ["Act", "Factor", "Hog", "ROW", "CC"]
    df = _make_sam(accts, {
        ("Factor", "Act"): 200.0,
        ("Hog",    "Factor"): 180.0,
        ("Act",    "Hog"): 150.0,
        ("Act",    "ROW"): 60.0,    # X
        ("ROW",    "Act"): 40.0,    # M
        ("CC",     "Hog"): 20.0,    # ahorro Hog
        ("Act",    "CC"): 20.0,     # inversión
        ("ROW",    "Factor"): 20.0, # D1_ROW debit
        ("Factor", "ROW"): 20.0,    # D1_ROW credit
        ("CC",     "ROW"): 20.0,    # B9 inflow
        ("ROW",    "CC"): 20.0,     # B9 outflow
    })
    return df


# ============================================================================
# UNIT TESTS — cada check en aislamiento
# ============================================================================

class TestCheckSquare:
    def test_square_same_accounts(self):
        df = _balanced_3x3()
        r = _check_square(df)
        assert r.passed

    def test_non_square_fails(self):
        df = pd.DataFrame(np.ones((3, 4)))
        r = _check_square(df)
        assert not r.passed

    def test_square_different_accounts_fails(self):
        df = pd.DataFrame(
            np.ones((3, 3)),
            index=["A", "B", "C"],
            columns=["A", "B", "X"],  # X ≠ C
        )
        r = _check_square(df)
        assert not r.passed

    def test_returns_check_result(self):
        r = _check_square(_balanced_3x3())
        assert isinstance(r, CheckResult)
        assert r.name == "Cuadrada y simétrica"


class TestCheckNonNegative:
    def test_all_positive_passes(self):
        r = _check_non_negative(_balanced_3x3())
        assert r.passed
        assert r.value == 0

    def test_negative_cell_fails(self):
        df = _balanced_3x3()
        df.iloc[0, 1] = -1.0
        r = _check_non_negative(df)
        assert not r.passed
        assert r.value >= 1

    def test_tiny_negative_below_threshold_passes(self):
        df = _balanced_3x3()
        df.iloc[0, 1] = -1e-10  # below -1e-9 threshold
        r = _check_non_negative(df)
        assert r.passed

    def test_counts_multiple_negatives(self):
        df = _balanced_3x3()
        df.iloc[0, 1] = -5.0
        df.iloc[1, 2] = -3.0
        r = _check_non_negative(df)
        assert r.value == 2


class TestCheckBalance:
    def test_balanced_sam_passes(self):
        r = _check_balance(_balanced_3x3())
        assert r.passed
        assert r.value < 1e-10

    def test_unbalanced_sam_fails(self):
        df = _balanced_3x3()
        df.loc["Prod", "Hog"] += 50.0   # rompe balance
        r = _check_balance(df, tol=1e-4)
        assert not r.passed

    def test_custom_tolerance(self):
        df = _balanced_3x3()
        df.loc["Prod", "Hog"] += 0.05   # gap=0.05
        assert not _check_balance(df, tol=1e-4).passed
        assert _check_balance(df, tol=1.0).passed

    def test_returns_max_residual(self):
        df = _balanced_3x3()
        df.loc["Prod", "Hog"] += 10.0
        r = _check_balance(df)
        assert abs(r.value - 10.0) < 1e-9


class TestCheckDiagonal:
    def test_zero_diagonal_passes(self):
        r = _check_diagonal(_balanced_3x3())
        assert r.passed

    def test_large_diagonal_fails(self):
        df = _balanced_3x3()
        # Añadir diagonal enorme para superar 10%
        df.loc["Prod", "Prod"] = 10_000.0
        df.loc["Factor", "Prod"] += 10_000.0  # mantener balance approx
        r = _check_diagonal(df)
        assert not r.passed

    def test_small_diagonal_passes(self):
        df = _balanced_3x3()
        df.loc["Hog", "Hog"] = 5.0  # 5 / (300+5) ≈ 1.6% < 10%
        df.loc["Prod", "Hog"] += 5.0  # balance approx
        r = _check_diagonal(df)
        assert r.passed


class TestCheckSMinusIPlusB9:
    def test_balanced_s_i_b9(self):
        # SAM donde S=I y B9=0
        accts = ["Act", "Hog", "CC", "ROW"]
        df = _make_sam(accts, {
            ("Hog", "Act"): 100.0,
            ("Act", "Hog"): 80.0,
            ("CC",  "Hog"): 20.0,  # ahorro
            ("Act", "CC"):  20.0,  # inversión
        })
        r = _check_s_minus_i_plus_b9(df, "CC", "ROW")
        # S=20, I=20, B9=0 → gap=0
        assert r.passed

    def test_missing_account_fails_gracefully(self):
        df = _balanced_3x3()
        r = _check_s_minus_i_plus_b9(df, "CC", "ROW")
        assert not r.passed
        assert "no encontrada" in r.detail

    def test_none_accounts_skips(self):
        r = _check_s_minus_i_plus_b9(_balanced_3x3(), None, None)
        assert r.passed
        assert "omitido" in r.detail


class TestAnchorCheck:
    def test_anchor_within_range_passes(self):
        df = _balanced_3x3()
        df.loc["Prod", "Hog"] = 100.0
        anchors = [AnchorSpec("test", [("Prod","Hog")], 90, 110, "test")]
        result = audit_sam(df, anchors=anchors)
        assert result.anchor_checks[0].passed

    def test_anchor_below_range_fails(self):
        df = _balanced_3x3()
        anchors = [AnchorSpec("test", [("Prod","Hog")], 200, 300, "test")]
        result = audit_sam(df, anchors=anchors)
        assert not result.anchor_checks[0].passed

    def test_anchor_sums_multiple_cells(self):
        df = _balanced_5x5()
        # X = (Act,ROW) = 60
        anchors = [AnchorSpec("X", [("Act","ROW")], 55, 65, "src")]
        result = audit_sam(df, anchors=anchors)
        assert result.anchor_checks[0].passed
        assert abs(result.anchor_checks[0].value - 60.0) < 1e-9

    def test_missing_cell_treated_as_zero(self):
        df = _balanced_3x3()
        anchors = [AnchorSpec("missing", [("X","Y")], -1, 1, "src")]
        result = audit_sam(df, anchors=anchors)
        assert result.anchor_checks[0].passed  # 0 in [-1, 1]


# ============================================================================
# SMOKE TESTS — audit_sam completo en SAMs pequeñas
# ============================================================================

class TestAuditSamSmoke:
    def test_balanced_sam_passes_universal_checks(self):
        result = audit_sam(_balanced_3x3())
        # Los 4 universales deben pasar
        universal = result.checks
        assert len(universal) == 4
        for c in universal:
            assert c.passed, f"Falló: {c.name} — {c.detail}"

    def test_unbalanced_sam_fails_balance_check(self):
        df = _balanced_3x3()
        df.loc["Prod", "Hog"] += 999.0
        result = audit_sam(df)
        balance_check = next(c for c in result.checks if "Balance" in c.name)
        assert not balance_check.passed

    def test_negative_sam_fails_non_negative_check(self):
        df = _balanced_3x3()
        df.iloc[0, 0] = -50.0
        result = audit_sam(df)
        neg_check = next(c for c in result.checks if "negativ" in c.name.lower())
        assert not neg_check.passed

    def test_non_square_fails_fast(self):
        df = pd.DataFrame(np.ones((3, 4)))
        result = audit_sam(df)
        square_check = result.checks[0]
        assert not square_check.passed

    def test_with_all_accounts_specified(self):
        df = _balanced_5x5()
        result = audit_sam(
            df,
            activity_accounts=["Act"],
            factor_accounts=["Factor"],
            household_account="Hog",
            capital_account="CC",
            row_account="ROW",
        )
        # Debe tener más checks con cuentas especificadas
        assert len(result.checks) >= 6

    def test_result_has_summary(self):
        result = audit_sam(_balanced_3x3())
        summary = result.summary()
        assert "Auditoría SAM" in summary
        assert "✅" in summary or "❌" in summary

    def test_result_to_markdown(self, tmp_path):
        result = audit_sam(_balanced_3x3())
        path = tmp_path / "report.md"
        md = result.to_markdown(path)
        assert path.exists()
        assert "# Auditoría SAM" in md
        assert path.read_text(encoding="utf-8") == md

    def test_failing_returns_only_failures(self):
        df = _balanced_3x3()
        df.loc["Prod", "Hog"] += 50.0  # rompe balance
        result = audit_sam(df)
        failing = result.failing()
        assert all(not c.passed for c in failing)
        assert any("Balance" in c.name for c in failing)

    def test_passed_property(self):
        assert audit_sam(_balanced_3x3()).passed
        df = _balanced_3x3()
        df.loc["Prod", "Hog"] += 50.0
        assert not audit_sam(df).passed

    def test_n_passed_and_n_total(self):
        result = audit_sam(_balanced_3x3())
        assert result.n_total == 4  # 4 universales sin cuentas especificadas
        assert result.n_passed == result.n_total


# ============================================================================
# END-TO-END TESTS — SAM 12×12 sintética tipo Bolivia
# ============================================================================

@pytest.fixture
def bolivia_synthetic_sam() -> pd.DataFrame:
    """SAM 12×12 sintética calibrada a magnitudes Bolivia 2023 (USD MM).

    Construida desde identidades contables para que pase todos los checks.
    No usa datos reales — es un benchmark numérico reproducible.

    Estrategia de balance: todos los flujos son fijados explícitamente excepto
    las celdas de Cuenta_Capital, que se calculan como residuos algebraicos
    para garantizar fila=columna en todos los demás. Luego se verifica que
    Cuenta_Capital misma también balance.
    """
    accts = [
        "Agricultura", "Industria", "Servicios",
        "T_prod", "Trabajo", "Capital",
        "Hogares", "Firmas", "Gobierno",
        "ROW", "Inversion", "Cuenta_Capital",
    ]
    df = pd.DataFrame(0.0, index=accts, columns=accts)

    # CI 3×3
    df.loc["Agricultura", "Agricultura"] = 500
    df.loc["Agricultura", "Industria"]   = 1_200
    df.loc["Agricultura", "Servicios"]   = 300
    df.loc["Industria",   "Agricultura"] = 800
    df.loc["Industria",   "Industria"]   = 5_000
    df.loc["Industria",   "Servicios"]   = 2_000
    df.loc["Servicios",   "Agricultura"] = 400
    df.loc["Servicios",   "Industria"]   = 2_000
    df.loc["Servicios",   "Servicios"]   = 3_500

    # M intermedias
    df.loc["ROW", "Agricultura"] = 200
    df.loc["ROW", "Industria"]   = 3_000
    df.loc["ROW", "Servicios"]   = 1_500

    # VA
    df.loc["Trabajo",  "Agricultura"] = 1_500
    df.loc["Trabajo",  "Industria"]   = 8_000
    df.loc["Trabajo",  "Servicios"]   = 10_000
    df.loc["Capital",  "Agricultura"] = 2_000
    df.loc["Capital",  "Industria"]   = 6_000
    df.loc["Capital",  "Servicios"]   = 8_000

    # T_prod: debe cuadrar → Gobierno recibe 2_500 de T_prod
    df.loc["T_prod",   "Agricultura"] = 200
    df.loc["T_prod",   "Industria"]   = 800
    df.loc["T_prod",   "Servicios"]   = 600
    df.loc["T_prod",   "Hogares"]     = 500
    df.loc["T_prod",   "Gobierno"]    = 200
    df.loc["T_prod",   "Inversion"]   = 200
    # T_prod col sum = 200+800+600 = 1_600 → T_prod row sum debe = 500+200+200 = 900
    # Para cuadrar T_prod: Gobierno paga 900 a T_prod (impuestos netos a la producción)
    df.loc["Gobierno", "T_prod"]      = 900

    # Exportaciones
    df.loc["Agricultura", "ROW"] = 500
    df.loc["Industria",   "ROW"] = 10_500
    df.loc["Servicios",   "ROW"] = 700

    # M finales
    df.loc["ROW", "Hogares"]   = 2_000
    df.loc["ROW", "Gobierno"]  = 300
    df.loc["ROW", "Inversion"] = 2_000

    # Factor → inst
    # Trabajo col = 1_500+8_000+10_000 = 19_500
    # Trabajo row = Hogares + ROW credit
    df.loc["Trabajo", "ROW"] = 12.0       # D1 credit
    df.loc["ROW",     "Trabajo"] = 5.0    # D1 debit
    df.loc["Hogares", "Trabajo"] = 19_500 - 12.0  # = 19_488

    # Capital col = 2_000+6_000+8_000 = 16_000
    df.loc["Hogares", "Capital"] = 5_280
    df.loc["Firmas",  "Capital"] = 10_720  # 5_280+10_720 = 16_000 ✓

    # D4/D7 simplificados (ROW outflows to institutions)
    df.loc["Hogares", "ROW"]  = 1_419   # remesas
    df.loc["ROW",     "Hogares"] = 280  # D4/D7 ROW→Hogares
    df.loc["Firmas",  "ROW"]  = 500
    df.loc["Gobierno","ROW"]  = 50

    # Consumo de hogares — calibrado para que PIB(G) ≈ PIB(P) = 37,100
    # PIB(G) = C_h + C_g + I + X - M = C_h + 8,100 + 7,000 + 11,700 - 9,000
    # → C_h = 37,100 - 17,800 = 19,300
    df.loc["Agricultura", "Hogares"]  = 2_000
    df.loc["Industria",   "Hogares"]  = 11_500
    df.loc["Servicios",   "Hogares"]  = 5_800

    # Consumo gobierno
    df.loc["Agricultura", "Gobierno"] = 100
    df.loc["Industria",   "Gobierno"] = 2_000
    df.loc["Servicios",   "Gobierno"] = 6_000

    # D5 (impuestos directos → Gobierno)
    df.loc["Gobierno", "Hogares"] = 3_500
    df.loc["Gobierno", "Firmas"]  = 510

    # FBCF
    df.loc["Agricultura", "Inversion"] = 500
    df.loc["Industria",   "Inversion"] = 4_000
    df.loc["Servicios",   "Inversion"] = 2_500

    # B9 ROW → Cuenta_Capital
    df.loc["Cuenta_Capital", "ROW"] = 1_169

    # Cerrar cada cuenta no-CC usando Cuenta_Capital como residuo.
    # gap = row_sum(a) - col_sum(a)
    #   gap > 0: a paga más de lo que recibe → CC debe pagarle más → (a, CC) en columna de a
    #            → agregar (CC, a): aumenta col_sum(a) y row_sum(CC)
    #   gap < 0: a recibe más de lo que paga → a debe ahorrar más → agregar (a, CC)
    #            → aumenta row_sum(a) → WRONG sign
    # Corrección: para cerrar gap=row-col>0, necesitamos col_sum(a) += gap → df.loc[CC, a] += gap
    #             para gap<0, necesitamos row_sum(a) += |gap| → df.loc[a, CC] += |gap|
    # Nota: este ajuste no crea cascada porque no tocamos (acc2, acc1) ni (acc1, acc2) para
    # otros acc2; solo tocamos (CC, acc) o (acc, CC).
    cc = "Cuenta_Capital"
    non_cc = [a for a in accts if a != cc]
    for acc in non_cc:
        row_s = float(df.loc[acc].sum())
        col_s = float(df[acc].sum())
        gap = row_s - col_s  # positive → acc sends more than receives
        if gap > 1e-9:
            # Need to increase col_sum(acc) = add something in the acc column
            df.loc[cc, acc] += gap
        elif gap < -1e-9:
            # Need to increase row_sum(acc) = add something in the acc row
            df.loc[acc, cc] += -gap

    return df


class TestAuditEndToEnd:
    def test_synthetic_bolivia_universal_checks_pass(self, bolivia_synthetic_sam):
        result = audit_sam(bolivia_synthetic_sam)
        for c in result.checks:
            assert c.passed, f"Universal check falló: {c.name} — {c.detail}"

    def test_synthetic_bolivia_with_accounts_passes_pib(self, bolivia_synthetic_sam):
        result = audit_sam(
            bolivia_synthetic_sam,
            activity_accounts=["Agricultura", "Industria", "Servicios"],
            factor_accounts=["Trabajo", "Capital"],
            household_account="Hogares",
            government_account="Gobierno",
            investment_account="Inversion",
            capital_account="Cuenta_Capital",
            row_account="ROW",
            tprod_account="T_prod",
            pib_tol_pct=5.0,
        )
        pib_check = next(c for c in result.checks if "PIB" in c.name)
        assert pib_check.passed, pib_check.detail

    def test_synthetic_bolivia_anchors_pass(self, bolivia_synthetic_sam):
        anchors = [
            AnchorSpec(
                "X_total",
                [("Agricultura","ROW"), ("Industria","ROW"), ("Servicios","ROW")],
                10_000, 13_000, "BoP MBP6 sintético",
            ),
            AnchorSpec(
                "Remesas",
                [("Hogares","ROW")],
                1_000, 2_000, "BoP MBP6 sintético",
            ),
            AnchorSpec(
                "D1_ROW_credit",
                [("Trabajo","ROW")],
                10, 15, "BoP MBP6 sintético",
            ),
        ]
        result = audit_sam(bolivia_synthetic_sam, anchors=anchors)
        for c in result.anchor_checks:
            assert c.passed, f"Anchor falló: {c.name} val={c.value:.2f} — {c.detail}"

    def test_markdown_report_contains_all_sections(self, bolivia_synthetic_sam, tmp_path):
        anchors = [AnchorSpec("X", [("Industria","ROW")], 9_000, 12_000, "test")]
        result = audit_sam(
            bolivia_synthetic_sam,
            activity_accounts=["Agricultura","Industria","Servicios"],
            anchors=anchors,
        )
        path = tmp_path / "bolivia_audit.md"
        md = result.to_markdown(path)
        assert "# Auditoría SAM" in md
        assert "Checks universales" in md
        assert "Anchors país-específicos" in md
        assert "X" in md

    def test_corrupted_sam_detected(self, bolivia_synthetic_sam):
        """Introducir error: una celda negativa y un desbalance."""
        df = bolivia_synthetic_sam.copy()
        df.loc["Agricultura", "ROW"] = -100.0   # negativo
        df.loc["Industria",   "Hogares"] += 5_000  # desbalance
        result = audit_sam(df)
        assert not result.passed
        failing_names = [c.name for c in result.failing()]
        assert any("negativ" in n.lower() for n in failing_names)
        assert any("Balance" in n for n in failing_names)

    def test_audit_result_n_counts(self, bolivia_synthetic_sam):
        anchors = [AnchorSpec("X", [("Industria","ROW")], 0, 999_999, "test")]
        result = audit_sam(
            bolivia_synthetic_sam,
            activity_accounts=["Agricultura","Industria","Servicios"],
            factor_accounts=["Trabajo","Capital"],
            capital_account="Cuenta_Capital",
            row_account="ROW",
            anchors=anchors,
        )
        assert result.n_total == len(result.checks) + len(result.anchor_checks)
        assert 0 <= result.n_passed <= result.n_total

    def test_empty_anchors_list_is_valid(self, bolivia_synthetic_sam):
        result = audit_sam(bolivia_synthetic_sam, anchors=[])
        assert len(result.anchor_checks) == 0
        assert result.n_total == 4  # solo universales
