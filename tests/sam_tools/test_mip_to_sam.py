"""Tests for MIP to SAM conversion."""

from pathlib import Path

import pytest

from equilibria.sam_tools import run_mip_to_sam
from equilibria.sam_tools.mip_raw_excel import MIPRawSAM
from equilibria.sam_tools.mip_to_sam_transforms import (
    create_factor_income_distribution,
    create_government_flows,
    create_household_expenditure,
    create_investment_account,
    create_row_account,
    disaggregate_va_to_factors,
    normalize_mip_accounts,
)
from equilibria.sam_tools.models import Sam


FIXTURES_DIR = Path(__file__).parent / "fixtures"
SIMPLE_MIP = FIXTURES_DIR / "simple_mip.xlsx"


class TestMIPParser:
    """Test MIP Excel parser."""

    def test_load_simple_mip(self):
        """Test loading simple 3x3 MIP."""
        sam = MIPRawSAM.from_mip_excel(SIMPLE_MIP, sheet_name="MIP")

        assert sam is not None
        assert sam.matrix.shape[0] > 0
        assert sam.matrix.shape[1] > 0

        # Should have RAW category
        categories = {cat for cat, _ in sam.row_keys}
        assert "RAW" in categories

    def test_mip_has_va_row(self):
        """Test that VA row is parsed."""
        sam = MIPRawSAM.from_mip_excel(SIMPLE_MIP, sheet_name="MIP")

        # Should have VA-aggregate label
        labels = [elem for _, elem in sam.row_keys]
        assert any("VA" in label and "aggregate" in label for label in labels)

    def test_mip_has_import_row(self):
        """Test that import row is parsed."""
        sam = MIPRawSAM.from_mip_excel(SIMPLE_MIP, sheet_name="MIP")

        # Should have IMP-total label
        labels = [elem for _, elem in sam.row_keys]
        assert any("IMP" in label and "total" in label for label in labels)

    def test_mip_has_final_demand(self):
        """Test that final demand columns are parsed."""
        sam = MIPRawSAM.from_mip_excel(SIMPLE_MIP, sheet_name="MIP")

        labels = [elem for _, elem in sam.col_keys]
        # Should have HH, GOV, INV, EXP
        assert any("HH" in label.upper() for label in labels)
        assert any("GOV" in label.upper() or "GOB" in label.upper() for label in labels)


class TestNormalizeMIPAccounts:
    """Test normalize_mip_accounts transformation."""

    def test_normalize_creates_i_j_categories(self):
        """Test RAW → (I, J) conversion."""
        sam = MIPRawSAM.from_mip_excel(SIMPLE_MIP, sheet_name="MIP")
        result = normalize_mip_accounts(sam, {})

        assert result["normalized_accounts"] > 0
        assert result["commodities"] > 0
        assert result["sectors"] > 0
        assert result["final_demand"] > 0

        # Check row categories exist
        row_categories = {cat for cat, _ in sam.row_keys}
        assert "I" in row_categories  # Commodities
        assert "VA" in row_categories  # VA aggregate

        # Check column categories exist
        col_categories = {cat for cat, _ in sam.col_keys}
        assert "J" in col_categories  # Sectors
        assert "FD" in col_categories  # Final demand

    def test_normalize_preserves_data(self):
        """Test that normalization preserves total flows."""
        sam = MIPRawSAM.from_mip_excel(SIMPLE_MIP, sheet_name="MIP")
        total_before = float(sam.matrix.sum())

        normalize_mip_accounts(sam, {})
        total_after = float(sam.matrix.sum())

        # Total should be preserved (within floating point tolerance)
        assert abs(total_before - total_after) < 1e-6


class TestDisaggregateVA:
    """Test disaggregate_va_to_factors transformation."""

    def test_disaggregate_creates_factors(self):
        """Test VA → L + K with default shares."""
        sam = MIPRawSAM.from_mip_excel(SIMPLE_MIP, sheet_name="MIP")
        normalize_mip_accounts(sam, {})

        result = disaggregate_va_to_factors(sam, {})

        assert "L" in result["factors_created"]
        assert "K" in result["factors_created"]
        assert result["total_va"] > 0

        # Check L and K exist in SAM
        categories = {cat for cat, _ in sam.row_keys}
        assert "L" in categories
        assert "K" in categories

    def test_disaggregate_with_custom_shares(self):
        """Test VA disaggregation with custom shares."""
        sam = MIPRawSAM.from_mip_excel(SIMPLE_MIP, sheet_name="MIP")
        normalize_mip_accounts(sam, {})

        custom_shares = {"L": 0.70, "K": 0.30}
        result = disaggregate_va_to_factors(sam, {"va_factor_shares": custom_shares})

        assert result["shares"] == custom_shares
        assert abs(result["total_va"]) > 1e-6

    def test_shares_must_sum_to_one(self):
        """Test that shares are validated to sum to 1.0."""
        sam = MIPRawSAM.from_mip_excel(SIMPLE_MIP, sheet_name="MIP")
        normalize_mip_accounts(sam, {})

        invalid_shares = {"L": 0.50, "K": 0.40}  # Sums to 0.90, not 1.0

        with pytest.raises(ValueError, match="must sum to 1.0"):
            disaggregate_va_to_factors(sam, {"va_factor_shares": invalid_shares})


class TestFactorIncomeDistribution:
    """Test create_factor_income_distribution transformation."""

    def test_creates_institution_accounts(self):
        """Test that institution accounts are created."""
        sam = MIPRawSAM.from_mip_excel(SIMPLE_MIP, sheet_name="MIP")
        normalize_mip_accounts(sam, {})
        disaggregate_va_to_factors(sam, {})

        result = create_factor_income_distribution(sam, {})

        assert result["total_distributed"] > 0
        assert len(result["institutions_created"]) > 0

        # Check AG category exists
        categories = {cat for cat, _ in sam.row_keys}
        assert "AG" in categories

    def test_income_flows_from_factors(self):
        """Test that income flows from L and K to institutions."""
        sam = MIPRawSAM.from_mip_excel(SIMPLE_MIP, sheet_name="MIP")
        normalize_mip_accounts(sam, {})
        disaggregate_va_to_factors(sam, {})
        create_factor_income_distribution(sam, {})

        df = sam.to_dataframe()

        # Should have AG.hh → L flow
        hh_key = None
        l_key = None

        for cat, elem in sam.row_keys:
            if cat == "AG" and elem == "hh":
                hh_key = (cat, elem)
            if cat == "L":
                l_key = (cat, elem)

        if hh_key and l_key and hh_key in df.index and l_key in df.columns:
            income = float(df.loc[hh_key, l_key])
            assert income > 0  # Households should receive labor income


class TestHouseholdExpenditure:
    """Test create_household_expenditure transformation."""

    def test_converts_hh_final_demand(self):
        """Test I → FD.HH becomes AG.hh → I."""
        sam = MIPRawSAM.from_mip_excel(SIMPLE_MIP, sheet_name="MIP")
        normalize_mip_accounts(sam, {})
        disaggregate_va_to_factors(sam, {})
        create_factor_income_distribution(sam, {})

        result = create_household_expenditure(sam, {})

        assert result["total_expenditure"] > 0


class TestGovernmentFlows:
    """Test create_government_flows transformation."""

    def test_creates_tax_accounts(self):
        """Test that ti and tm accounts are created."""
        sam = MIPRawSAM.from_mip_excel(SIMPLE_MIP, sheet_name="MIP")
        normalize_mip_accounts(sam, {})
        disaggregate_va_to_factors(sam, {})
        create_factor_income_distribution(sam, {})
        create_household_expenditure(sam, {})

        result = create_government_flows(sam, {})

        # Should have some tax revenue
        assert result["total_ti"] >= 0 or result["total_tm"] >= 0

        # Check tax accounts exist
        elements = {elem for _, elem in sam.row_keys}
        assert "ti" in elements or "gvt" in elements


class TestROWAccount:
    """Test create_row_account transformation."""

    def test_creates_row_account(self):
        """Test that ROW account is created for trade."""
        sam = MIPRawSAM.from_mip_excel(SIMPLE_MIP, sheet_name="MIP")
        normalize_mip_accounts(sam, {})
        disaggregate_va_to_factors(sam, {})
        create_factor_income_distribution(sam, {})
        create_household_expenditure(sam, {})
        create_government_flows(sam, {})

        result = create_row_account(sam, {})

        assert result["total_imports"] > 0 or result["total_exports"] > 0

        # Check ROW exists
        elements = {elem for _, elem in sam.row_keys}
        assert "row" in elements


class TestInvestmentAccount:
    """Test create_investment_account transformation."""

    def test_creates_investment_account(self):
        """Test that investment account is created."""
        sam = MIPRawSAM.from_mip_excel(SIMPLE_MIP, sheet_name="MIP")
        normalize_mip_accounts(sam, {})
        disaggregate_va_to_factors(sam, {})
        create_factor_income_distribution(sam, {})
        create_household_expenditure(sam, {})
        create_government_flows(sam, {})
        create_row_account(sam, {})

        result = create_investment_account(sam, {})

        # Should have investment and/or savings
        assert result["total_investment"] >= 0
        assert result["total_savings"] >= 0


class TestFullPipeline:
    """Test complete MIP → SAM conversion pipeline."""

    def test_run_mip_to_sam_with_defaults(self):
        """Test full pipeline with default parameters."""
        result = run_mip_to_sam(SIMPLE_MIP)

        assert result.sam is not None
        assert len(result.steps) > 0
        assert result.output_path is None  # No output specified

        # Verify key accounts exist
        categories = {cat for cat, _ in result.sam.row_keys}
        assert "I" in categories  # Commodities
        assert "J" in categories  # Sectors
        assert "L" in categories  # Labor
        assert "K" in categories  # Capital
        assert "AG" in categories  # Institutions
        assert "X" in categories  # Exports

    def test_run_mip_to_sam_with_custom_shares(self):
        """Test pipeline with custom factor shares."""
        result = run_mip_to_sam(
            SIMPLE_MIP,
            va_factor_shares={"L": 0.70, "K": 0.30},
        )

        assert result.sam is not None

        # Check that L and K exist
        categories = {cat for cat, _ in result.sam.row_keys}
        assert "L" in categories
        assert "K" in categories

    def test_sam_balance(self):
        """Test that final SAM is reasonably balanced."""
        result = run_mip_to_sam(SIMPLE_MIP, ras_max_iter=100)

        # Check balance
        balance_stats = result.steps[-1]["balance"]
        max_diff = balance_stats["max_row_col_abs_diff"]

        # After RAS, should be well balanced
        assert max_diff < 1.0  # Allow some tolerance for test data

    def test_output_export(self, tmp_path):
        """Test that SAM can be exported."""
        output_file = tmp_path / "test_sam.xlsx"

        result = run_mip_to_sam(
            SIMPLE_MIP,
            output_path=output_file,
        )

        assert result.output_path == output_file
        assert output_file.exists()

    def test_report_export(self, tmp_path):
        """Test that transformation report can be exported."""
        report_file = tmp_path / "test_report.json"

        result = run_mip_to_sam(
            SIMPLE_MIP,
            report_path=report_file,
        )

        assert result.report_path == report_file
        assert report_file.exists()

        # Verify report contains steps
        import json

        report = json.loads(report_file.read_text())
        assert "steps" in report
        assert len(report["steps"]) > 0

    def test_transformation_steps_recorded(self):
        """Test that all transformation steps are recorded."""
        result = run_mip_to_sam(SIMPLE_MIP)

        step_names = [step["step"] for step in result.steps]

        # Verify key steps exist
        assert "normalize_mip" in step_names
        assert "disaggregate_va" in step_names
        assert "factor_income" in step_names
        assert "household_expenditure" in step_names
        assert "government" in step_names
        assert "row_account" in step_names
        assert "investment" in step_names
        assert "create_x_block" in step_names
        assert "convert_exports" in step_names
        assert "balance_ras" in step_names

    def test_each_step_has_balance_info(self):
        """Test that each step records balance statistics."""
        result = run_mip_to_sam(SIMPLE_MIP)

        for step in result.steps:
            assert "balance" in step
            assert "total" in step["balance"]
            assert "max_row_col_abs_diff" in step["balance"]
