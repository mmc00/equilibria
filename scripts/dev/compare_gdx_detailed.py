"""Comprehensive GDX Value Comparison Tool

Compares original cge_babel GDX files with equilibria-generated GDX files.
Performs 100% value matching on SAM transactions and VAL_PAR parameters.

Usage:
    python scripts/dev/compare_gdx_detailed.py

Output:
    reports/gdx_comparison_report.md - Detailed Markdown report
"""

import json
from pathlib import Path
from typing import Any
import pandas as pd

from equilibria.babel.gdx.reader import (
    read_gdx,
    get_symbol,
    read_parameter_values,
    read_set_elements,
)


class GDXComparator:
    """Compare GDX files with detailed mismatch reporting."""

    def __init__(self, original_dir: Path, equilibria_dir: Path):
        """Initialize comparator with paths to GDX files.

        Args:
            original_dir: Directory containing original cge_babel GDX files
            equilibria_dir: Directory containing equilibria-generated GDX files
        """
        self.original_dir = original_dir
        self.equilibria_dir = equilibria_dir
        self.mismatches = []
        self.matches = []

    def compare_sam(self) -> dict:
        """Compare SAM GDX files value by value.

        Returns:
            Dictionary with comparison statistics
        """
        print("Comparing SAM GDX files...")

        # Read both files
        original = read_gdx(self.original_dir / "SAM-V2_0.gdx")
        equilibria = read_gdx(self.equilibria_dir / "SAM-V2_0.gdx")

        # Get SAM parameter from both
        original_sam = self._extract_sam_data(original)
        equilibria_sam = self._extract_sam_data(equilibria)

        # Normalize account names to lowercase for comparison
        original_norm = {self._normalize_key(k): v for k, v in original_sam.items()}
        equilibria_norm = {self._normalize_key(k): v for k, v in equilibria_sam.items()}

        # Compare
        all_keys = set(original_norm.keys()) | set(equilibria_norm.keys())

        stats = {
            "total_keys": len(all_keys),
            "exact_matches": 0,
            "missing_in_original": 0,
            "missing_in_equilibria": 0,
            "value_mismatches": 0,
            "mismatch_details": [],
        }

        for key in sorted(all_keys):
            orig_val = original_norm.get(key)
            eq_val = equilibria_norm.get(key)

            if orig_val is None:
                stats["missing_in_original"] += 1
                stats["mismatch_details"].append(
                    {
                        "key": key,
                        "type": "missing_in_original",
                        "equilibria_value": eq_val,
                    }
                )
            elif eq_val is None:
                stats["missing_in_equilibria"] += 1
                stats["mismatch_details"].append(
                    {
                        "key": key,
                        "type": "missing_in_equilibria",
                        "original_value": orig_val,
                    }
                )
            elif orig_val != eq_val:
                stats["value_mismatches"] += 1
                stats["mismatch_details"].append(
                    {
                        "key": key,
                        "type": "value_mismatch",
                        "original_value": orig_val,
                        "equilibria_value": eq_val,
                        "difference": eq_val - orig_val,
                        "relative_diff": ((eq_val - orig_val) / orig_val * 100)
                        if orig_val != 0
                        else None,
                    }
                )
            else:
                stats["exact_matches"] += 1

        return stats

    def compare_val_par(self) -> dict:
        """Compare VAL_PAR GDX files parameter by parameter.

        Returns:
            Dictionary with comparison statistics
        """
        print("Comparing VAL_PAR GDX files...")

        # Read both files
        original = read_gdx(self.original_dir / "VAL_PAR.gdx")
        equilibria = read_gdx(self.equilibria_dir / "VAL_PAR.gdx")

        stats = {
            "total_parameters": 0,
            "exact_matches": 0,
            "missing_in_original": 0,
            "missing_in_equilibria": 0,
            "value_mismatches": 0,
            "mismatch_details": [],
        }

        # Get list of parameters from both files
        orig_params = {sym["name"] for sym in original["symbols"]}
        eq_params = {sym["name"] for sym in equilibria["symbols"]}

        all_params = orig_params | eq_params
        stats["total_parameters"] = len(all_params)

        for param_name in sorted(all_params):
            if param_name not in orig_params:
                stats["missing_in_original"] += 1
                stats["mismatch_details"].append(
                    {"parameter": param_name, "type": "missing_in_original"}
                )
            elif param_name not in eq_params:
                stats["missing_in_equilibria"] += 1
                stats["mismatch_details"].append(
                    {"parameter": param_name, "type": "missing_in_equilibria"}
                )
            else:
                # Compare parameter values
                param_stats = self._compare_parameter(original, equilibria, param_name)
                stats["exact_matches"] += param_stats["matches"]
                stats["value_mismatches"] += param_stats["mismatches"]
                stats["mismatch_details"].extend(param_stats["details"])

        return stats

    def _extract_sam_data(self, gdx_data: dict) -> dict:
        """Extract SAM data from GDX into dictionary format.

        Args:
            gdx_data: GDX data from read_gdx()

        Returns:
            Dictionary mapping (from, to) -> value
        """
        sam_dict = {}

        # Get SAM symbol
        sam_sym = get_symbol(gdx_data, "SAM")
        if not sam_sym:
            return sam_dict

        # Try to read SAM values
        try:
            sam_values = read_parameter_values(gdx_data, "SAM")
            if isinstance(sam_values, dict):
                sam_dict = sam_values
            elif isinstance(sam_values, list):
                # Convert list format to dict
                for item in sam_values:
                    if isinstance(item, dict) and "keys" in item and "value" in item:
                        key = tuple(item["keys"])
                        sam_dict[key] = item["value"]
        except Exception as e:
            print(f"Warning: Could not read SAM values: {e}")

        return sam_dict

    def _normalize_key(self, key) -> tuple:
        """Normalize key to lowercase tuple."""
        if isinstance(key, (list, tuple)):
            return tuple(str(k).lower() for k in key)
        return (str(key).lower(),)

    def _compare_parameter(
        self, original: dict, equilibria: dict, param_name: str
    ) -> dict:
        """Compare a single parameter between two GDX files.

        Returns:
            Dictionary with comparison stats for this parameter
        """
        stats = {"matches": 0, "mismatches": 0, "details": []}

        try:
            orig_values = read_parameter_values(original, param_name)
            eq_values = read_parameter_values(equilibria, param_name)

            # Convert to comparable format
            orig_dict = self._values_to_dict(orig_values)
            eq_dict = self._values_to_dict(eq_values)

            # Compare
            all_keys = set(orig_dict.keys()) | set(eq_dict.keys())

            for key in all_keys:
                orig_val = orig_dict.get(key)
                eq_val = eq_dict.get(key)

                if orig_val is None:
                    stats["mismatches"] += 1
                    stats["details"].append(
                        {
                            "parameter": param_name,
                            "key": key,
                            "type": "missing_in_original",
                            "equilibria_value": eq_val,
                        }
                    )
                elif eq_val is None:
                    stats["mismatches"] += 1
                    stats["details"].append(
                        {
                            "parameter": param_name,
                            "key": key,
                            "type": "missing_in_equilibria",
                            "original_value": orig_val,
                        }
                    )
                elif orig_val != eq_val:
                    stats["mismatches"] += 1
                    stats["details"].append(
                        {
                            "parameter": param_name,
                            "key": key,
                            "type": "value_mismatch",
                            "original_value": orig_val,
                            "equilibria_value": eq_val,
                            "difference": eq_val - orig_val,
                        }
                    )
                else:
                    stats["matches"] += 1

        except Exception as e:
            stats["mismatches"] += 1
            stats["details"].append(
                {"parameter": param_name, "type": "read_error", "error": str(e)}
            )

        return stats

    def _values_to_dict(self, values) -> dict:
        """Convert parameter values to dictionary format."""
        result = {}

        if isinstance(values, dict):
            # Already a dictionary
            for k, v in values.items():
                key = self._normalize_key(k)
                result[key] = float(v) if v is not None else 0.0
        elif isinstance(values, list):
            # List of records
            for item in values:
                if isinstance(item, dict):
                    keys = item.get("keys", [])
                    value = item.get("value", 0.0)
                    key = self._normalize_key(keys)
                    result[key] = float(value)

        return result

    def generate_report(self, sam_stats: dict, val_par_stats: dict, output_path: Path):
        """Generate Markdown comparison report.

        Args:
            sam_stats: SAM comparison statistics
            val_par_stats: VAL_PAR comparison statistics
            output_path: Path to write report
        """
        report = []

        # Header
        report.append("# GDX Value Comparison Report")
        report.append("")
        report.append("## Executive Summary")
        report.append("")

        # Overall status
        sam_match_pct = (
            (sam_stats["exact_matches"] / sam_stats["total_keys"] * 100)
            if sam_stats["total_keys"] > 0
            else 0
        )
        val_par_match_pct = (
            (val_par_stats["exact_matches"] / val_par_stats["total_parameters"] * 100)
            if val_par_stats["total_parameters"] > 0
            else 0
        )

        if (
            sam_stats["value_mismatches"] == 0
            and sam_stats["missing_in_original"] == 0
            and sam_stats["missing_in_equilibria"] == 0
        ):
            report.append("✅ **SAM: PASS** - 100% match")
        else:
            report.append(f"❌ **SAM: FAIL** - {sam_match_pct:.2f}% match")

        if (
            val_par_stats["value_mismatches"] == 0
            and val_par_stats["missing_in_original"] == 0
            and val_par_stats["missing_in_equilibria"] == 0
        ):
            report.append("✅ **VAL_PAR: PASS** - 100% match")
        else:
            report.append(f"❌ **VAL_PAR: FAIL** - {val_par_match_pct:.2f}% match")

        report.append("")

        # SAM Comparison
        report.append("## SAM Comparison")
        report.append("")
        report.append(f"- **Total transactions:** {sam_stats['total_keys']}")
        report.append(
            f"- **Exact matches:** {sam_stats['exact_matches']} ({sam_match_pct:.2f}%)"
        )
        report.append(f"- **Missing in original:** {sam_stats['missing_in_original']}")
        report.append(
            f"- **Missing in equilibria:** {sam_stats['missing_in_equilibria']}"
        )
        report.append(f"- **Value mismatches:** {sam_stats['value_mismatches']}")
        report.append("")

        if sam_stats["mismatch_details"]:
            report.append("### SAM Mismatch Details")
            report.append("")
            for detail in sam_stats["mismatch_details"][:50]:  # Limit to first 50
                report.append(self._format_mismatch(detail))
            if len(sam_stats["mismatch_details"]) > 50:
                report.append(
                    f"\n... and {len(sam_stats['mismatch_details']) - 50} more mismatches"
                )
            report.append("")

        # VAL_PAR Comparison
        report.append("## VAL_PAR Comparison")
        report.append("")
        report.append(f"- **Total parameters:** {val_par_stats['total_parameters']}")
        report.append(
            f"- **Exact matches:** {val_par_stats['exact_matches']} ({val_par_match_pct:.2f}%)"
        )
        report.append(
            f"- **Missing in original:** {val_par_stats['missing_in_original']}"
        )
        report.append(
            f"- **Missing in equilibria:** {val_par_stats['missing_in_equilibria']}"
        )
        report.append(f"- **Value mismatches:** {val_par_stats['value_mismatches']}")
        report.append("")

        if val_par_stats["mismatch_details"]:
            report.append("### VAL_PAR Mismatch Details")
            report.append("")
            for detail in val_par_stats["mismatch_details"][:50]:  # Limit to first 50
                report.append(self._format_mismatch(detail))
            if len(val_par_stats["mismatch_details"]) > 50:
                report.append(
                    f"\n... and {len(val_par_stats['mismatch_details']) - 50} more mismatches"
                )
            report.append("")

        # Recommendations
        report.append("## Recommendations")
        report.append("")

        if (
            sam_stats["value_mismatches"] == 0
            and sam_stats["missing_in_original"] == 0
            and sam_stats["missing_in_equilibria"] == 0
            and val_par_stats["value_mismatches"] == 0
            and val_par_stats["missing_in_original"] == 0
            and val_par_stats["missing_in_equilibria"] == 0
        ):
            report.append("✅ **Validation Passed**")
            report.append("")
            report.append(
                "The equilibria-generated GDX files are identical to the original cge_babel files."
            )
            report.append("You can use the equilibria GDX files with GAMS.")
        else:
            report.append("🔧 **Validation Failed**")
            report.append("")
            report.append("The following fixes are needed:")
            report.append("")

            if sam_stats["missing_in_equilibria"] > 0:
                report.append(
                    f"1. **SAM Extraction**: {sam_stats['missing_in_equilibria']} transactions are missing in equilibria"
                )
            if sam_stats["value_mismatches"] > 0:
                report.append(
                    f"2. **SAM Values**: {sam_stats['value_mismatches']} transactions have different values"
                )
            if val_par_stats["missing_in_equilibria"] > 0:
                report.append(
                    f"3. **VAL_PAR**: {val_par_stats['missing_in_equilibria']} parameters are missing"
                )
            if val_par_stats["value_mismatches"] > 0:
                report.append(
                    f"4. **VAL_PAR Values**: {val_par_stats['value_mismatches']} parameter values differ"
                )

        # Write report
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            f.write("\n".join(report))

        print(f"\n✓ Report generated: {output_path}")

    def _format_mismatch(self, detail: dict) -> str:
        """Format a single mismatch detail for the report."""
        mismatch_type = detail.get("type", "unknown")
        key = detail.get("key", detail.get("parameter", "unknown"))

        if mismatch_type == "missing_in_original":
            return f"- **Missing in original**: `{key}` = {detail.get('equilibria_value', 'N/A')}"
        elif mismatch_type == "missing_in_equilibria":
            return f"- **Missing in equilibria**: `{key}` = {detail.get('original_value', 'N/A')}"
        elif mismatch_type == "value_mismatch":
            orig = detail.get("original_value", "N/A")
            eq = detail.get("equilibria_value", "N/A")
            diff = detail.get("difference", "N/A")
            rel_diff = detail.get("relative_diff")

            if rel_diff is not None:
                return f"- **Value mismatch**: `{key}` - original={orig}, equilibria={eq}, diff={diff} ({rel_diff:.2f}%)"
            else:
                return f"- **Value mismatch**: `{key}` - original={orig}, equilibria={eq}, diff={diff}"
        else:
            return f"- **{mismatch_type}**: {json.dumps(detail, indent=2)}"


def main():
    """Run GDX comparison and generate report."""
    print("=" * 70)
    print("GDX VALUE COMPARISON TOOL")
    print("=" * 70)
    print()

    repo_root = Path(__file__).resolve().parents[2]

    # Define paths
    original_dir = repo_root / "src" / "equilibria" / "templates" / "reference" / "pep2" / "data"
    equilibria_dir = repo_root / "src" / "equilibria" / "templates" / "data" / "pep"
    report_path = repo_root / "reports" / "gdx_comparison_report.md"

    # Create comparator
    comparator = GDXComparator(original_dir, equilibria_dir)

    # Run comparisons
    sam_stats = comparator.compare_sam()
    val_par_stats = comparator.compare_val_par()

    # Generate report
    comparator.generate_report(sam_stats, val_par_stats, report_path)

    print()
    print("=" * 70)
    print("COMPARISON COMPLETE")
    print("=" * 70)

    # Print summary
    sam_pass = (
        sam_stats["value_mismatches"] == 0
        and sam_stats["missing_in_original"] == 0
        and sam_stats["missing_in_equilibria"] == 0
    )
    val_par_pass = (
        val_par_stats["value_mismatches"] == 0
        and val_par_stats["missing_in_original"] == 0
        and val_par_stats["missing_in_equilibria"] == 0
    )

    print()
    if sam_pass and val_par_pass:
        print("✅ VALIDATION PASSED - 100% match")
    else:
        print("❌ VALIDATION FAILED - see report for details")
        print(f"   SAM: {sam_stats['exact_matches']}/{sam_stats['total_keys']} matches")
        print(f"   VAL_PAR: Issues found")


if __name__ == "__main__":
    main()
