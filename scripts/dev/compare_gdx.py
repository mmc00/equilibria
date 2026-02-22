#!/usr/bin/env python3
"""
GDX Comparison Tool

Compares original GAMS GDX files with data from Excel files.
Generates a Markdown report in reports/gdx_comparison_report.md

Usage:
    python scripts/dev/compare_gdx.py

Output:
    reports/gdx_comparison_report.md
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from typing import Any

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from equilibria.babel.gdx.reader import read_gdx, read_parameter_values
from equilibria.templates.data.pep import (
    load_pep_sam,
    load_pep_parameters,
    get_sam_indices_mapping,
)


class GDXComparator:
    """Compare GDX files and generate comparison report."""

    TOLERANCE = 0.0001  # 0.01%
    REPORT_PATH = Path("reports/gdx_comparison_report.md")

    def __init__(self, original_dir: Path):
        self.original_dir = original_dir
        self.results: dict[str, Any] = {}

    def compare_all(self) -> dict[str, Any]:
        """Run all comparisons."""
        print("Comparing SAM-V2_0...")
        self.results["SAM-V2_0"] = self.compare_sam()

        print("Comparing VAL_PAR...")
        self.results["VAL_PAR"] = self.compare_val_par()

        return self.results

    def compare_sam(self) -> dict[str, Any]:
        """Compare SAM-V2_0.gdx with Excel data."""
        result = {
            "name": "SAM-V2_0",
            "status": "pending",
            "metadata": {},
            "excel_records": 0,
            "gdx_records": 0,
            "common_records": 0,
            "differences": [],
            "only_in_excel": [],
            "only_in_gdx": [],
        }

        try:
            # 1. Read Excel data
            sam_excel = load_pep_sam(
                self.original_dir / "SAM-V2_0.xls", rdim=2, cdim=2, unique_elements=True
            )
            excel_data = {}
            for indices, value in sam_excel.records:
                key = tuple(indices)
                excel_data[key] = value

            result["excel_records"] = len(excel_data)

            # 2. Get indices mapping from Excel
            indices_mapping = get_sam_indices_mapping(
                self.original_dir / "SAM-V2_0.xls"
            )

            # 3. Read GDX data with indices mapping
            sam_gdx = read_gdx(self.original_dir / "SAM-V2_0.gdx")
            gdx_data = read_parameter_values(
                sam_gdx,
                "SAM",
                rdim=2,
                cdim=2,
                indices_mapping=indices_mapping,
            )

            result["gdx_records"] = len(gdx_data)

            # 3. Compare
            excel_keys = set(excel_data.keys())
            gdx_keys = set(gdx_data.keys())
            common_keys = excel_keys & gdx_keys

            result["common_records"] = len(common_keys)
            result["only_in_excel"] = list(excel_keys - gdx_keys)
            result["only_in_gdx"] = list(gdx_keys - excel_keys)

            # Check value differences
            for key in common_keys:
                excel_val = excel_data[key]
                gdx_val = gdx_data[key]

                if excel_val != 0:
                    diff_pct = abs(excel_val - gdx_val) / abs(excel_val)
                else:
                    diff_pct = abs(gdx_val)

                if diff_pct > self.TOLERANCE:
                    result["differences"].append(
                        {
                            "key": key,
                            "excel": excel_val,
                            "gdx": gdx_val,
                            "diff_pct": diff_pct * 100,
                        }
                    )

            # Determine status
            if (
                len(result["differences"]) == 0
                and len(result["only_in_excel"]) == 0
                and len(result["only_in_gdx"]) == 0
            ):
                result["status"] = "pass"
            elif len(result["differences"]) == 0:
                result["status"] = "partial"
            else:
                result["status"] = "fail"

        except Exception as e:
            result["status"] = "error"
            result["error"] = str(e)

        return result

    def compare_val_par(self) -> dict[str, Any]:
        """Compare VAL_PAR.gdx with Excel data."""
        result = {
            "name": "VAL_PAR",
            "status": "pending",
            "metadata": {},
            "excel_records": 0,
            "gdx_records": 0,
            "common_records": 0,
            "differences": [],
            "only_in_excel": [],
            "only_in_gdx": [],
        }

        try:
            # 1. Read Excel data
            val_excel = load_pep_parameters(self.original_dir / "VAL_PAR.xlsx")
            excel_data = {}

            # Process each sheet
            for sheet_name, df in val_excel.items():
                if sheet_name == "PAR" and not df.empty:
                    # Convert DataFrame to dict
                    for idx, row in df.iterrows():
                        if len(row) >= 2:
                            key = (str(row.iloc[0]),)
                            try:
                                value = float(row.iloc[1])
                                excel_data[key] = value
                            except (ValueError, TypeError):
                                pass

            result["excel_records"] = len(excel_data)

            # 2. Read GDX data
            val_gdx = read_gdx(self.original_dir / "VAL_PAR.gdx")

            # VAL_PAR has multiple parameters
            gdx_data = {}
            for symbol in val_gdx.get("symbols", []):
                if symbol["type"] == 1:  # Parameter
                    param_name = symbol["name"]
                    try:
                        param_data = read_parameter_values(val_gdx, param_name)
                        for key, value in param_data.items():
                            gdx_data[(param_name,) + key] = value
                    except Exception:
                        pass

            result["gdx_records"] = len(gdx_data)

            # 3. Compare (simplified - just count for now)
            result["status"] = "info"
            result["note"] = "VAL_PAR comparison is simplified due to complex structure"

        except Exception as e:
            result["status"] = "error"
            result["error"] = str(e)

        return result

    def generate_report(self) -> str:
        """Generate Markdown report."""
        lines = [
            "# GDX Comparison Report",
            "",
            f"**Generated:** {datetime.now().isoformat()}",
            f"**Tolerance:** {self.TOLERANCE * 100:.2f}%",
            f"**Source:** {self.original_dir}",
            "",
        ]

        # Summary table
        lines.extend(
            [
                "## Summary",
                "",
                "| File | Status | Excel Records | GDX Records | Common | Differences |",
                "|------|--------|---------------|-------------|--------|-------------|",
            ]
        )

        for name, result in self.results.items():
            status = result.get("status", "unknown")
            status_icon = {
                "pass": "✅",
                "fail": "❌",
                "partial": "⚠️",
                "error": "💥",
                "info": "ℹ️",
            }.get(status, "❓")

            lines.append(
                f"| {name} | {status_icon} {status.upper()} | "
                f"{result.get('excel_records', 0)} | "
                f"{result.get('gdx_records', 0)} | "
                f"{result.get('common_records', 0)} | "
                f"{len(result.get('differences', []))} |"
            )

        # Detailed sections
        for name, result in self.results.items():
            lines.extend(self._format_detail_section(result))

        return "\n".join(lines)

    def _format_detail_section(self, result: dict[str, Any]) -> list[str]:
        """Format detailed section for one file."""
        lines = [
            "",
            f"## {result['name']}",
            "",
            f"**Status:** {result.get('status', 'unknown').upper()}",
            "",
        ]

        if "error" in result:
            lines.extend(
                [
                    f"**Error:** {result['error']}",
                    "",
                ]
            )
            return lines

        if "note" in result:
            lines.extend(
                [
                    f"**Note:** {result['note']}",
                    "",
                ]
            )

        # Record counts
        lines.extend(
            [
                "### Record Counts",
                "",
                f"- **Excel:** {result.get('excel_records', 0)} records",
                f"- **GDX:** {result.get('gdx_records', 0)} records",
                f"- **Common:** {result.get('common_records', 0)} records",
                "",
            ]
        )

        # Differences
        if result.get("differences"):
            lines.extend(
                [
                    "### Value Differences (>0.01%)",
                    "",
                    "| Key | Excel | GDX | Diff % |",
                    "|-----|-------|-----|--------|",
                ]
            )

            for diff in result["differences"][:20]:  # Limit to 20
                key_str = str(diff["key"])
                lines.append(
                    f"| {key_str} | {diff['excel']:.4f} | "
                    f"{diff['gdx']:.4f} | {diff['diff_pct']:.2f}% |"
                )

            if len(result["differences"]) > 20:
                lines.append(f"| ... | ... | ... | ... |")
                lines.append(
                    f"\n*... and {len(result['differences']) - 20} more differences*"
                )

            lines.append("")

        # Only in Excel
        if result.get("only_in_excel"):
            lines.extend(
                [
                    "### Records Only in Excel",
                    "",
                    "| Key |",
                    "|-----|",
                ]
            )
            for key in result["only_in_excel"][:10]:
                lines.append(f"| {key} |")
            if len(result["only_in_excel"]) > 10:
                lines.append(f"| ... ({len(result['only_in_excel']) - 10} more) |")
            lines.append("")

        # Only in GDX
        if result.get("only_in_gdx"):
            lines.extend(
                [
                    "### Records Only in GDX",
                    "",
                    "| Key |",
                    "|-----|",
                ]
            )
            for key in result["only_in_gdx"][:10]:
                lines.append(f"| {key} |")
            if len(result["only_in_gdx"]) > 10:
                lines.append(f"| ... ({len(result['only_in_gdx']) - 10} more) |")
            lines.append("")

        return lines

    def save_report(self) -> Path:
        """Save report to file."""
        report = self.generate_report()
        self.REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        self.REPORT_PATH.write_text(report)
        return self.REPORT_PATH


def main():
    """Main entry point."""
    # Determine original data directory
    original_dir = Path(
        "/Users/marmol/proyectos/cge_babel/pep_static_clean/data/original"
    )

    if not original_dir.exists():
        print(f"Error: Directory not found: {original_dir}")
        print("Please ensure the original GDX files are available.")
        sys.exit(1)

    # Run comparison
    comparator = GDXComparator(original_dir)
    comparator.compare_all()

    # Save report
    report_path = comparator.save_report()
    print(f"\nReport saved to: {report_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("COMPARISON SUMMARY")
    print("=" * 60)

    for name, result in comparator.results.items():
        status = result.get("status", "unknown")
        icon = {
            "pass": "✅",
            "fail": "❌",
            "partial": "⚠️",
            "error": "💥",
            "info": "ℹ️",
        }.get(status, "❓")

        print(f"\n{icon} {name}: {status.upper()}")
        print(f"   Excel: {result.get('excel_records', 0)} records")
        print(f"   GDX: {result.get('gdx_records', 0)} records")

        if result.get("differences"):
            print(f"   Differences: {len(result['differences'])}")


if __name__ == "__main__":
    main()
