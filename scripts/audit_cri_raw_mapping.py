#!/usr/bin/env python3
"""Audit full raw-CRI SAM account mapping into PEP accounts."""

from __future__ import annotations

import argparse
import json
import sys
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass
class RawAccount:
    idx: int
    raw_group: str
    original: str
    row_total: float
    col_total: float


def _normalize(text: Any) -> str:
    s = " ".join(str(text).split())
    s = "".join(ch for ch in unicodedata.normalize("NFD", s) if unicodedata.category(ch) != "Mn")
    return s.lower()


def _to_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0


def _raw_group_to_mapping_group(raw_group: str) -> str:
    if raw_group in {"activities", "commodities", "factors", "households"}:
        return raw_group
    return "other"


def _pep_target_from_aggregated(aggregated: str) -> tuple[str, str, str]:
    agg = str(aggregated)
    upper = agg.upper()
    if upper.startswith("A-"):
        return ("J", agg[2:].lower(), "direct")
    if upper.startswith("C-"):
        return ("I", agg[2:].lower(), "direct")
    if upper in {"USK", "SK"}:
        return ("L", upper.lower(), "direct")
    if upper in {"CAP", "LAND"}:
        return ("K", upper.lower(), "direct")
    if upper in {"HRP", "HUP", "HRR", "HUR", "FIRM", "GVT", "ROW", "TD", "TI", "TM", "TX"}:
        return ("AG", upper.lower(), "direct")
    if upper == "MARG":
        return ("MARG", "MARG", "direct")
    if upper in {"INV", "S-HH", "S-FIRM", "S-GVT", "S-ROW"}:
        return ("OTH", "INV", "consolidated")
    if upper == "VSTK":
        return ("OTH", "VSTK", "direct")
    return ("UNMAPPED", agg, "unknown")


def _is_tax_like(normalized: str) -> bool:
    return any(t in normalized for t in ("impuesto", "subsidio", " iva ", " arancel "))


def _is_social_like(normalized: str) -> bool:
    return any(
        t in normalized
        for t in (
            "seguridad social",
            "ssoc",
            "contribucion seguro social",
            "constribucion seguro social",
            "transferencias de la seguridad social",
        )
    )


def _load_raw_accounts(raw_sam_path: Path, reader_module_path: Path) -> list[RawAccount]:
    sys.path.insert(0, str(reader_module_path))
    try:
        from sam_reader import SAMReader  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            f"Could not import SAMReader from {reader_module_path}: {exc}"
        ) from exc

    reader = SAMReader(raw_sam_path)
    structure = reader.detect_structure()
    sam_matrix = reader.get_sam_matrix().to_numpy(dtype=float)

    ordered_labels: list[tuple[str, str]] = []
    for grp in structure.get_all_groups():
        for label in grp.labels:
            ordered_labels.append((grp.name, label))

    if sam_matrix.shape[0] != len(ordered_labels) or sam_matrix.shape[1] != len(ordered_labels):
        raise RuntimeError(
            "Raw SAM shape does not match ordered labels: "
            f"matrix={sam_matrix.shape}, labels={len(ordered_labels)}"
        )

    rows: list[RawAccount] = []
    for idx, (raw_group, original) in enumerate(ordered_labels):
        rows.append(
            RawAccount(
                idx=idx,
                raw_group=raw_group,
                original=original,
                row_total=_to_float(sam_matrix[idx, :].sum()),
                col_total=_to_float(sam_matrix[:, idx].sum()),
            )
        )
    return rows


def _match_mapping(
    mapping_df: pd.DataFrame,
    original: str,
    mapping_group: str,
) -> tuple[str, str]:
    norm_original = _normalize(original)
    exact = mapping_df[
        (mapping_df["norm_original"] == norm_original) & (mapping_df["group"] == mapping_group)
    ]
    if len(exact) == 1:
        return (str(exact.iloc[0]["aggregated"]), "exact_group")
    if len(exact) > 1:
        return (str(exact.iloc[0]["aggregated"]), "ambiguous_exact_group")

    fallback = mapping_df[mapping_df["norm_original"] == norm_original]
    if len(fallback) == 1:
        return (str(fallback.iloc[0]["aggregated"]), "fallback_no_group")
    if len(fallback) > 1:
        return (str(fallback.iloc[0]["aggregated"]), "ambiguous_fallback")
    return ("__UNMAPPED__", "missing")


def build_audit_dataframe(
    raw_sam_path: Path,
    mapping_path: Path,
    reader_module_path: Path,
) -> pd.DataFrame:
    raw_accounts = _load_raw_accounts(raw_sam_path, reader_module_path)
    mapping_df = pd.read_excel(mapping_path, sheet_name="mapping")
    mapping_df["group"] = mapping_df["group"].astype(str).str.strip().str.lower()
    mapping_df["norm_original"] = mapping_df["original"].map(_normalize)

    rows: list[dict[str, Any]] = []
    for acc in raw_accounts:
        mapping_group = _raw_group_to_mapping_group(acc.raw_group)
        aggregated, match_status = _match_mapping(mapping_df, acc.original, mapping_group)
        pep_domain, pep_code, pep_status = _pep_target_from_aggregated(aggregated)
        norm_original = _normalize(acc.original)
        rows.append(
            {
                "idx": acc.idx,
                "raw_group": acc.raw_group,
                "mapping_group_expected": mapping_group,
                "original": acc.original,
                "original_norm": norm_original,
                "row_total": acc.row_total,
                "col_total": acc.col_total,
                "abs_flow": abs(acc.row_total),
                "aggregated": aggregated,
                "match_status": match_status,
                "pep_domain": pep_domain,
                "pep_code": pep_code,
                "pep_status": pep_status,
                "is_tax_like": _is_tax_like(f" {norm_original} "),
                "is_social_like": _is_social_like(norm_original),
            }
        )
    return pd.DataFrame(rows)


def build_summary(audit_df: pd.DataFrame) -> dict[str, Any]:
    unmapped = audit_df[audit_df["aggregated"] == "__UNMAPPED__"]
    unknown_pep = audit_df[audit_df["pep_domain"] == "UNMAPPED"]
    consolidated = audit_df[audit_df["pep_status"] == "consolidated"]

    by_agg = (
        audit_df.groupby("aggregated")
        .agg(
            account_count=("original", "count"),
            row_total=("row_total", "sum"),
            abs_flow=("abs_flow", "sum"),
        )
        .sort_values("abs_flow", ascending=False)
    )

    tax_social = audit_df[audit_df["is_tax_like"] | audit_df["is_social_like"]].copy()
    tax_social = tax_social.sort_values(["is_tax_like", "is_social_like", "abs_flow"], ascending=[False, False, False])

    return {
        "raw_account_count": int(len(audit_df)),
        "unmapped_count": int(len(unmapped)),
        "unknown_pep_target_count": int(len(unknown_pep)),
        "consolidated_count": int(len(consolidated)),
        "match_status_counts": audit_df["match_status"].value_counts().to_dict(),
        "pep_status_counts": audit_df["pep_status"].value_counts().to_dict(),
        "top_aggregated_by_abs_flow": by_agg.head(20).reset_index().to_dict(orient="records"),
        "tax_social_count": int(len(tax_social)),
        "tax_social_by_target": (
            tax_social.groupby(["aggregated", "pep_domain", "pep_code"])
            .agg(account_count=("original", "count"), row_total=("row_total", "sum"))
            .reset_index()
            .sort_values("row_total", key=lambda s: s.abs(), ascending=False)
            .to_dict(orient="records")
        ),
    }


def write_markdown_report(
    audit_df: pd.DataFrame,
    summary: dict[str, Any],
    report_path: Path,
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)

    unmapped_df = audit_df[audit_df["aggregated"] == "__UNMAPPED__"]
    tax_social_df = audit_df[audit_df["is_tax_like"] | audit_df["is_social_like"]].copy()
    tax_social_df["kind"] = tax_social_df.apply(
        lambda r: "tax+social"
        if bool(r["is_tax_like"]) and bool(r["is_social_like"])
        else ("tax" if bool(r["is_tax_like"]) else "social"),
        axis=1,
    )
    tax_social_df = tax_social_df.sort_values(["kind", "abs_flow"], ascending=[True, False])

    consolidated_df = audit_df[audit_df["pep_status"] == "consolidated"].copy()
    consolidated_df = consolidated_df.sort_values("abs_flow", ascending=False)

    lines: list[str] = []
    lines.append("# CRI Raw SAM Mapping Audit")
    lines.append("")
    lines.append("## Coverage")
    lines.append(f"- Raw accounts audited: **{summary['raw_account_count']}**")
    lines.append(f"- Unmapped raw accounts: **{summary['unmapped_count']}**")
    lines.append(f"- Unknown PEP targets: **{summary['unknown_pep_target_count']}**")
    lines.append(f"- Consolidated targets (many-to-one): **{summary['consolidated_count']}**")
    lines.append("")

    lines.append("## Match Status")
    for status, count in summary["match_status_counts"].items():
        lines.append(f"- `{status}`: {count}")
    lines.append("")

    lines.append("## Fiscal/Social Accounts")
    lines.append("| kind | raw_group | original | aggregated | pep_target | row_total |")
    lines.append("|---|---|---|---|---|---:|")
    for _, r in tax_social_df.iterrows():
        pep_target = f"{r['pep_domain']}.{r['pep_code']}"
        lines.append(
            f"| {r['kind']} | {r['raw_group']} | {r['original']} | {r['aggregated']} | {pep_target} | {r['row_total']:.6f} |"
        )
    lines.append("")

    lines.append("## Consolidated To OTH.INV")
    lines.append("| raw_group | original | aggregated | pep_target | row_total |")
    lines.append("|---|---|---|---|---:|")
    for _, r in consolidated_df.iterrows():
        pep_target = f"{r['pep_domain']}.{r['pep_code']}"
        lines.append(
            f"| {r['raw_group']} | {r['original']} | {r['aggregated']} | {pep_target} | {r['row_total']:.6f} |"
        )
    lines.append("")

    lines.append("## Unmapped Accounts")
    if unmapped_df.empty:
        lines.append("- None")
    else:
        lines.append("| raw_group | original | row_total |")
        lines.append("|---|---|---:|")
        for _, r in unmapped_df.iterrows():
            lines.append(f"| {r['raw_group']} | {r['original']} | {r['row_total']:.6f} |")
    lines.append("")

    report_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit CRI raw SAM -> PEP mapping coverage")
    parser.add_argument(
        "--raw-sam",
        type=Path,
        default=Path("/Users/marmol/proyectos/cge_babel/sam/cri/2016/data/Matriz_Contabilidad_Social_2016.xlsx"),
    )
    parser.add_argument(
        "--mapping",
        type=Path,
        default=Path("/Users/marmol/proyectos/cge_babel/sam/cri/2016/output/mapping_template.xlsx"),
    )
    parser.add_argument(
        "--reader-module-path",
        type=Path,
        default=Path("/Users/marmol/proyectos/cge_babel/sam/cri/2016/src"),
        help="Directory containing sam_reader.py",
    )
    parser.add_argument(
        "--accounts-csv",
        type=Path,
        default=Path("output/cri_raw_mapping_audit_accounts.csv"),
    )
    parser.add_argument(
        "--tax-social-csv",
        type=Path,
        default=Path("output/cri_raw_mapping_audit_tax_social.csv"),
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=Path("output/cri_raw_mapping_audit_summary.json"),
    )
    parser.add_argument(
        "--report-md",
        type=Path,
        default=Path("output/cri_raw_mapping_audit_report.md"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    audit_df = build_audit_dataframe(args.raw_sam, args.mapping, args.reader_module_path)
    summary = build_summary(audit_df)

    args.accounts_csv.parent.mkdir(parents=True, exist_ok=True)
    audit_df.to_csv(args.accounts_csv, index=False, encoding="utf-8")

    tax_social_df = audit_df[audit_df["is_tax_like"] | audit_df["is_social_like"]].copy()
    tax_social_df.to_csv(args.tax_social_csv, index=False, encoding="utf-8")

    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    write_markdown_report(audit_df, summary, args.report_md)

    print(f"Saved accounts CSV: {args.accounts_csv}")
    print(f"Saved tax/social CSV: {args.tax_social_csv}")
    print(f"Saved summary JSON: {args.summary_json}")
    print(f"Saved markdown report: {args.report_md}")
    print(
        "Coverage "
        f"raw={summary['raw_account_count']} "
        f"unmapped={summary['unmapped_count']} "
        f"unknown_pep={summary['unknown_pep_target_count']}"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

