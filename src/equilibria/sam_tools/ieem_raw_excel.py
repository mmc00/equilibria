"""IEEM raw SAM support and PEP preprocessing operations for SAM pipelines."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from equilibria.sam_tools.models import SAMTransformState

IEEM_GROUP_LABELS: dict[str, str] = {
    "actividades productivas": "activities",
    "bienes y servicios": "commodities",
    "margenes": "margins",
    "factores": "factors",
    "hogares": "households",
    "empresas": "enterprises",
    "gobierno": "government",
    "resto del mundo": "row",
    "ahorro": "savings",
    "inversion": "investment",
    "inversión": "investment",
}

SPECIAL_LABELS = {"S-HH", "S-FIRM", "S-GVT", "S-ROW", "INV", "VSTK"}


@dataclass(frozen=True)
class _IEEMGroup:
    name: str
    start_row: int
    labels: list[str]


def _norm_text(value: Any) -> str:
    return " ".join(str(value).strip().split())


def _norm_text_lower(value: Any) -> str:
    return _norm_text(value).lower()


def _is_numeric(value: Any) -> bool:
    return isinstance(value, (int, float, np.integer, np.floating))


def _detect_groups(raw_df: pd.DataFrame) -> list[_IEEMGroup]:
    if raw_df.shape[1] < 3:
        raise ValueError("IEEM raw SAM must have at least 3 columns")

    group_positions: dict[str, int] = {}
    for i, val in enumerate(raw_df.iloc[:, 1]):
        if pd.isna(val):
            continue
        text = _norm_text_lower(val)
        for label, name in IEEM_GROUP_LABELS.items():
            if label in text and name not in group_positions:
                group_positions[name] = i
                break

    if not group_positions:
        raise ValueError(
            "Could not detect IEEM account groups in column 2. "
            "Verify sheet/options for raw IEEM input."
        )

    sorted_groups = sorted(group_positions.items(), key=lambda item: item[1])
    groups: list[_IEEMGroup] = []
    for idx, (group_name, start_row) in enumerate(sorted_groups):
        end_row = (
            sorted_groups[idx + 1][1]
            if idx + 1 < len(sorted_groups)
            else len(raw_df)
        )
        labels: list[str] = []
        for row in range(start_row, end_row):
            val = raw_df.iat[row, 2]
            if pd.isna(val):
                continue
            label = _norm_text(val)
            if not label or _norm_text_lower(label) == "total":
                continue
            labels.append(label)
        if labels:
            groups.append(_IEEMGroup(name=group_name, start_row=start_row, labels=labels))

    if not groups:
        raise ValueError("Detected IEEM groups but no account labels were extracted")
    return groups


def _extract_raw_sam_matrix(raw_df: pd.DataFrame, groups: list[_IEEMGroup]) -> pd.DataFrame:
    data_start_col = 3
    group_sizes = [len(g.labels) for g in groups]
    n_accounts = int(sum(group_sizes))
    matrix = np.zeros((n_accounts, n_accounts), dtype=float)

    labels: list[str] = []
    for group in groups:
        labels.extend(group.labels)

    row_offset = 0
    for group_r in groups:
        for i, _ in enumerate(group_r.labels):
            row_idx = group_r.start_row + i
            col_offset = 0
            for g_idx, group_c in enumerate(groups):
                for j, _ in enumerate(group_c.labels):
                    col_idx = data_start_col + sum(group_sizes[:g_idx]) + j
                    if row_idx >= raw_df.shape[0] or col_idx >= raw_df.shape[1]:
                        continue
                    value = raw_df.iat[row_idx, col_idx]
                    if pd.isna(value):
                        continue
                    if _is_numeric(value):
                        matrix[row_offset + i, col_offset + j] = float(value)
                col_offset += len(group_c.labels)
        row_offset += len(group_r.labels)

    return pd.DataFrame(matrix, index=labels, columns=labels, dtype=float)


def _load_mapping(mapping_path: Path) -> tuple[dict[str, str], list[str]]:
    mapping_df = pd.read_excel(mapping_path, sheet_name="mapping")
    required = {"original", "aggregated"}
    if not required.issubset(set(mapping_df.columns)):
        raise ValueError(f"Mapping file missing required columns: {sorted(required)}")

    mapping: dict[str, str] = {}
    ordered_aggregated: list[str] = []
    seen: set[str] = set()
    for _, row in mapping_df.iterrows():
        original = _norm_text(row["original"])
        aggregated = _norm_text(row["aggregated"])
        if not original or not aggregated:
            continue
        mapping[_norm_text_lower(original)] = aggregated
        if aggregated not in seen:
            seen.add(aggregated)
            ordered_aggregated.append(aggregated)

    if not mapping:
        raise ValueError(f"Mapping file has no usable rows: {mapping_path}")
    return mapping, ordered_aggregated


def _state_to_dataframe(state: SAMTransformState, expected_category: str) -> pd.DataFrame:
    if not state.row_keys or not state.col_keys:
        raise ValueError("State has no support keys")
    if any(cat != expected_category for cat, _ in state.row_keys):
        raise ValueError(f"State rows must be category '{expected_category}' for this operation")
    if any(cat != expected_category for cat, _ in state.col_keys):
        raise ValueError(f"State cols must be category '{expected_category}' for this operation")
    labels_r = [elem for _, elem in state.row_keys]
    labels_c = [elem for _, elem in state.col_keys]
    return pd.DataFrame(state.matrix.copy(), index=labels_r, columns=labels_c, dtype=float)


def _replace_state_from_dataframe(state: SAMTransformState, df: pd.DataFrame) -> None:
    labels = [_norm_text(label) for label in df.index]
    state.matrix = df.to_numpy(dtype=float)
    state.row_keys = [("RAW", label) for label in labels]
    state.col_keys = [("RAW", label) for label in labels]


def _aggregate_with_mapping(
    matrix_df: pd.DataFrame,
    mapping: dict[str, str],
    ordered_aggregated: list[str],
) -> pd.DataFrame:
    def mapped(label: Any) -> str:
        label_txt = _norm_text(label)
        return mapping.get(_norm_text_lower(label_txt), label_txt)

    renamed = matrix_df.copy()
    renamed.index = [mapped(idx) for idx in matrix_df.index]
    renamed.columns = [mapped(col) for col in matrix_df.columns]

    aggregated = renamed.groupby(level=0).sum()
    aggregated = aggregated.T.groupby(level=0).sum().T

    ordered = [lab for lab in ordered_aggregated if lab in aggregated.index]
    for lab in aggregated.index:
        if lab not in ordered:
            ordered.append(lab)
    return aggregated.reindex(index=ordered, columns=ordered, fill_value=0.0)


def _balance_ras(
    matrix_df: pd.DataFrame,
    *,
    max_iterations: int = 200,
    tolerance: float = 1e-9,
) -> pd.DataFrame:
    sam = matrix_df.values.copy().astype(float)
    row_totals = sam.sum(axis=1)
    col_totals = sam.sum(axis=0)
    target = 0.5 * (row_totals + col_totals)

    for _ in range(max_iterations):
        current_rows = sam.sum(axis=1)
        for i in range(sam.shape[0]):
            if current_rows[i] > 0 and target[i] > 0:
                sam[i, :] *= target[i] / current_rows[i]

        current_cols = sam.sum(axis=0)
        for j in range(sam.shape[1]):
            if current_cols[j] > 0 and target[j] > 0:
                sam[:, j] *= target[j] / current_cols[j]

        max_diff = float(np.max(np.abs(sam.sum(axis=1) - sam.sum(axis=0))))
        if max_diff <= tolerance:
            break

    return pd.DataFrame(sam, index=matrix_df.index, columns=matrix_df.columns)


def _label_to_key(label: str) -> tuple[str, str] | None:
    label_up = _norm_text(label).upper()
    if label_up in SPECIAL_LABELS:
        return None
    if label_up.startswith("A-"):
        return ("J", label_up[2:].lower())
    if label_up.startswith("C-"):
        return ("I", label_up[2:].lower())
    if label_up in {"USK", "SK"}:
        return ("L", label_up.lower())
    if label_up in {"CAP", "LAND"}:
        return ("K", label_up.lower())
    if label_up in {"HRP", "HRR", "HUP", "HUR", "FIRM", "GVT", "ROW", "TD", "TI", "TM", "TX"}:
        return ("AG", label_up.lower())
    if label_up == "MARG":
        return ("MARG", "MARG")
    return None


def _build_pep_key_order(labels: list[str]) -> list[tuple[str, str]]:
    j_keys: list[tuple[str, str]] = []
    i_keys: list[tuple[str, str]] = []
    l_keys: list[tuple[str, str]] = []
    k_keys: list[tuple[str, str]] = []
    ag_keys: list[tuple[str, str]] = []
    has_marg = False

    for label in labels:
        key = _label_to_key(label)
        if key is None:
            continue
        cat, _ = key
        if cat == "J" and key not in j_keys:
            j_keys.append(key)
        elif cat == "I" and key not in i_keys:
            i_keys.append(key)
        elif cat == "L" and key not in l_keys:
            l_keys.append(key)
        elif cat == "K" and key not in k_keys:
            k_keys.append(key)
        elif cat == "AG" and key not in ag_keys:
            ag_keys.append(key)
        elif cat == "MARG":
            has_marg = True

    ag_order = ["ti", "tm", "tx", "td", "hrp", "hrr", "hup", "hur", "firm", "gvt", "row"]
    ag_sorted = [("AG", elem) for elem in ag_order if ("AG", elem) in ag_keys]

    keys: list[tuple[str, str]] = []
    keys.extend(j_keys)
    keys.extend(i_keys)
    keys.extend(l_keys)
    keys.extend(k_keys)
    keys.extend(ag_sorted)
    if has_marg:
        keys.append(("MARG", "MARG"))
    keys.extend([("OTH", "INV"), ("OTH", "VSTK")])
    return keys


def _ensure_key(state: SAMTransformState, key: tuple[str, str]) -> bool:
    if key in state.row_keys:
        return False
    old_n = state.matrix.shape[0]
    new_matrix = np.zeros((old_n + 1, old_n + 1), dtype=float)
    new_matrix[:old_n, :old_n] = state.matrix
    state.matrix = new_matrix
    state.row_keys = state.row_keys + [key]
    state.col_keys = state.col_keys + [key]
    return True


def _add_value(
    matrix: np.ndarray,
    key_index: dict[tuple[str, str], int],
    row_key: tuple[str, str],
    col_key: tuple[str, str],
    value: float,
) -> None:
    if abs(value) <= 1e-14:
        return
    if row_key not in key_index or col_key not in key_index:
        return
    matrix[key_index[row_key], key_index[col_key]] += float(value)


def load_ieem_raw_excel_state(
    input_path: Path,
    *,
    sheet_name: str = "MCS2016",
) -> SAMTransformState:
    """Read an IEEM raw Excel SAM and return a RAW state."""
    raw_df = pd.read_excel(input_path, sheet_name=sheet_name, header=None)
    groups = _detect_groups(raw_df)
    matrix_df = _extract_raw_sam_matrix(raw_df, groups)

    labels = [_norm_text(label) for label in matrix_df.index]
    matrix = matrix_df.to_numpy(dtype=float)
    return SAMTransformState(
        matrix=matrix,
        row_keys=[("RAW", label) for label in labels],
        col_keys=[("RAW", label) for label in labels],
        source_path=input_path,
        source_format="ieem_raw_excel",
        raw_df=None,
        data_start_row=None,
        data_start_col=None,
    )


class SAM:
    """Minimal SAM container for raw-IEEM preprocessing before PEP transforms."""

    def __init__(self, matrix: pd.DataFrame):
        if matrix.shape[0] != matrix.shape[1]:
            raise ValueError("SAM must be square")
        self.matrix = matrix.copy()

    @classmethod
    def from_ieem_excel(cls, path: Path, sheet_name: str = "MCS2016") -> SAM:
        state = load_ieem_raw_excel_state(path, sheet_name=sheet_name)
        matrix_df = _state_to_dataframe(state, expected_category="RAW")
        return cls(matrix_df)

    def aggregate(self, mapping_path: Path) -> SAM:
        mapping, ordered = _load_mapping(mapping_path)
        self.matrix = _aggregate_with_mapping(self.matrix, mapping, ordered)
        return self

    def balance_ras(self, *, tolerance: float = 1e-9, max_iterations: int = 200) -> SAM:
        self.matrix = _balance_ras(
            self.matrix,
            tolerance=tolerance,
            max_iterations=max_iterations,
        )
        return self

    def to_raw_state(self) -> SAMTransformState:
        labels = [_norm_text(label) for label in self.matrix.index]
        return SAMTransformState(
            matrix=self.matrix.to_numpy(dtype=float),
            row_keys=[("RAW", label) for label in labels],
            col_keys=[("RAW", label) for label in labels],
            source_path=Path("<memory>"),
            source_format="raw",
        )


def aggregate_state_with_mapping(state: SAMTransformState, op: dict[str, Any]) -> dict[str, Any]:
    mapping_path = op.get("mapping_path")
    if not mapping_path:
        raise ValueError("aggregate_mapping requires mapping_path")
    mapping, ordered = _load_mapping(Path(mapping_path))
    before_shape = list(state.matrix.shape)
    df = _state_to_dataframe(state, expected_category="RAW")
    aggregated = _aggregate_with_mapping(df, mapping, ordered)
    _replace_state_from_dataframe(state, aggregated)
    return {
        "mapping_path": str(mapping_path),
        "shape_before": before_shape,
        "shape_after": list(state.matrix.shape),
    }


def balance_state_ras(state: SAMTransformState, op: dict[str, Any]) -> dict[str, Any]:
    tol = float(op.get("tol", op.get("tolerance", 1e-9)))
    max_iter = int(op.get("max_iter", 200))
    df = _state_to_dataframe(state, expected_category="RAW")
    before_max = float(np.max(np.abs(df.sum(axis=1) - df.sum(axis=0))))
    balanced = _balance_ras(df, tolerance=tol, max_iterations=max_iter)
    after_max = float(np.max(np.abs(balanced.sum(axis=1) - balanced.sum(axis=0))))
    _replace_state_from_dataframe(state, balanced)
    return {"tol": tol, "max_iter": max_iter, "max_diff_before": before_max, "max_diff_after": after_max}


def normalize_state_to_pep_accounts(state: SAMTransformState, _op: dict[str, Any]) -> dict[str, Any]:
    df = _state_to_dataframe(state, expected_category="RAW")
    labels = [_norm_text(label) for label in df.index]
    pep_keys = _build_pep_key_order(labels)
    key_index = {k: i for i, k in enumerate(pep_keys)}
    pep_matrix = np.zeros((len(pep_keys), len(pep_keys)), dtype=float)

    for r_label in df.index:
        r_key = _label_to_key(_norm_text(r_label))
        if r_key is None:
            continue
        for c_label in df.columns:
            c_key = _label_to_key(_norm_text(c_label))
            if c_key is None:
                continue
            _add_value(pep_matrix, key_index, r_key, c_key, float(df.loc[r_label, c_label]))

    savings_to_agent: dict[str, list[str]] = {
        "S-HH": ["HRP", "HRR", "HUP", "HUR"],
        "S-FIRM": ["FIRM"],
        "S-GVT": ["GVT"],
        "S-ROW": ["ROW"],
    }
    for savings_row, agents in savings_to_agent.items():
        if savings_row not in df.index:
            continue
        for agent in agents:
            if agent not in df.columns:
                continue
            value = float(df.loc[savings_row, agent])
            _add_value(pep_matrix, key_index, ("OTH", "INV"), ("AG", agent.lower()), value)

    if "VSTK" in df.index:
        vstk_total = float(df.loc["VSTK", :].sum())
        _add_value(pep_matrix, key_index, ("OTH", "VSTK"), ("OTH", "INV"), vstk_total)

    for key in pep_keys:
        if key[0] != "I":
            continue
        commodity = key[1]
        c_label = f"C-{commodity.upper()}"
        if c_label in df.index and "INV" in df.columns:
            _add_value(
                pep_matrix,
                key_index,
                ("I", commodity),
                ("OTH", "INV"),
                float(df.loc[c_label, "INV"]),
            )
        if c_label in df.index and "VSTK" in df.columns:
            _add_value(
                pep_matrix,
                key_index,
                ("I", commodity),
                ("OTH", "VSTK"),
                float(df.loc[c_label, "VSTK"]),
            )

    state.matrix = pep_matrix
    state.row_keys = pep_keys
    state.col_keys = pep_keys.copy()
    return {
        "raw_labels": len(labels),
        "pep_accounts": len(pep_keys),
        "commodities": len([k for k in pep_keys if k[0] == "I"]),
    }


def create_x_block(state: SAMTransformState, _op: dict[str, Any]) -> dict[str, Any]:
    commodities = [elem for cat, elem in state.row_keys if cat == "I"]
    added = 0
    for commodity in commodities:
        if _ensure_key(state, ("X", commodity)):
            added += 1
    return {"commodities": len(commodities), "added_x_accounts": added}


def convert_exports_to_x(state: SAMTransformState, _op: dict[str, Any]) -> dict[str, Any]:
    key_index = {k: i for i, k in enumerate(state.row_keys)}
    if ("AG", "row") not in key_index:
        return {"converted_commodities": 0, "total_export_value": 0.0}

    ag_row_col = key_index[("AG", "row")]
    commodities = [elem for cat, elem in state.row_keys if cat == "I"]
    j_rows = [k for k in state.row_keys if k[0] == "J"]

    converted = 0
    total_export = 0.0
    for commodity in commodities:
        i_key = ("I", commodity)
        x_key = ("X", commodity)
        if i_key not in key_index or x_key not in key_index:
            continue
        i_row = key_index[i_key]
        x_row = key_index[x_key]
        export_value = float(state.matrix[i_row, ag_row_col])
        if abs(export_value) <= 1e-14:
            continue

        state.matrix[i_row, ag_row_col] = 0.0
        state.matrix[x_row, ag_row_col] += export_value

        i_col = key_index[i_key]
        x_col = key_index[x_key]
        j_supply: list[tuple[int, float]] = []
        for j_key in j_rows:
            j_idx = key_index[j_key]
            value = float(state.matrix[j_idx, i_col])
            if value > 0:
                j_supply.append((j_idx, value))
        total_supply = float(sum(v for _, v in j_supply))
        if total_supply > 1e-14:
            for j_idx, value in j_supply:
                moved = export_value * (value / total_supply)
                state.matrix[j_idx, i_col] -= moved
                state.matrix[j_idx, x_col] += moved

        converted += 1
        total_export += export_value

    return {"converted_commodities": converted, "total_export_value": total_export}


def align_ti_to_gvt_j(state: SAMTransformState, _op: dict[str, Any]) -> dict[str, Any]:
    key_index = {k: i for i, k in enumerate(state.row_keys)}
    if ("AG", "ti") not in key_index or ("AG", "gvt") not in key_index:
        return {"moved_total": 0.0, "columns": 0}

    ti_row = key_index[("AG", "ti")]
    gvt_row = key_index[("AG", "gvt")]
    moved = 0.0
    cols = 0
    for key in state.col_keys:
        if key[0] != "J":
            continue
        col = key_index[key]
        value = float(state.matrix[ti_row, col])
        if abs(value) <= 1e-14:
            continue
        state.matrix[ti_row, col] = 0.0
        state.matrix[gvt_row, col] += value
        moved += value
        cols += 1
    return {"moved_total": moved, "columns": cols}
