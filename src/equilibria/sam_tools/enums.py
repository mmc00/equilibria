"""Enum definitions for SAM tools configuration and execution."""

from __future__ import annotations

from enum import Enum


class SAMFormat(str, Enum):
    """Supported SAM storage formats for workflow input/output."""

    EXCEL = "excel"
    GDX = "gdx"
    IEEM_RAW_EXCEL = "ieem_raw_excel"

    @classmethod
    def from_alias(cls, value: str) -> SAMFormat:
        """Normalize format aliases into one canonical ``SAMFormat`` value."""
        normalized = str(value).strip().lower()
        aliases: dict[str, SAMFormat] = {
            "xlsx": cls.EXCEL,
            "xls": cls.EXCEL,
            "excel": cls.EXCEL,
            "pep_excel": cls.EXCEL,
            "gdx": cls.GDX,
            "ieem_raw_excel": cls.IEEM_RAW_EXCEL,
            "ieem_raw": cls.IEEM_RAW_EXCEL,
            "ieem_excel_raw": cls.IEEM_RAW_EXCEL,
        }
        if normalized not in aliases:
            raise ValueError(f"Unsupported SAM format: {value}")
        return aliases[normalized]


class WorkflowOperation(str, Enum):
    """Operation names accepted in SAM workflow transform steps."""

    SCALE_ALL = "scale_all"
    SCALE_SLICE = "scale_slice"
    AGGREGATE_MAPPING = "aggregate_mapping"
    BALANCE_RAS = "balance_ras"
    NORMALIZE_PEP_ACCOUNTS = "normalize_pep_accounts"
    CREATE_X_BLOCK = "create_x_block"
    CONVERT_EXPORTS_TO_X = "convert_exports_to_x"
    ALIGN_TI_TO_GVT_J = "align_ti_to_gvt_j"
    SHIFT_ROW_SLICE = "shift_row_slice"
    MOVE_CELL = "move_cell"
    MOVE_K_TO_JI = "move_k_to_ji"
    MOVE_L_TO_JI = "move_l_to_ji"
    MOVE_MARGIN_TO_I_MARGIN = "move_margin_to_i_margin"
    MOVE_TX_TO_TI_ON_I = "move_tx_to_ti_on_i"
    PEP_STRUCTURAL_MOVES = "pep_structural_moves"
    REBALANCE_IPFP = "rebalance_ipfp"
    ENFORCE_EXPORT_BALANCE = "enforce_export_balance"


class RASMode(str, Enum):
    """Modes that define RAS balancing targets."""

    ARITHMETIC = "arithmetic"
    GEOMETRIC = "geometric"
    ROW = "row"
    COLUMN = "column"

    @classmethod
    def from_alias(cls, value: str | None) -> RASMode:
        """Normalize RAS mode aliases into a canonical ``RASMode``."""
        normalized = str(value or cls.ARITHMETIC.value).strip().lower()
        aliases: dict[str, RASMode] = {
            "arithmetic": cls.ARITHMETIC,
            "mean": cls.ARITHMETIC,
            "avg": cls.ARITHMETIC,
            "arithmetic_mean": cls.ARITHMETIC,
            "symmetric": cls.ARITHMETIC,
            "geometric": cls.GEOMETRIC,
            "geomean": cls.GEOMETRIC,
            "geometric_mean": cls.GEOMETRIC,
            "row": cls.ROW,
            "rows": cls.ROW,
            "column": cls.COLUMN,
            "col": cls.COLUMN,
            "cols": cls.COLUMN,
            "columns": cls.COLUMN,
        }
        if normalized not in aliases:
            allowed = [m.value for m in cls]
            raise ValueError(f"Unsupported ras_type '{value}'. Allowed: {allowed}")
        return aliases[normalized]


class IPFPTargetMode(str, Enum):
    """Target definitions used by the IPFP rebalancing step."""

    GEOMEAN = "geomean"
    AVERAGE = "average"
    ORIGINAL = "original"

    @classmethod
    def from_alias(cls, value: str | None) -> IPFPTargetMode:
        """Normalize target mode aliases into ``IPFPTargetMode``."""
        normalized = str(value or cls.GEOMEAN.value).strip().lower()
        aliases: dict[str, IPFPTargetMode] = {
            "geomean": cls.GEOMEAN,
            "geometric": cls.GEOMEAN,
            "average": cls.AVERAGE,
            "mean": cls.AVERAGE,
            "original": cls.ORIGINAL,
        }
        if normalized not in aliases:
            allowed = [m.value for m in cls]
            raise ValueError(f"Unsupported target_mode '{value}'. Allowed: {allowed}")
        return aliases[normalized]


class IPFPSupportMode(str, Enum):
    """Support masks accepted by the IPFP rebalancing step."""

    PEP_COMPAT = "pep_compat"
    FULL = "full"

    @classmethod
    def from_alias(cls, value: str | None) -> IPFPSupportMode:
        """Normalize support mode aliases into ``IPFPSupportMode``."""
        normalized = str(value or cls.PEP_COMPAT.value).strip().lower()
        aliases: dict[str, IPFPSupportMode] = {
            "pep_compat": cls.PEP_COMPAT,
            "pep": cls.PEP_COMPAT,
            "full": cls.FULL,
        }
        if normalized not in aliases:
            allowed = [m.value for m in cls]
            raise ValueError(f"Unsupported support mode '{value}'. Allowed: {allowed}")
        return aliases[normalized]
