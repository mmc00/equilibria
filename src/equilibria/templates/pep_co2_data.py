"""Helpers for attaching sector CO2 policy data to the PEP runtime."""

from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd

CO2_STATE_ATTR = "_pep_co2"


def get_state_co2_block(state: Any) -> dict[str, Any]:
    """Return the mutable CO2 block attached to one calibrated state."""
    block = getattr(state, CO2_STATE_ATTR, None)
    if isinstance(block, dict):
        return block
    empty = {"co2_intensity": {}, "tco2b": {}, "tco2scal": 1.0}
    setattr(state, CO2_STATE_ATTR, empty)
    return empty


def set_state_co2_block(state: Any, block: Mapping[str, Any]) -> None:
    """Attach one normalized CO2 block to a calibrated state."""
    payload = {
        "co2_intensity": dict(block.get("co2_intensity", {})),
        "tco2b": dict(block.get("tco2b", {})),
        "tco2scal": float(block.get("tco2scal", 1.0)),
    }
    setattr(state, CO2_STATE_ATTR, payload)


def normalize_co2_inputs(
    *,
    co2_data: Any | None = None,
    co2_intensity: Mapping[str, float] | None = None,
    tco2b: Mapping[str, float] | None = None,
    tco2scal: float = 1.0,
) -> dict[str, Any]:
    """Normalize flexible user inputs into one canonical CO2 block."""
    block = {"co2_intensity": {}, "tco2b": {}, "tco2scal": float(tco2scal)}

    if co2_data is not None:
        if isinstance(co2_data, (str, Path, pd.DataFrame)):
            block.update(_load_co2_table(co2_data))
        elif isinstance(co2_data, Mapping):
            if {"co2_intensity", "tco2b", "tco2scal"} & set(co2_data):
                if "co2_intensity" in co2_data:
                    block["co2_intensity"] = _normalize_numeric_map(
                        co2_data["co2_intensity"],
                        field_name="co2_intensity",
                        allow_negative=False,
                    )
                if "tco2b" in co2_data:
                    block["tco2b"] = _normalize_numeric_map(
                        co2_data["tco2b"],
                        field_name="tco2b",
                        allow_negative=True,
                    )
                if "tco2scal" in co2_data:
                    block["tco2scal"] = _normalize_scalar(
                        co2_data["tco2scal"],
                        field_name="tco2scal",
                    )
            else:
                block["co2_intensity"] = _normalize_numeric_map(
                    co2_data,
                    field_name="co2_intensity",
                    allow_negative=False,
                )
        else:
            raise TypeError(
                "`co2_data` must be a mapping, pandas DataFrame, or path to CSV/XLSX."
            )

    if co2_intensity is not None:
        block["co2_intensity"] = _normalize_numeric_map(
            co2_intensity,
            field_name="co2_intensity",
            allow_negative=False,
        )
    if tco2b is not None:
        block["tco2b"] = _normalize_numeric_map(
            tco2b,
            field_name="tco2b",
            allow_negative=True,
        )
    block["tco2scal"] = _normalize_scalar(block.get("tco2scal", tco2scal), field_name="tco2scal")
    return block


def validate_and_fill_co2_block(block: Mapping[str, Any], sectors: list[str]) -> dict[str, Any]:
    """Validate CO2 data against calibrated sectors and fill missing entries with zero."""
    sector_keys = [str(item).strip().lower() for item in sectors]
    sector_set = set(sector_keys)

    intensity = _normalize_numeric_map(
        block.get("co2_intensity", {}),
        field_name="co2_intensity",
        allow_negative=False,
    )
    tax_base = _normalize_numeric_map(
        block.get("tco2b", {}),
        field_name="tco2b",
        allow_negative=True,
    )
    scale = _normalize_scalar(block.get("tco2scal", 1.0), field_name="tco2scal")

    unknown = sorted((set(intensity) | set(tax_base)) - sector_set)
    if unknown:
        bad = ", ".join(unknown)
        raise ValueError(f"CO2 data includes unknown sectors: {bad}")

    return {
        "co2_intensity": {j: float(intensity.get(j, 0.0)) for j in sector_keys},
        "tco2b": {j: float(tax_base.get(j, 0.0)) for j in sector_keys},
        "tco2scal": float(scale),
    }


def carbon_unit_tax(params: Mapping[str, Any], vars_obj: Any, sector: str) -> float:
    """Return one sector-specific carbon tax wedge per unit of activity."""
    intensity = float(params.get("co2_intensity", {}).get(sector, 0.0))
    tax_base = float(params.get("tco2b", {}).get(sector, 0.0))
    scale = float(params.get("tco2scal", 1.0))
    pixcon = float(getattr(vars_obj, "PIXCON", 1.0))
    return intensity * tax_base * scale * pixcon


def attach_co2_metrics(
    vars_obj: Any,
    params: Mapping[str, Any],
    sectors: list[str],
) -> None:
    """Attach derived CO2 metrics to one solved variable bundle."""
    co2_intensity = {
        j: float(params.get("co2_intensity", {}).get(j, 0.0))
        for j in sectors
    }
    tco2b = {
        j: float(params.get("tco2b", {}).get(j, 0.0))
        for j in sectors
    }
    tco2scal = float(params.get("tco2scal", 1.0))
    emissions = {
        j: co2_intensity[j] * float(getattr(vars_obj, "XST", {}).get(j, 0.0))
        for j in sectors
    }
    unit_tax = {
        j: carbon_unit_tax(params, vars_obj, j)
        for j in sectors
    }
    tax_revenue = {
        j: unit_tax[j] * float(getattr(vars_obj, "XST", {}).get(j, 0.0))
        for j in sectors
    }

    setattr(vars_obj, "co2_intensity", co2_intensity)
    setattr(vars_obj, "tco2b", tco2b)
    setattr(vars_obj, "tco2scal", tco2scal)
    setattr(vars_obj, "co2_emissions", emissions)
    setattr(vars_obj, "co2_unit_tax", unit_tax)
    setattr(vars_obj, "co2_tax_revenue", tax_revenue)
    setattr(vars_obj, "co2_total_emissions", float(sum(emissions.values())))
    setattr(vars_obj, "co2_total_tax", float(sum(tax_revenue.values())))


def _load_co2_table(source: str | Path | pd.DataFrame) -> dict[str, Any]:
    if isinstance(source, pd.DataFrame):
        frame = source.copy()
    else:
        path = Path(source)
        suffix = path.suffix.lower()
        if suffix == ".csv":
            frame = pd.read_csv(path)
        elif suffix in {".xlsx", ".xls"}:
            frame = pd.read_excel(path)
        else:
            raise ValueError(
                f"Unsupported CO2 data file extension {suffix!r}. Use CSV or Excel."
            )

    columns = {str(col).strip().lower(): str(col) for col in frame.columns}
    if "sector" not in columns or "co2_intensity" not in columns:
        raise ValueError(
            "CO2 input table must include `sector` and `co2_intensity` columns."
        )

    sector_col = columns["sector"]
    intensity_col = columns["co2_intensity"]
    tco2b_col = columns.get("tco2b")
    tco2scal_col = columns.get("tco2scal")

    intensity: dict[str, float] = {}
    tax_base: dict[str, float] = {}
    for row in frame.itertuples(index=False):
        raw_sector = getattr(row, sector_col)
        if pd.isna(raw_sector):
            continue
        sector = str(raw_sector).strip().lower()
        if not sector:
            continue

        raw_intensity = getattr(row, intensity_col)
        intensity[sector] = _normalize_scalar(
            raw_intensity,
            field_name=f"co2_intensity[{sector}]",
            allow_negative=False,
        )

        if tco2b_col is not None:
            raw_tco2b = getattr(row, tco2b_col)
            if not pd.isna(raw_tco2b):
                tax_base[sector] = _normalize_scalar(
                    raw_tco2b,
                    field_name=f"tco2b[{sector}]",
                    allow_negative=True,
                )

    scale = 1.0
    if tco2scal_col is not None:
        values = [
            _normalize_scalar(v, field_name="tco2scal", allow_negative=True)
            for v in frame[tco2scal_col].tolist()
            if not pd.isna(v)
        ]
        if values:
            first = values[0]
            if any(abs(v - first) > 1e-12 for v in values[1:]):
                raise ValueError("`tco2scal` column must contain one constant value.")
            scale = first

    return {
        "co2_intensity": intensity,
        "tco2b": tax_base,
        "tco2scal": scale,
    }


def _normalize_numeric_map(
    raw: Any,
    *,
    field_name: str,
    allow_negative: bool,
) -> dict[str, float]:
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise TypeError(f"`{field_name}` must be a mapping of sector -> value.")

    out: dict[str, float] = {}
    for key, value in raw.items():
        sector = str(key).strip().lower()
        if not sector:
            raise ValueError(f"`{field_name}` includes an empty sector label.")
        out[sector] = _normalize_scalar(
            value,
            field_name=f"{field_name}[{sector}]",
            allow_negative=allow_negative,
        )
    return out


def _normalize_scalar(
    raw: Any,
    *,
    field_name: str,
    allow_negative: bool = True,
) -> float:
    value = float(raw)
    if not math.isfinite(value):
        raise ValueError(f"`{field_name}` must be finite.")
    if not allow_negative and value < 0.0:
        raise ValueError(f"`{field_name}` must be non-negative.")
    return value
