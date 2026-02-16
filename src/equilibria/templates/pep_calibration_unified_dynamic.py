"""
PEP unified calibration with dynamic set derivation from SAM.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from equilibria.babel.gdx.reader import read_parameter_values
from equilibria.templates.pep_calibration_unified import PEPModelCalibrator
from equilibria.templates.pep_calibration_unified_excel import PEPModelCalibratorExcel
from equilibria.templates.pep_dynamic_sets import derive_dynamic_sets_from_sam


def _norm_label(value: str | None) -> str:
    return (value or "").strip().upper()


def _resolve_accounts(accounts: dict[str, str] | None) -> dict[str, str]:
    base = {
        "gvt": "GVT",
        "row": "ROW",
        "td": "TD",
        "ti": "TI",
        "tm": "TM",
        "tx": "TX",
        "inv": "INV",
        "vstk": "VSTK",
    }
    if not accounts:
        return base
    out = base.copy()
    for k, v in accounts.items():
        key = str(k).strip().lower()
        if key in out:
            out[key] = _norm_label(str(v))
    return out


def _map_account(cat: str, elem: str, cfg: dict[str, str]) -> tuple[str, str]:
    c = _norm_label(cat)
    e = _norm_label(elem)
    if c == "AG":
        if e == cfg["gvt"]:
            return c, "GVT"
        if e == cfg["row"]:
            return c, "ROW"
        if e == cfg["td"]:
            return c, "TD"
        if e == cfg["ti"]:
            return c, "TI"
        if e == cfg["tm"]:
            return c, "TM"
        if e == cfg["tx"]:
            return c, "TX"
    if c == "OTH":
        if e == cfg["inv"]:
            return c, "INV"
        if e == cfg["vstk"]:
            return c, "VSTK"
    return c, e


def remap_dynamic_sam_accounts(
    sam_data: dict[str, Any],
    accounts: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Normalize configurable SAM account labels to canonical PEP labels."""
    cfg = _resolve_accounts(accounts)
    src = sam_data.get("sam_matrix")
    if not isinstance(src, dict) or not src:
        try:
            src = read_parameter_values(sam_data, "SAM")
        except Exception:
            src = {}

    dst: dict[tuple[str, str, str, str], float] = {}
    for (r_cat, r_elem, c_cat, c_elem), val in src.items():
        nr_cat, nr_elem = _map_account(str(r_cat), str(r_elem), cfg)
        nc_cat, nc_elem = _map_account(str(c_cat), str(c_elem), cfg)
        key = (nr_cat, nr_elem, nc_cat, nc_elem)
        dst[key] = dst.get(key, 0.0) + float(val)

    out = dict(sam_data)
    out["sam_matrix"] = dst
    return out


class PEPModelCalibratorDynamic(PEPModelCalibrator):
    """GDX-based unified calibrator with dynamic sets enabled by default."""

    def __init__(self, sam_file: Path | str, val_par_file: Path | str | None = None):
        super().__init__(sam_file=sam_file, val_par_file=val_par_file, dynamic_sets=True)


class PEPModelCalibratorExcelDynamic(PEPModelCalibratorExcel):
    """Excel-based unified calibrator with dynamic sets enabled by default."""

    def __init__(self, sam_file: Path | str, val_par_file: Path | str | None = None):
        super().__init__(sam_file=sam_file, val_par_file=val_par_file, dynamic_sets=True)


class PEPModelCalibratorDynamicSAM(PEPModelCalibrator):
    """
    GDX-based dynamic SAM calibrator with configurable structural account labels.

    Accounts are remapped to canonical PEP labels before calibration:
    gvt,row,td,ti,tm,tx,inv,vstk.
    """

    def __init__(
        self,
        sam_file: Path | str,
        val_par_file: Path | str | None = None,
        accounts: dict[str, str] | None = None,
    ):
        super().__init__(sam_file=sam_file, val_par_file=val_par_file, dynamic_sets=True)
        self.sam_data = remap_dynamic_sam_accounts(self.sam_data, accounts=accounts)
        self._resolved_sets = derive_dynamic_sets_from_sam(self.sam_data)


class PEPModelCalibratorExcelDynamicSAM(PEPModelCalibratorExcel):
    """Excel-based dynamic SAM calibrator with configurable structural account labels."""

    def __init__(
        self,
        sam_file: Path | str,
        val_par_file: Path | str | None = None,
        accounts: dict[str, str] | None = None,
    ):
        super().__init__(sam_file=sam_file, val_par_file=val_par_file, dynamic_sets=True)
        self.sam_data = remap_dynamic_sam_accounts(self.sam_data, accounts=accounts)
        self._resolved_sets = derive_dynamic_sets_from_sam(self.sam_data)
