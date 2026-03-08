"""Utilities to extract and compare PEP `.L` levels from Equilibria and GAMS.

This module provides:
- `EquilibriaLevelsExtractor`: read `val*`-style levels from Python solution objects
- `GAMSLevelsExtractor`: read `val*` symbols from GAMS `Results.gdx` slices
- `LevelsComparator`: compare both sources with abs/rel tolerances
"""

from __future__ import annotations

import math
import re
import shutil
import subprocess
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from equilibria.babel.gdx.reader import read_gdx, read_parameter_values
from equilibria.simulations.pep_compare import get_solution_value
from equilibria.templates.pep_model_equations import PEPModelVariables

DEFAULT_GDXDUMP_BIN = "/Library/Frameworks/GAMS.framework/Versions/48/Resources/gdxdump"
_SCENARIOS = {"base", "sim1", "var"}
_NUM_RE = re.compile(r"([-+]?\d+(?:\.\d+)?(?:[Ee][-+]?\d+)?)")
_LAB_RE = re.compile(r"'([^']*)'")


def _normalize_key(raw: Any) -> tuple[str, ...]:
    if raw == ():
        return ()
    if isinstance(raw, tuple):
        return tuple(str(x).lower() for x in raw)
    return (str(raw).lower(),)


@dataclass
class LevelsComparisonReport:
    compared: int
    missing_in_equilibria: int
    mismatches: int
    passed: bool
    abs_tol: float
    rel_tol: float
    max_abs_diff: float
    max_rel_diff: float
    rms_abs_diff: float
    top_mismatches: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class EquilibriaLevelsExtractor:
    """Extract `val*` levels from Equilibria solved variables/parameters."""

    def __init__(self, vars_obj: PEPModelVariables, params: dict[str, Any] | None = None) -> None:
        self.vars_obj = vars_obj
        self.params = params or {}

    def get(self, symbol: str, key: tuple[str, ...]) -> float | None:
        return get_solution_value(
            self.vars_obj,
            symbol,
            tuple(str(x).lower() for x in key),
            self.params,
        )

    def extract(
        self,
        symbol_keys: dict[str, Iterable[tuple[str, ...]]],
    ) -> dict[str, dict[tuple[str, ...], float]]:
        out: dict[str, dict[tuple[str, ...], float]] = {}
        for symbol, keys in symbol_keys.items():
            out[symbol] = {}
            for key in keys:
                v = self.get(symbol, key)
                if v is None:
                    continue
                out[symbol][tuple(str(x).lower() for x in key)] = float(v)
        return out


class GAMSLevelsExtractor:
    """Extract `val*` levels from GAMS `Results.gdx` for one scenario slice."""

    def __init__(
        self,
        results_gdx: Path | str,
        *,
        gams_slice: str = "base",
        gdxdump_bin: str = DEFAULT_GDXDUMP_BIN,
    ) -> None:
        self.results_gdx = Path(results_gdx)
        if not self.results_gdx.exists():
            raise FileNotFoundError(f"Results.gdx not found: {self.results_gdx}")
        self.gams_slice = gams_slice.lower()
        self.gdxdump_bin = gdxdump_bin

    @staticmethod
    def slice_records(
        records: list[tuple[tuple[str, ...], float]],
        gams_slice: str,
    ) -> dict[tuple[str, ...], float]:
        wanted = gams_slice.lower()
        out: dict[tuple[str, ...], float] = {}
        for labels, value in records:
            labels_l = tuple(str(x).lower() for x in labels)
            if labels_l and labels_l[-1] in _SCENARIOS:
                if labels_l[-1] != wanted:
                    continue
                out[labels_l[:-1]] = float(value)
                continue
            if wanted == "base":
                out[labels_l] = float(value)
        return out

    def list_val_symbols(self) -> list[str]:
        return [
            s["name"]
            for s in read_gdx(self.results_gdx).get("symbols", [])
            if s["name"].startswith("val")
        ]

    def _resolve_gdxdump_binary(self) -> Path | None:
        raw = str(self.gdxdump_bin).strip()
        if not raw:
            return None

        if "/" in raw or raw.startswith("."):
            p = Path(raw)
            if p.exists():
                return p
            return None

        resolved = shutil.which(raw)
        if resolved:
            return Path(resolved)
        return None

    @staticmethod
    def _gdxdump_records(gdxdump_bin: Path, gdx_file: Path, symbol: str) -> list[tuple[tuple[str, ...], float]]:
        out = subprocess.check_output(
            [str(gdxdump_bin), str(gdx_file), f"symb={symbol}"],
            text=True,
            stderr=subprocess.STDOUT,
        )
        rows: list[tuple[tuple[str, ...], float]] = []
        for raw in out.splitlines():
            line = raw.strip()
            if not line or line.startswith(("/", "Parameter ", "Set ", "*")):
                continue
            nums = _NUM_RE.findall(line)
            if not nums:
                continue
            labels = tuple(x.lower() for x in _LAB_RE.findall(line))
            value = float(nums[-1])
            rows.append((labels, value))
        return rows

    def _records_for_symbol(self, symbol: str) -> list[tuple[tuple[str, ...], float]]:
        gdxdump = self._resolve_gdxdump_binary()
        if gdxdump is not None:
            try:
                return self._gdxdump_records(gdxdump, self.results_gdx, symbol)
            except Exception:
                pass

        gdx = read_gdx(self.results_gdx)
        values = read_parameter_values(gdx, symbol)
        rows: list[tuple[tuple[str, ...], float]] = []
        for raw_key, raw_value in values.items():
            rows.append((_normalize_key(raw_key), float(raw_value)))
        return rows

    def extract(
        self,
        symbols: Iterable[str] | None = None,
    ) -> dict[str, dict[tuple[str, ...], float]]:
        selected = list(symbols) if symbols is not None else self.list_val_symbols()
        out: dict[str, dict[tuple[str, ...], float]] = {}
        for symbol in selected:
            raw_records = self._records_for_symbol(symbol)
            out[symbol] = self.slice_records(raw_records, self.gams_slice)
        return out


class LevelsComparator:
    """Compare GAMS `val*` levels against Equilibria levels."""

    def __init__(self, *, abs_tol: float = 1e-6, rel_tol: float = 1e-6) -> None:
        self.abs_tol = float(abs_tol)
        self.rel_tol = float(rel_tol)

    def compare(
        self,
        gams_levels: dict[str, dict[tuple[str, ...], float]],
        equilibria: EquilibriaLevelsExtractor | dict[str, dict[tuple[str, ...], float]],
    ) -> LevelsComparisonReport:
        compared = 0
        missing = 0
        mismatches: list[dict[str, Any]] = []

        for symbol, records in gams_levels.items():
            for key, gams_value in records.items():
                if isinstance(equilibria, EquilibriaLevelsExtractor):
                    py_value = equilibria.get(symbol, key)
                else:
                    py_value = equilibria.get(symbol, {}).get(key)

                if py_value is None:
                    missing += 1
                    continue

                compared += 1
                abs_diff = abs(float(py_value) - float(gams_value))
                rel_diff = abs_diff / max(abs(float(gams_value)), abs(float(py_value)), 1.0)
                if abs_diff > self.abs_tol and rel_diff > self.rel_tol:
                    mismatches.append(
                        {
                            "symbol": symbol,
                            "key": list(key),
                            "gams": float(gams_value),
                            "equilibria": float(py_value),
                            "abs_diff": abs_diff,
                            "rel_diff": rel_diff,
                        }
                    )

        mismatches.sort(key=lambda x: x["abs_diff"], reverse=True)
        max_abs = max((m["abs_diff"] for m in mismatches), default=0.0)
        max_rel = max((m["rel_diff"] for m in mismatches), default=0.0)
        rms = (
            math.sqrt(sum(m["abs_diff"] ** 2 for m in mismatches) / len(mismatches))
            if mismatches
            else 0.0
        )

        return LevelsComparisonReport(
            compared=compared,
            missing_in_equilibria=missing,
            mismatches=len(mismatches),
            passed=(len(mismatches) == 0 and missing == 0),
            abs_tol=self.abs_tol,
            rel_tol=self.rel_tol,
            max_abs_diff=max_abs,
            max_rel_diff=max_rel,
            rms_abs_diff=rms,
            top_mismatches=mismatches[:30],
        )
