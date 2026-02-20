"""Compatibility checks for strict-gams baseline alignment."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from equilibria.babel.gdx.reader import read_gdx, read_parameter_values
from equilibria.baseline.manifest import (
    compute_state_anchors,
    file_sha256,
    load_baseline_manifest,
)


def _safe_scalar(value: Any) -> Any:
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value


@dataclass
class BaselineCheckResult:
    """One strict-baseline compatibility check result."""

    code: str
    passed: bool
    message: str
    expected: Any = None
    actual: Any = None
    abs_delta: float | None = None
    rel_delta: float | None = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        return {k: _safe_scalar(v) for k, v in data.items()}


@dataclass
class BaselineCompatibilityReport:
    """Top-level strict baseline compatibility report."""

    passed: bool
    gams_slice: str
    results_gdx: str
    rel_tol: float
    manifest_path: str | None
    checks: list[BaselineCheckResult]
    state_anchors: dict[str, float]
    results_anchors: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "gams_slice": self.gams_slice,
            "results_gdx": self.results_gdx,
            "rel_tol": self.rel_tol,
            "manifest_path": self.manifest_path,
            "checks": [c.to_dict() for c in self.checks],
            "state_anchors": {k: _safe_scalar(v) for k, v in self.state_anchors.items()},
            "results_anchors": {k: _safe_scalar(v) for k, v in self.results_anchors.items()},
        }

    def summary(self) -> str:
        failed = [c for c in self.checks if not c.passed]
        if not failed:
            return "baseline compatibility passed"
        codes = ", ".join(c.code for c in failed[:6])
        return f"baseline compatibility failed: {len(failed)} failed checks ({codes})"


_NUM_RE = re.compile(r"[-+]?\d+(?:\.\d+)?(?:[Ee][-+]?\d+)?")
_LAB_RE = re.compile(r"'([^']*)'")


def _gdxdump_records(
    gdxdump_bin: str,
    gdx_path: Path,
    symbol: str,
) -> list[tuple[tuple[str, ...], float]]:
    out = subprocess.check_output(
        [gdxdump_bin, str(gdx_path), f"symb={symbol}"],
        text=True,
        stderr=subprocess.STDOUT,
    )
    records: list[tuple[tuple[str, ...], float]] = []
    for raw in out.splitlines():
        line = raw.strip()
        if not line or line.startswith(("/", "Parameter ", "Set ", "*")):
            continue
        nums = _NUM_RE.findall(line)
        if not nums:
            continue
        labels = tuple(x.lower() for x in _LAB_RE.findall(line))
        value = float(nums[-1])
        records.append((labels, value))
    return records


def _slice_map(
    records: list[tuple[tuple[str, ...], float]],
    gams_slice: str,
) -> dict[tuple[str, ...], float]:
    selected: dict[tuple[str, ...], float] = {}
    slice_name = gams_slice.lower()
    for labels, value in records:
        if not labels:
            selected[()] = value
            continue
        if labels[-1] in {"base", "sim1", "var"}:
            if labels[-1] != slice_name:
                continue
            selected[labels[:-1]] = value
        else:
            selected[labels] = value
    return selected


def _read_symbol_records(
    gdx_path: Path,
    symbol: str,
    gams_slice: str,
    gdxdump_bin: str,
) -> dict[tuple[str, ...], float]:
    gdxdump_path = shutil.which(gdxdump_bin) if Path(gdxdump_bin).name == gdxdump_bin else gdxdump_bin
    records: list[tuple[tuple[str, ...], float]] = []
    if gdxdump_path and Path(gdxdump_path).exists():
        try:
            records = _gdxdump_records(str(gdxdump_path), gdx_path, symbol)
        except Exception:
            records = []

    if not records:
        gdx = read_gdx(gdx_path)
        try:
            values = read_parameter_values(gdx, symbol)
        except Exception:
            return {}
        for raw_key, raw_val in values.items():
            if isinstance(raw_key, tuple):
                labels = tuple(str(k).lower() for k in raw_key)
            elif raw_key == ():
                labels = ()
            else:
                labels = (str(raw_key).lower(),)
            records.append((labels, float(raw_val)))

    return _slice_map(records, gams_slice)


def _extract_results_anchors(
    *,
    results_gdx: Path,
    gams_slice: str,
    gdxdump_bin: str,
) -> dict[str, float]:
    gdp_bp = _read_symbol_records(results_gdx, "valGDP_BP", gams_slice, gdxdump_bin)
    exd = _read_symbol_records(results_gdx, "valEXD", gams_slice, gdxdump_bin)
    im = _read_symbol_records(results_gdx, "valIM", gams_slice, gdxdump_bin)
    cab = _read_symbol_records(results_gdx, "valCAB", gams_slice, gdxdump_bin)

    gdp_bp_scalar = float(gdp_bp.get((), 0.0))
    exd_total = float(sum(exd.values()))
    im_total = float(sum(im.values()))
    cab_scalar = float(cab.get((), 0.0))
    return {
        "GDP_BP": gdp_bp_scalar,
        "EXD_TOTAL": exd_total,
        "IM_TOTAL": im_total,
        "TRADE_BALANCE": exd_total - im_total,
        "CAB": cab_scalar,
    }


def _extract_results_symbol_presence(
    *,
    results_gdx: Path,
    gams_slice: str,
    gdxdump_bin: str,
) -> dict[str, int]:
    """Return record counts for key symbols in the requested Results.gdx slice."""
    symbols = ("valGDP_BP", "valEXD", "valIM", "valCAB")
    return {
        sym: len(_read_symbol_records(results_gdx, sym, gams_slice, gdxdump_bin))
        for sym in symbols
    }


def _compare_float(expected: float, actual: float, rel_tol: float) -> tuple[bool, float, float]:
    abs_delta = abs(actual - expected)
    rel_delta = abs_delta / max(abs(expected), abs(actual), 1.0)
    passed = (abs_delta <= 1e-6) or (rel_delta <= rel_tol)
    return passed, abs_delta, rel_delta


def evaluate_strict_gams_baseline_compatibility(
    *,
    state: Any,
    results_gdx: Path | str,
    gams_slice: str = "sim1",
    manifest_path: Path | str | None = None,
    sam_file: Path | str | None = None,
    val_par_file: Path | str | None = None,
    rel_tol: float = 1e-4,
    gdxdump_bin: str = "/Library/Frameworks/GAMS.framework/Versions/48/Resources/gdxdump",
    require_manifest: bool = False,
) -> BaselineCompatibilityReport:
    """Evaluate whether strict-gams baseline is compatible with current state."""
    checks: list[BaselineCheckResult] = []
    results_path = Path(results_gdx)
    slice_name = str(gams_slice).lower()

    if not results_path.exists():
        checks.append(
            BaselineCheckResult(
                code="BSL000",
                passed=False,
                message="Results.gdx file not found",
                expected="existing file",
                actual=str(results_path),
            )
        )
        return BaselineCompatibilityReport(
            passed=False,
            gams_slice=slice_name,
            results_gdx=str(results_path),
            rel_tol=rel_tol,
            manifest_path=str(manifest_path) if manifest_path else None,
            checks=checks,
            state_anchors=compute_state_anchors(state),
            results_anchors={},
        )

    state_anchors = compute_state_anchors(state)
    results_anchors = _extract_results_anchors(
        results_gdx=results_path,
        gams_slice=slice_name,
        gdxdump_bin=gdxdump_bin,
    )

    # BASE slices are benchmark-compatible only if anchors match calibrated state.
    # Scenario slices (SIM1, ...) are validated against BASE anchors from the same
    # Results.gdx, then checked for own-slice symbol presence.
    if slice_name == "base":
        for key, state_value in state_anchors.items():
            results_value = results_anchors.get(key, 0.0)
            passed, abs_delta, rel_delta = _compare_float(state_value, results_value, rel_tol)
            checks.append(
                BaselineCheckResult(
                    code=f"BSL_ANCHOR_{key}",
                    passed=passed,
                    message=f"State anchor {key} matches Results.gdx BASE slice",
                    expected=state_value,
                    actual=results_value,
                    abs_delta=abs_delta,
                    rel_delta=rel_delta,
                )
            )
    else:
        base_anchors = _extract_results_anchors(
            results_gdx=results_path,
            gams_slice="base",
            gdxdump_bin=gdxdump_bin,
        )
        for key, state_value in state_anchors.items():
            results_value = base_anchors.get(key, 0.0)
            passed, abs_delta, rel_delta = _compare_float(state_value, results_value, rel_tol)
            checks.append(
                BaselineCheckResult(
                    code=f"BSL_BASE_ANCHOR_{key}",
                    passed=passed,
                    message=f"State anchor {key} matches Results.gdx BASE slice",
                    expected=state_value,
                    actual=results_value,
                    abs_delta=abs_delta,
                    rel_delta=rel_delta,
                )
            )

        presence = _extract_results_symbol_presence(
            results_gdx=results_path,
            gams_slice=slice_name,
            gdxdump_bin=gdxdump_bin,
        )
        for symbol, count in presence.items():
            checks.append(
                BaselineCheckResult(
                    code=f"BSL_SLICE_{symbol}",
                    passed=count > 0,
                    message=f"{symbol} has records for Results.gdx slice '{slice_name}'",
                    expected="record count > 0",
                    actual=count,
                )
            )

    manifest_obj = None
    if manifest_path:
        manifest_file = Path(manifest_path)
        if manifest_file.exists():
            try:
                manifest_obj = load_baseline_manifest(manifest_file)
            except Exception as exc:
                checks.append(
                    BaselineCheckResult(
                        code="BSL_MANIFEST_PARSE",
                        passed=False,
                        message=f"Failed parsing manifest JSON: {exc}",
                        actual=str(manifest_file),
                    )
                )
        else:
            checks.append(
                BaselineCheckResult(
                    code="BSL_MANIFEST_MISSING",
                    passed=not require_manifest,
                    message="Manifest file missing",
                    expected=str(manifest_file),
                    actual="missing",
                )
            )
    elif require_manifest:
        checks.append(
            BaselineCheckResult(
                code="BSL_MANIFEST_REQUIRED",
                passed=False,
                message="Manifest is required for strict_gams baseline check",
            )
        )

    if manifest_obj is not None:
        checks.append(
            BaselineCheckResult(
                code="BSL_MANIFEST_SCHEMA",
                passed=manifest_obj.schema_version == "pep_baseline_manifest/v1",
                message="Manifest schema version",
                expected="pep_baseline_manifest/v1",
                actual=manifest_obj.schema_version,
            )
        )
        checks.append(
            BaselineCheckResult(
                code="BSL_MANIFEST_SLICE",
                passed=manifest_obj.gams_slice in ({slice_name, "base"} if slice_name != "base" else {"base"}),
                message="Manifest gams slice is compatible with requested strict_gams slice",
                expected="base or " + slice_name if slice_name != "base" else "base",
                actual=manifest_obj.gams_slice,
            )
        )

        current_results_hash = file_sha256(results_path)
        checks.append(
            BaselineCheckResult(
                code="BSL_RESULTS_HASH",
                passed=manifest_obj.results_gdx_sha256 == current_results_hash,
                message="Results.gdx hash matches manifest",
                expected=manifest_obj.results_gdx_sha256,
                actual=current_results_hash,
            )
        )

        if manifest_obj.sam_sha256 is not None:
            if sam_file is None:
                checks.append(
                    BaselineCheckResult(
                        code="BSL_SAM_HASH_MISSING_INPUT",
                        passed=False,
                        message="Manifest contains SAM hash but no current sam_file was provided",
                        expected=manifest_obj.sam_sha256,
                        actual=None,
                    )
                )
            else:
                sam_path = Path(sam_file)
                current_sam_hash = file_sha256(sam_path) if sam_path.exists() else ""
                checks.append(
                    BaselineCheckResult(
                        code="BSL_SAM_HASH",
                        passed=current_sam_hash == manifest_obj.sam_sha256,
                        message="SAM hash matches manifest",
                        expected=manifest_obj.sam_sha256,
                        actual=current_sam_hash,
                    )
                )

        if manifest_obj.val_par_sha256 is not None:
            if val_par_file is None:
                checks.append(
                    BaselineCheckResult(
                        code="BSL_VALPAR_HASH_MISSING_INPUT",
                        passed=False,
                        message="Manifest contains VAL_PAR hash but no current val_par_file was provided",
                        expected=manifest_obj.val_par_sha256,
                        actual=None,
                    )
                )
            else:
                val_path = Path(val_par_file)
                current_val_hash = file_sha256(val_path) if val_path.exists() else ""
                checks.append(
                    BaselineCheckResult(
                        code="BSL_VALPAR_HASH",
                        passed=current_val_hash == manifest_obj.val_par_sha256,
                        message="VAL_PAR hash matches manifest",
                        expected=manifest_obj.val_par_sha256,
                        actual=current_val_hash,
                    )
                )

        for set_name, expected_size in manifest_obj.set_sizes.items():
            actual_size = len(state.sets.get(set_name, []))
            checks.append(
                BaselineCheckResult(
                    code=f"BSL_SET_{set_name}",
                    passed=actual_size == expected_size,
                    message=f"Set size matches manifest for {set_name}",
                    expected=expected_size,
                    actual=actual_size,
                )
            )

        for key, expected_value in manifest_obj.state_anchors.items():
            actual_value = float(state_anchors.get(key, 0.0))
            passed, abs_delta, rel_delta = _compare_float(expected_value, actual_value, rel_tol)
            checks.append(
                BaselineCheckResult(
                    code=f"BSL_MANIFEST_ANCHOR_{key}",
                    passed=passed,
                    message=f"State anchor {key} matches manifest",
                    expected=expected_value,
                    actual=actual_value,
                    abs_delta=abs_delta,
                    rel_delta=rel_delta,
                )
            )

    passed = all(check.passed for check in checks)
    return BaselineCompatibilityReport(
        passed=passed,
        gams_slice=slice_name,
        results_gdx=str(results_path),
        rel_tol=rel_tol,
        manifest_path=str(manifest_path) if manifest_path else None,
        checks=checks,
        state_anchors=state_anchors,
        results_anchors=results_anchors,
    )


def save_baseline_compatibility_report(
    report: BaselineCompatibilityReport,
    path: Path | str,
) -> None:
    """Persist compatibility report as JSON."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report.to_dict(), indent=2))
