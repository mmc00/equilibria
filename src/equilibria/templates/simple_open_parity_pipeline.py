"""GAMS parity helpers for the canonical SimpleOpen benchmark."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import struct
from typing import Any

from equilibria.babel.gdx.reader import read_data_sections, read_gdx
from equilibria.templates.simple_open_contract import (
    SimpleOpenContract,
    build_simple_open_contract,
)
from equilibria.templates.simple_open_parity_spec import (
    SimpleOpenParitySpec,
    build_simple_open_parity_spec,
)

_VARIABLE_ORDER = ("VA", "INT", "X", "D", "E", "ER", "PFX", "CAB", "FSAV")
_EQUATION_ORDER = ("EQ_VA", "EQ_INT", "EQ_CET")
_CALIBRATION_ORDER = (
    "alpha_va",
    "rho_va",
    "a_int",
    "b_ext",
    "theta_cet",
    "phi_cet",
    "closure_code",
    "modelstat",
    "solvestat",
)
_SPECIAL_FLOAT_CODES = {
    0x06: 1.0,
    0x08: 0.5,
}


@dataclass(frozen=True)
class SimpleOpenGAMSReference:
    """Canonical symbols loaded from a SimpleOpen benchmark GDX."""

    closure_names: tuple[str, ...]
    benchmark: dict[str, float]
    level: dict[str, float]
    residual: dict[str, float]
    calib: dict[str, float]


@dataclass(frozen=True)
class SimpleOpenParityComparison:
    """Parity result for one SimpleOpen closure against a GAMS benchmark."""

    closure_name: str
    gdx_path: str
    passed: bool
    active_closure_match: bool
    benchmark_compared: int
    benchmark_mismatches: int
    benchmark_max_abs_diff: float
    level_compared: int
    level_mismatches: int
    level_max_abs_diff: float
    residual_compared: int
    residual_mismatches: int
    residual_max_abs: float
    parameter_compared: int
    parameter_mismatches: int
    parameter_max_abs_diff: float
    modelstat: float | None
    solvestat: float | None
    details: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly payload."""

        return asdict(self)

def _read_symbol_section(
    *,
    gdx_path: Path,
    gdx_data: dict[str, Any],
    symbol_name: str,
) -> tuple[dict[str, Any], bytes]:
    symbols = gdx_data.get("symbols", [])
    symbol_index = -1
    symbol_meta: dict[str, Any] | None = None
    for idx, symbol in enumerate(symbols):
        if symbol["name"] == symbol_name:
            symbol_index = idx
            symbol_meta = symbol
            break
    if symbol_index < 0 or symbol_meta is None:
        raise ValueError(f"Symbol '{symbol_name}' not found in GDX data: {gdx_path}")

    sections = read_data_sections(gdx_path.read_bytes())
    if symbol_index >= len(sections):
        raise ValueError(f"No _DATA_ section found for symbol '{symbol_name}' in {gdx_path}")
    _, section = sections[symbol_index]
    return symbol_meta, section


def _decode_ordered_values(section: bytes) -> list[float]:
    values: list[float] = []
    if len(section) < 20:
        return values

    pos = 19
    if pos + 5 <= len(section) and section[pos] == 0x01:
        pos += 5

    while pos < len(section):
        byte = section[pos]
        if byte == 0xFF:
            break
        if byte == 0x02:
            pos += 1
            continue
        if byte == 0x0A and pos + 9 <= len(section):
            values.append(float(struct.unpack_from("<d", section, pos + 1)[0]))
            pos += 9
            continue
        if byte in _SPECIAL_FLOAT_CODES:
            values.append(float(_SPECIAL_FLOAT_CODES[byte]))
            pos += 1
            continue
        pos += 1
    return values


def _decode_named_parameter_symbol(
    *,
    gdx_path: Path,
    gdx_data: dict[str, Any],
    symbol_name: str,
    ordered_names: tuple[str, ...],
    fill_missing_with_zero: bool,
) -> dict[str, float]:
    symbol_meta, section = _read_symbol_section(
        gdx_path=gdx_path,
        gdx_data=gdx_data,
        symbol_name=symbol_name,
    )
    raw_values = _decode_ordered_values(section)
    expected_records = int(symbol_meta.get("records", 0))
    if expected_records and len(raw_values) > expected_records:
        raw_values = raw_values[:expected_records]

    values: dict[str, float] = {}
    for idx, value in enumerate(raw_values[: len(ordered_names)]):
        values[ordered_names[idx]] = float(value)

    if fill_missing_with_zero:
        for name in ordered_names:
            values.setdefault(name, 0.0)
    return values


def load_simple_open_gams_reference(gdx_path: str | Path) -> SimpleOpenGAMSReference:
    """Load the canonical benchmark symbols from a SimpleOpen benchmark GDX."""

    path = Path(gdx_path)
    gdx = read_gdx(path)
    benchmark = _decode_named_parameter_symbol(
        gdx_path=path,
        gdx_data=gdx,
        symbol_name="benchmark",
        ordered_names=_VARIABLE_ORDER,
        fill_missing_with_zero=False,
    )
    level = _decode_named_parameter_symbol(
        gdx_path=path,
        gdx_data=gdx,
        symbol_name="level",
        ordered_names=_VARIABLE_ORDER,
        fill_missing_with_zero=False,
    )
    residual = _decode_named_parameter_symbol(
        gdx_path=path,
        gdx_data=gdx,
        symbol_name="residual",
        ordered_names=_EQUATION_ORDER,
        fill_missing_with_zero=True,
    )
    calib = _decode_named_parameter_symbol(
        gdx_path=path,
        gdx_data=gdx,
        symbol_name="calib",
        ordered_names=_CALIBRATION_ORDER,
        fill_missing_with_zero=False,
    )
    closure_code = int(round(float(calib.get("closure_code", -1)))) if "closure_code" in calib else -1
    closure_names: tuple[str, ...]
    if closure_code == 101:
        closure_names = ("simple_open_default",)
    elif closure_code == 202:
        closure_names = ("flexible_external_balance",)
    else:
        closure_names = ()
    return SimpleOpenGAMSReference(
        closure_names=closure_names,
        benchmark=benchmark,
        level=level,
        residual=residual,
        calib=calib,
    )


def _compare_named_values(
    expected: dict[str, float],
    observed: dict[str, float],
    *,
    abs_tol: float,
) -> tuple[int, int, float, dict[str, dict[str, float | None]]]:
    compared = 0
    mismatches = 0
    max_abs = 0.0
    details: dict[str, dict[str, float | None]] = {}
    for name, expected_value in expected.items():
        if name not in observed:
            mismatches += 1
            details[name] = {
                "expected": float(expected_value),
                "observed": None,
                "abs_diff": None,
            }
            continue
        compared += 1
        observed_value = float(observed[name])
        abs_diff = abs(float(expected_value) - observed_value)
        max_abs = max(max_abs, abs_diff)
        if abs_diff > abs_tol:
            mismatches += 1
            details[name] = {
                "expected": float(expected_value),
                "observed": observed_value,
                "abs_diff": abs_diff,
            }
    return compared, mismatches, max_abs, details


def compare_simple_open_gams_parity(
    *,
    contract: str | dict[str, object] | SimpleOpenContract | None,
    gdx_path: str | Path,
    abs_tol: float = 1e-9,
) -> SimpleOpenParityComparison:
    """Compare one SimpleOpen GAMS benchmark GDX against the canonical Python spec."""

    resolved_contract = build_simple_open_contract(contract)
    spec: SimpleOpenParitySpec = build_simple_open_parity_spec(resolved_contract)
    reference = load_simple_open_gams_reference(gdx_path)

    expected_closure_code = 101.0 if spec.closure_name == "simple_open_default" else 202.0
    expected_params = {
        "alpha_va": float(spec.benchmark_parameters.alpha_va),
        "rho_va": float(spec.benchmark_parameters.rho_va),
        "a_int": float(spec.benchmark_parameters.a_int),
        "b_ext": float(spec.benchmark_parameters.b_ext),
        "theta_cet": float(spec.benchmark_parameters.theta_cet),
        "phi_cet": float(spec.benchmark_parameters.phi_cet),
        "closure_code": expected_closure_code,
    }
    benchmark_compared, benchmark_mismatches, benchmark_max_abs, benchmark_details = _compare_named_values(
        spec.benchmark_levels,
        reference.benchmark,
        abs_tol=float(abs_tol),
    )
    level_compared, level_mismatches, level_max_abs, level_details = _compare_named_values(
        spec.benchmark_levels,
        reference.level,
        abs_tol=float(abs_tol),
    )
    expected_residual = {name: 0.0 for name in spec.equation_names}
    residual_compared, residual_mismatches, residual_max_abs, residual_details = _compare_named_values(
        expected_residual,
        reference.residual,
        abs_tol=float(abs_tol),
    )
    parameter_compared, parameter_mismatches, parameter_max_abs, parameter_details = _compare_named_values(
        expected_params,
        reference.calib,
        abs_tol=float(abs_tol),
    )

    active_closure_match = spec.closure_name in reference.closure_names
    modelstat = reference.calib.get("modelstat")
    solvestat = reference.calib.get("solvestat")
    passed = bool(
        active_closure_match
        and benchmark_mismatches == 0
        and level_mismatches == 0
        and residual_mismatches == 0
        and parameter_mismatches == 0
        and (modelstat in {1.0, 2.0} if modelstat is not None else True)
        and (solvestat in {1.0} if solvestat is not None else True)
    )

    return SimpleOpenParityComparison(
        closure_name=spec.closure_name,
        gdx_path=str(Path(gdx_path)),
        passed=passed,
        active_closure_match=active_closure_match,
        benchmark_compared=benchmark_compared,
        benchmark_mismatches=benchmark_mismatches,
        benchmark_max_abs_diff=benchmark_max_abs,
        level_compared=level_compared,
        level_mismatches=level_mismatches,
        level_max_abs_diff=level_max_abs,
        residual_compared=residual_compared,
        residual_mismatches=residual_mismatches,
        residual_max_abs=residual_max_abs,
        parameter_compared=parameter_compared,
        parameter_mismatches=parameter_mismatches,
        parameter_max_abs_diff=parameter_max_abs,
        modelstat=float(modelstat) if modelstat is not None else None,
        solvestat=float(solvestat) if solvestat is not None else None,
        details={
            "benchmark": benchmark_details,
            "level": level_details,
            "residual": residual_details,
            "parameters": parameter_details,
            "active_closure": list(reference.closure_names),
        },
    )
