"""SAM QA checks for pre-calibration structural validation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from equilibria.babel.gdx.reader import read_gdx, read_parameter_values
from equilibria.qa.reporting import SAMQACheckResult, SAMQAReport
from equilibria.qa.sam_contracts import SAMContractSpec, default_sam_contracts
from equilibria.templates.data.pep import load_pep_sam
from equilibria.templates.pep_calibration_unified_dynamic import (
    remap_dynamic_sam_accounts,
)
from equilibria.templates.pep_dynamic_sets import derive_dynamic_sets_from_sam

SAMKey = tuple[str, str, str, str]


def _to_upper(value: str) -> str:
    return str(value).strip().upper()


def _normalize_sam_matrix(source: dict[tuple[Any, ...], Any]) -> dict[SAMKey, float]:
    matrix: dict[SAMKey, float] = {}
    for raw_key, raw_value in source.items():
        if len(raw_key) != 4:
            continue
        key: SAMKey = (
            _to_upper(raw_key[0]),
            _to_upper(raw_key[1]),
            _to_upper(raw_key[2]),
            _to_upper(raw_key[3]),
        )
        matrix[key] = matrix.get(key, 0.0) + float(raw_value)
    return matrix


def _build_sam_data_from_excel(filepath: Path | str) -> dict[str, Any]:
    """Convert SAM Excel into internal dict with a normalized sam_matrix."""
    sam_path = Path(filepath)
    sam4d = load_pep_sam(
        sam_path,
        rdim=2,
        cdim=2,
        sparse=True,
        unique_elements=False,
    )

    sam_matrix: dict[SAMKey, float] = {}
    records: list[dict[str, Any]] = []
    for keys, value in sam4d.to_gdx_records():
        if len(keys) != 4:
            continue
        key: SAMKey = tuple(_to_upper(k) for k in keys)  # type: ignore[assignment]
        val = float(value)
        sam_matrix[key] = sam_matrix.get(key, 0.0) + val
        records.append({"indices": list(key), "value": val})

    return {
        "filepath": str(sam_path),
        "source": "excel",
        "sam_matrix": sam_matrix,
        "symbols": [{"name": "SAM", "records": records}],
        "elements": [],
    }


def load_sam_data(sam_file: Path | str) -> dict[str, Any]:
    """Load SAM from `.gdx` or Excel and return a SAM data dict."""
    sam_path = Path(sam_file)
    suffix = sam_path.suffix.lower()
    if suffix in {".xls", ".xlsx"}:
        return _build_sam_data_from_excel(sam_path)
    return read_gdx(sam_path)


def _extract_sam_matrix(sam_data: dict[str, Any]) -> dict[SAMKey, float]:
    sam_matrix = sam_data.get("sam_matrix")
    if isinstance(sam_matrix, dict):
        return _normalize_sam_matrix(sam_matrix)
    return _normalize_sam_matrix(read_parameter_values(sam_data, "SAM"))


def _sam_value(
    matrix: dict[SAMKey, float],
    row_cat: str,
    row_elem: str,
    col_cat: str,
    col_elem: str,
) -> float:
    return float(
        matrix.get(
            (
                _to_upper(row_cat),
                _to_upper(row_elem),
                _to_upper(col_cat),
                _to_upper(col_elem),
            ),
            0.0,
        )
    )


def _eq_delta(lhs: float, rhs: float) -> tuple[float, float]:
    abs_delta = abs(lhs - rhs)
    rel_delta = abs_delta / max(abs(lhs), abs(rhs), 1.0)
    return abs_delta, rel_delta


def _eq_pass(lhs: float, rhs: float, abs_tol: float, rel_tol: float) -> tuple[bool, float, float]:
    abs_delta, rel_delta = _eq_delta(lhs, rhs)
    return (abs_delta <= abs_tol) or (rel_delta <= rel_tol), abs_delta, rel_delta


def _build_check_result(
    spec: SAMContractSpec,
    *,
    evaluated: int,
    failures: list[dict[str, Any]],
    max_samples: int,
) -> SAMQACheckResult:
    top_failures = sorted(
        failures,
        key=lambda row: max(abs(float(row.get("abs_delta", 0.0))), abs(float(row.get("rel_delta", 0.0)))),
        reverse=True,
    )[:max_samples]
    max_abs_delta = max((float(row.get("abs_delta", 0.0)) for row in failures), default=0.0)
    max_rel_delta = max((float(row.get("rel_delta", 0.0)) for row in failures), default=0.0)
    return SAMQACheckResult(
        code=spec.code,
        title=spec.title,
        category=spec.category,
        description=spec.description,
        severity=spec.severity,
        passed=len(failures) == 0,
        evaluated=evaluated,
        failures=len(failures),
        max_abs_delta=max_abs_delta,
        max_rel_delta=max_rel_delta,
        abs_tol=spec.abs_tol,
        rel_tol=spec.rel_tol,
        samples=top_failures,
    )


def _check_export_value_balance(
    matrix: dict[SAMKey, float],
    sets: dict[str, list[str]],
    spec: SAMContractSpec,
    max_samples: int,
) -> SAMQACheckResult:
    i_set = sets.get("I", [])
    j_set = sets.get("J", [])
    failures: list[dict[str, Any]] = []
    evaluated = 0
    for i in i_set:
        i_upper = _to_upper(i)
        lhs = _sam_value(matrix, "X", i_upper, "AG", "ROW")
        rhs = (
            sum(_sam_value(matrix, "J", _to_upper(j), "X", i_upper) for j in j_set)
            + sum(_sam_value(matrix, "I", _to_upper(ij), "X", i_upper) for ij in i_set)
            + _sam_value(matrix, "AG", "GVT", "X", i_upper)
        )
        evaluated += 1
        passed, abs_delta, rel_delta = _eq_pass(lhs, rhs, spec.abs_tol, spec.rel_tol)
        if not passed:
            failures.append(
                {
                    "i": i,
                    "lhs_exdo": lhs,
                    "rhs_exo_margin_tax": rhs,
                    "abs_delta": abs_delta,
                    "rel_delta": rel_delta,
                }
            )
    return _build_check_result(spec, evaluated=evaluated, failures=failures, max_samples=max_samples)


def _check_commodity_account_balance(
    matrix: dict[SAMKey, float],
    sets: dict[str, list[str]],
    spec: SAMContractSpec,
    max_samples: int,
) -> SAMQACheckResult:
    i_set = sets.get("I", [])
    failures: list[dict[str, Any]] = []
    evaluated = 0
    for i in i_set:
        i_upper = _to_upper(i)
        lhs_row_total = sum(
            value for (row_cat, row_elem, _col_cat, _col_elem), value in matrix.items()
            if row_cat == "I" and row_elem == i_upper
        )
        rhs_col_total = sum(
            value for (_row_cat, _row_elem, col_cat, col_elem), value in matrix.items()
            if col_cat == "I" and col_elem == i_upper
        )
        evaluated += 1
        passed, abs_delta, rel_delta = _eq_pass(
            lhs_row_total,
            rhs_col_total,
            spec.abs_tol,
            spec.rel_tol,
        )
        if not passed:
            failures.append(
                {
                    "i": i,
                    "lhs_row_total": lhs_row_total,
                    "rhs_col_total": rhs_col_total,
                    "abs_delta": abs_delta,
                    "rel_delta": rel_delta,
                }
            )
    return _build_check_result(spec, evaluated=evaluated, failures=failures, max_samples=max_samples)


def _check_margin_domestic_denominator(
    matrix: dict[SAMKey, float],
    sets: dict[str, list[str]],
    spec: SAMContractSpec,
    max_samples: int,
) -> SAMQACheckResult:
    i_set = sets.get("I", [])
    j_set = sets.get("J", [])
    failures: list[dict[str, Any]] = []
    evaluated = 0
    for ij in i_set:
        ij_upper = _to_upper(ij)
        raw_margin_total = sum(
            _sam_value(matrix, "I", _to_upper(i), "I", ij_upper) for i in i_set
        )
        if abs(raw_margin_total) <= spec.abs_tol:
            continue
        ddo = sum(_sam_value(matrix, "J", _to_upper(j), "I", ij_upper) for j in j_set)
        imo = _sam_value(matrix, "AG", "ROW", "I", ij_upper)
        denom = ddo + imo
        evaluated += 1
        if denom <= spec.abs_tol:
            failures.append(
                {
                    "ij": ij,
                    "raw_margin_total": raw_margin_total,
                    "denominator_ddo_plus_imo": denom,
                    "abs_delta": abs(denom),
                    "rel_delta": 1.0,
                }
            )
    return _build_check_result(spec, evaluated=evaluated, failures=failures, max_samples=max_samples)


def _check_margin_export_denominator(
    matrix: dict[SAMKey, float],
    sets: dict[str, list[str]],
    spec: SAMContractSpec,
    max_samples: int,
) -> SAMQACheckResult:
    i_set = sets.get("I", [])
    j_set = sets.get("J", [])
    failures: list[dict[str, Any]] = []
    evaluated = 0
    for ij in i_set:
        ij_upper = _to_upper(ij)
        raw_margin_x_total = sum(
            _sam_value(matrix, "I", _to_upper(i), "X", ij_upper) for i in i_set
        )
        if abs(raw_margin_x_total) <= spec.abs_tol:
            continue
        sum_exo = sum(_sam_value(matrix, "J", _to_upper(j), "X", ij_upper) for j in j_set)
        evaluated += 1
        if sum_exo <= spec.abs_tol:
            failures.append(
                {
                    "ij": ij,
                    "raw_export_margin_total": raw_margin_x_total,
                    "denominator_sum_exo": sum_exo,
                    "abs_delta": abs(sum_exo),
                    "rel_delta": 1.0,
                }
            )
    return _build_check_result(spec, evaluated=evaluated, failures=failures, max_samples=max_samples)


def _check_tax_import_base(
    matrix: dict[SAMKey, float],
    sets: dict[str, list[str]],
    spec: SAMContractSpec,
    max_samples: int,
) -> SAMQACheckResult:
    i_set = sets.get("I", [])
    failures: list[dict[str, Any]] = []
    evaluated = 0
    for i in i_set:
        i_upper = _to_upper(i)
        tax = _sam_value(matrix, "AG", "TM", "I", i_upper)
        if abs(tax) <= spec.abs_tol:
            continue
        base = _sam_value(matrix, "AG", "ROW", "I", i_upper)
        evaluated += 1
        if base <= spec.abs_tol:
            failures.append(
                {
                    "i": i,
                    "tax_timo": tax,
                    "base_imo": base,
                    "abs_delta": abs(base),
                    "rel_delta": 1.0,
                }
            )
    return _build_check_result(spec, evaluated=evaluated, failures=failures, max_samples=max_samples)


def _check_tax_export_base(
    matrix: dict[SAMKey, float],
    sets: dict[str, list[str]],
    spec: SAMContractSpec,
    max_samples: int,
) -> SAMQACheckResult:
    i_set = sets.get("I", [])
    failures: list[dict[str, Any]] = []
    evaluated = 0
    for i in i_set:
        i_upper = _to_upper(i)
        tax = _sam_value(matrix, "AG", "GVT", "X", i_upper)
        if abs(tax) <= spec.abs_tol:
            continue
        base = _sam_value(matrix, "X", i_upper, "AG", "ROW") - tax
        evaluated += 1
        if base <= spec.abs_tol:
            failures.append(
                {
                    "i": i,
                    "tax_tixo": tax,
                    "base_exdo_minus_tixo": base,
                    "abs_delta": abs(base),
                    "rel_delta": 1.0,
                }
            )
    return _build_check_result(spec, evaluated=evaluated, failures=failures, max_samples=max_samples)


def _check_tax_commodity_base(
    matrix: dict[SAMKey, float],
    sets: dict[str, list[str]],
    spec: SAMContractSpec,
    max_samples: int,
) -> SAMQACheckResult:
    i_set = sets.get("I", [])
    j_set = sets.get("J", [])
    failures: list[dict[str, Any]] = []
    evaluated = 0
    for i in i_set:
        i_upper = _to_upper(i)
        tax = _sam_value(matrix, "AG", "TI", "I", i_upper)
        if abs(tax) <= spec.abs_tol:
            continue
        ddo = sum(_sam_value(matrix, "J", _to_upper(j), "I", i_upper) for j in j_set)
        imo = _sam_value(matrix, "AG", "ROW", "I", i_upper)
        margin_sum = sum(_sam_value(matrix, "I", _to_upper(ij), "I", i_upper) for ij in i_set)
        timo = _sam_value(matrix, "AG", "TM", "I", i_upper)
        base = (1.0 + margin_sum) * (ddo + imo) + timo
        evaluated += 1
        if base <= spec.abs_tol:
            failures.append(
                {
                    "i": i,
                    "tax_tico": tax,
                    "base_eq40_like": base,
                    "ddo": ddo,
                    "imo": imo,
                    "margin_sum": margin_sum,
                    "timo": timo,
                    "abs_delta": abs(base),
                    "rel_delta": 1.0,
                }
            )
    return _build_check_result(spec, evaluated=evaluated, failures=failures, max_samples=max_samples)


def _check_tax_production_base(
    matrix: dict[SAMKey, float],
    sets: dict[str, list[str]],
    spec: SAMContractSpec,
    max_samples: int,
) -> SAMQACheckResult:
    j_set = sets.get("J", [])
    i_set = sets.get("I", [])
    l_set = sets.get("L", [])
    k_set = sets.get("K", [])
    failures: list[dict[str, Any]] = []
    evaluated = 0
    for j in j_set:
        j_upper = _to_upper(j)
        tax = _sam_value(matrix, "AG", "GVT", "J", j_upper)
        if abs(tax) <= spec.abs_tol:
            continue
        vao_proxy = sum(
            _sam_value(matrix, "L", _to_upper(labor), "J", j_upper)
            for labor in l_set
        ) + sum(
            _sam_value(matrix, "K", _to_upper(k), "J", j_upper) for k in k_set
        )
        cio = sum(_sam_value(matrix, "I", _to_upper(i), "J", j_upper) for i in i_set)
        base = vao_proxy + cio
        evaluated += 1
        if base <= spec.abs_tol:
            failures.append(
                {
                    "j": j,
                    "tax_tipo": tax,
                    "base_vao_plus_cio": base,
                    "vao_proxy": vao_proxy,
                    "cio": cio,
                    "abs_delta": abs(base),
                    "rel_delta": 1.0,
                }
            )
    return _build_check_result(spec, evaluated=evaluated, failures=failures, max_samples=max_samples)


def _check_macro_savings_investment_balance(
    matrix: dict[SAMKey, float],
    sets: dict[str, list[str]],
    spec: SAMContractSpec,
    max_samples: int,
) -> SAMQACheckResult:
    i_set = sets.get("I", [])
    ag_set = sets.get("AG", [])
    savings_total = sum(_sam_value(matrix, "OTH", "INV", "AG", _to_upper(ag)) for ag in ag_set)
    investment_total = sum(
        _sam_value(matrix, "I", _to_upper(i), "OTH", "INV")
        + _sam_value(matrix, "I", _to_upper(i), "OTH", "VSTK")
        for i in i_set
    )
    passed, abs_delta, rel_delta = _eq_pass(
        savings_total,
        investment_total,
        spec.abs_tol,
        spec.rel_tol,
    )
    failures: list[dict[str, Any]] = []
    if not passed:
        failures.append(
            {
                "lhs_savings_total": savings_total,
                "rhs_investment_total": investment_total,
                "abs_delta": abs_delta,
                "rel_delta": rel_delta,
            }
        )
    return _build_check_result(spec, evaluated=1, failures=failures, max_samples=max_samples)


def _check_macro_gdp_proxy_closure(
    matrix: dict[SAMKey, float],
    sets: dict[str, list[str]],
    spec: SAMContractSpec,
    max_samples: int,
) -> SAMQACheckResult:
    i_set = sets.get("I", [])
    j_set = sets.get("J", [])
    l_set = sets.get("L", [])
    k_set = sets.get("K", [])
    h_set = sets.get("H", [])

    factor_income = sum(
        _sam_value(matrix, "L", _to_upper(labor), "J", _to_upper(j))
        for labor in l_set
        for j in j_set
    ) + sum(
        _sam_value(matrix, "K", _to_upper(k), "J", _to_upper(j))
        for k in k_set
        for j in j_set
    )

    taxes = (
        sum(_sam_value(matrix, "AG", "GVT", "J", _to_upper(j)) for j in j_set)
        + sum(_sam_value(matrix, "AG", "TI", "I", _to_upper(i)) for i in i_set)
        + sum(_sam_value(matrix, "AG", "TM", "I", _to_upper(i)) for i in i_set)
        + sum(_sam_value(matrix, "AG", "GVT", "X", _to_upper(i)) for i in i_set)
    )
    gdp_income_proxy = factor_income + taxes

    final_domestic = 0.0
    for i in i_set:
        i_upper = _to_upper(i)
        final_domestic += sum(
            _sam_value(matrix, "I", i_upper, "AG", _to_upper(h)) for h in h_set
        )
        final_domestic += _sam_value(matrix, "I", i_upper, "AG", "GVT")
        final_domestic += _sam_value(matrix, "I", i_upper, "OTH", "INV")
        final_domestic += _sam_value(matrix, "I", i_upper, "OTH", "VSTK")

    exports = sum(_sam_value(matrix, "X", _to_upper(i), "AG", "ROW") for i in i_set)
    imports = sum(_sam_value(matrix, "AG", "ROW", "I", _to_upper(i)) for i in i_set)
    gdp_expenditure_proxy = final_domestic + exports - imports

    passed, abs_delta, rel_delta = _eq_pass(
        gdp_income_proxy,
        gdp_expenditure_proxy,
        spec.abs_tol,
        spec.rel_tol,
    )
    failures: list[dict[str, Any]] = []
    if not passed:
        failures.append(
            {
                "lhs_gdp_income_proxy": gdp_income_proxy,
                "rhs_gdp_expenditure_proxy": gdp_expenditure_proxy,
                "abs_delta": abs_delta,
                "rel_delta": rel_delta,
            }
        )
    return _build_check_result(spec, evaluated=1, failures=failures, max_samples=max_samples)


def run_sam_data_contracts(
    sam_data: dict[str, Any],
    *,
    sets: dict[str, list[str]] | None = None,
    balance_rel_tol: float = 1e-6,
    gdp_rel_tol: float = 0.08,
    max_samples: int = 8,
    source: str | None = None,
) -> SAMQAReport:
    """Run Phase 1 SAM QA contracts on loaded SAM data."""
    matrix = _extract_sam_matrix(sam_data)
    data_for_sets = dict(sam_data)
    data_for_sets["sam_matrix"] = matrix
    resolved_sets = sets or derive_dynamic_sets_from_sam(data_for_sets)
    specs = default_sam_contracts(
        balance_rel_tol=balance_rel_tol,
        gdp_rel_tol=gdp_rel_tol,
    )

    checks = [
        _check_export_value_balance(matrix, resolved_sets, specs["EXP001"], max_samples),
        _check_commodity_account_balance(matrix, resolved_sets, specs["DSP001"], max_samples),
        _check_margin_domestic_denominator(matrix, resolved_sets, specs["MRG001"], max_samples),
        _check_margin_export_denominator(matrix, resolved_sets, specs["MRG002"], max_samples),
        _check_tax_import_base(matrix, resolved_sets, specs["TAX001"], max_samples),
        _check_tax_export_base(matrix, resolved_sets, specs["TAX002"], max_samples),
        _check_tax_commodity_base(matrix, resolved_sets, specs["TAX003"], max_samples),
        _check_tax_production_base(matrix, resolved_sets, specs["TAX004"], max_samples),
        _check_macro_savings_investment_balance(matrix, resolved_sets, specs["MAC001"], max_samples),
        _check_macro_gdp_proxy_closure(matrix, resolved_sets, specs["MAC002"], max_samples),
    ]

    failed_checks = [check for check in checks if not check.passed]
    failed_error_checks = [check for check in failed_checks if check.severity == "error"]
    failed_warning_checks = [check for check in failed_checks if check.severity == "warning"]
    metadata = {
        "records": len(matrix),
        "set_sizes": {name: len(values) for name, values in resolved_sets.items()},
        "balance_rel_tol": balance_rel_tol,
        "gdp_rel_tol": gdp_rel_tol,
    }

    return SAMQAReport(
        schema_version="sam_qa_report/v1",
        source=source or str(sam_data.get("filepath", "")),
        passed=len(failed_error_checks) == 0,
        evaluated_checks=len(checks),
        failed_checks=len(failed_checks),
        failed_error_checks=len(failed_error_checks),
        failed_warning_checks=len(failed_warning_checks),
        checks=checks,
        metadata=metadata,
    )


def run_sam_qa_from_file(
    sam_file: Path | str,
    *,
    dynamic_sam: bool = False,
    accounts: dict[str, str] | None = None,
    balance_rel_tol: float = 1e-6,
    gdp_rel_tol: float = 0.08,
    max_samples: int = 8,
) -> SAMQAReport:
    """Load SAM from file and execute all Phase 1 QA contracts."""
    sam_path = Path(sam_file)
    sam_data = load_sam_data(sam_path)
    if dynamic_sam:
        sam_data = remap_dynamic_sam_accounts(sam_data, accounts=accounts)
    return run_sam_data_contracts(
        sam_data,
        balance_rel_tol=balance_rel_tol,
        gdp_rel_tol=gdp_rel_tol,
        max_samples=max_samples,
        source=str(sam_path),
    )
