"""Reporting models for SAM QA checks."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

Severity = Literal["error", "warning"]


@dataclass
class SAMQACheckResult:
    """Result for one QA contract gate."""

    code: str
    title: str
    category: str
    description: str
    severity: Severity
    passed: bool
    evaluated: int
    failures: int
    max_abs_delta: float
    max_rel_delta: float
    abs_tol: float
    rel_tol: float
    samples: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SAMQAReport:
    """Top-level SAM QA report."""

    schema_version: str
    source: str
    passed: bool
    evaluated_checks: int
    failed_checks: int
    failed_error_checks: int
    failed_warning_checks: int
    checks: list[SAMQACheckResult]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "source": self.source,
            "passed": self.passed,
            "evaluated_checks": self.evaluated_checks,
            "failed_checks": self.failed_checks,
            "failed_error_checks": self.failed_error_checks,
            "failed_warning_checks": self.failed_warning_checks,
            "checks": [check.to_dict() for check in self.checks],
            "metadata": self.metadata,
        }

    def save_json(self, path: Path | str) -> None:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(self.to_dict(), indent=2))


def format_report_summary(report: SAMQAReport) -> str:
    """Compact human-readable summary line."""
    status = "PASS" if report.passed else "FAIL"
    return (
        f"SAM QA {status} | checks={report.evaluated_checks} "
        f"failed={report.failed_checks} errors={report.failed_error_checks} "
        f"warnings={report.failed_warning_checks}"
    )
