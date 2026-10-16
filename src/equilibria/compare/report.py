"""Report: resultado de :func:`compare` y su exportacion.

Todo lo que no se compara queda declarado: celdas sin pareja (``unmatched``),
variables excluidas con su motivo (``excluded``) y periodos que solo existen en un
lado (``periods_only_in_a``/``_b``).
"""

from __future__ import annotations

import csv
import io
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from equilibria.compare.align import align, common_periods
from equilibria.compare.mapping import Mapping
from equilibria.compare.measure import rel_diff, summarize
from equilibria.compare.solution import Index, Solution


@dataclass(frozen=True)
class Cell:
    """Una pareja comparada. ``index`` en el orden de dims de A, con nombres comunes."""

    variable: str
    period: str
    index: Index
    a: float
    b: float
    d: float
    passed: bool


@dataclass(frozen=True)
class Unmatched:
    """Una celda presente en un lado y ausente en el otro (``missing_in``)."""

    variable: str
    period: str
    index: Index
    missing_in: str


@dataclass(frozen=True)
class Report:
    reference: str
    thresholds: tuple[float, ...]
    cells: list[Cell]
    unmatched: list[Unmatched]
    summary: dict[str, dict[str, Any]]
    overall: dict[str, Any]
    excluded: dict[str, str]
    periods_compared: list[str]
    periods_only_in_a: list[str]
    periods_only_in_b: list[str]

    def to_dict(self) -> dict[str, Any]:
        def fix(s: dict[str, Any]) -> dict[str, Any]:
            s = dict(s)
            s["pct_within"] = {repr(t): v for t, v in s["pct_within"].items()}
            if s.get("max_cell") is not None:
                period, index = s["max_cell"]
                s["max_cell"] = {"period": period, "index": list(index)}
            return s

        return {
            "reference": self.reference,
            "thresholds": list(self.thresholds),
            "periods_compared": self.periods_compared,
            "periods_only_in_a": self.periods_only_in_a,
            "periods_only_in_b": self.periods_only_in_b,
            "overall": fix(self.overall),
            "summary": {k: fix(v) for k, v in self.summary.items()},
            "excluded": [{"var": k, "why": v} for k, v in self.excluded.items()],
            "unmatched": [
                {
                    "variable": u.variable,
                    "period": u.period,
                    "index": list(u.index),
                    "missing_in": u.missing_in,
                }
                for u in self.unmatched
            ],
            "cells": [
                {
                    "variable": c.variable,
                    "period": c.period,
                    "index": list(c.index),
                    "a": c.a,
                    "b": c.b,
                    "d": c.d,
                    "passed": c.passed,
                }
                for c in self.cells
            ],
        }

    def to_json(self, path: str | Path | None = None, *, indent: int | None = 2) -> str:
        """JSON del reporte; si se da ``path``, ademas lo escribe ahi."""
        text = json.dumps(self.to_dict(), indent=indent)
        if path is not None:
            Path(path).write_text(text)
        return text

    def to_csv(self, path: str | Path | None = None) -> str:
        """CSV del reporte, una fila por celda; si se da ``path``, ademas lo escribe.

        Columnas ``variable, period, index, a, b, d, status``. ``index`` une los
        elementos con ``"."`` (como GAMS: ``USA.Food``). ``status`` es ``pass``,
        ``fail``, ``missing_in_a`` o ``missing_in_b``; una celda sin pareja deja
        ``a``, ``b`` y ``d`` vacios. Primero las parejas, luego las sin pareja.
        """
        buf = io.StringIO()
        w = csv.writer(buf, lineterminator="\n")
        w.writerow(["variable", "period", "index", "a", "b", "d", "status"])
        for c in self.cells:
            status = "pass" if c.passed else "fail"
            w.writerow([c.variable, c.period, ".".join(c.index), c.a, c.b, c.d, status])
        for u in self.unmatched:
            w.writerow(
                [
                    u.variable,
                    u.period,
                    ".".join(u.index),
                    "",
                    "",
                    "",
                    f"missing_in_{u.missing_in}",
                ]
            )
        text = buf.getvalue()
        if path is not None:
            Path(path).write_text(text)
        return text


def compare(a: Solution, b: Solution, mapping: Mapping, reference: str = "b") -> Report:
    """Compara ``a`` y ``b`` en niveles, celda por celda, segun ``mapping``.

    ``reference`` es el lado del denominador de ``d = |A-B| / max(|ref|, abs_floor)``.
    Solo se comparan los periodos (ya renombrados) presentes en los dos lados.
    """
    if reference not in ("a", "b"):
        raise ValueError(f"reference debe ser 'a' o 'b', no {reference!r}")
    th = mapping.thresholds
    _, only_a, only_b = common_periods(a, b, mapping)
    cells: list[Cell] = []
    unmatched: list[Unmatched] = []
    summary: dict[str, dict[str, Any]] = {}
    for al in align(a, b, mapping):
        vm = al.var
        mine = []
        for (period, index), va, vb in al.pairs:
            d = rel_diff(va, vb, vb if reference == "b" else va, vm.tol.abs_floor)
            mine.append(Cell(vm.name, period, index, va, vb, d, d <= vm.tol.rel))
        cells.extend(mine)
        unmatched += [Unmatched(vm.name, p, i, "b") for p, i in al.missing_in_b]
        unmatched += [Unmatched(vm.name, p, i, "a") for p, i in al.missing_in_a]
        s = summarize(
            [c.d for c in mine],
            [c.passed for c in mine],
            [(c.period, c.index) for c in mine],
            th,
        )
        s["missing_in_a"] = len(al.missing_in_a)
        s["missing_in_b"] = len(al.missing_in_b)
        s["tol"] = {"rel": vm.tol.rel, "abs_floor": vm.tol.abs_floor}
        summary[vm.name] = s
    overall = summarize(
        [c.d for c in cells],
        [c.passed for c in cells],
        [(c.variable, c.period, c.index) for c in cells],
        th,
    )
    mc = overall.pop("max_cell")
    overall["max_variable"] = None if mc is None else mc[0]
    overall["max_cell"] = None if mc is None else mc[1:]
    overall["missing_in_a"] = sum(u.missing_in == "a" for u in unmatched)
    overall["missing_in_b"] = sum(u.missing_in == "b" for u in unmatched)
    return Report(
        reference,
        th,
        cells,
        unmatched,
        summary,
        overall,
        dict(mapping.excluded),
        common_periods(a, b, mapping)[0],
        only_a,
        only_b,
    )
