"""Report: resultado de :func:`compare` y su exportacion.

Todo lo que no se compara queda declarado: celdas sin pareja (``unmatched``),
variables excluidas con su motivo (``excluded``), celdas excluidas por el mapeo
(``exclusions`` con su motivo y conteo, ``excluded_cells`` una por una) y periodos
que solo existen en un lado (``periods_only_in_a``/``_b``). Una celda excluida no
se mide: nunca cuenta como que pasa ni como sin pareja.
"""

from __future__ import annotations

import csv
import io
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from equilibria.compare.align import align, common_periods
from equilibria.compare.errors import CompareError
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
class Exclusion:
    """Una exclusion por celda del mapeo y cuantas celdas saco (``cells``).

    Una celda que cumple varias exclusiones cuenta en la primera declarada."""

    variable: str
    where: dict[str, list[str]]  # incluye "period" si la exclusion lo restringe
    why: str
    cells: int


@dataclass(frozen=True)
class ExcludedCell:
    """Una celda excluida; ``a``/``b`` son ``None`` si falta en ese lado."""

    variable: str
    period: str
    index: Index
    a: float | None
    b: float | None
    why: str


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
    exclusions: list[Exclusion]
    excluded_cells: list[ExcludedCell]

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
            "exclusions": [
                {"var": e.variable, "where": e.where, "why": e.why, "cells": e.cells}
                for e in self.exclusions
            ],
            "excluded_cells": [
                {
                    "variable": x.variable,
                    "period": x.period,
                    "index": list(x.index),
                    "a": x.a,
                    "b": x.b,
                    "why": x.why,
                }
                for x in self.excluded_cells
            ],
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
        ``fail``, ``missing_in_a``, ``missing_in_b`` o ``excluded``; una celda sin
        pareja deja ``a``, ``b`` y ``d`` vacios; una excluida deja ``d`` vacio (no se
        mide) y ``a``/``b`` vacios si falta en ese lado. Primero las parejas, luego
        las sin pareja, luego las excluidas.
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
        for x in self.excluded_cells:
            w.writerow(
                [
                    x.variable,
                    x.period,
                    ".".join(x.index),
                    "" if x.a is None else x.a,
                    "" if x.b is None else x.b,
                    "",
                    "excluded",
                ]
            )
        text = buf.getvalue()
        if path is not None:
            Path(path).write_text(text)
        return text


def compare(a: Solution, b: Solution, mapping: Mapping, reference: str = "b") -> Report:
    """Compara ``a`` y ``b`` en niveles, celda por celda, segun ``mapping``.

    ``reference`` es el lado del denominador de ``d = |A-B| / max(|ref|, abs_floor)``.
    Solo se comparan los periodos (ya renombrados) presentes en los dos lados. Las
    celdas que cumplen una exclusion por celda del mapeo (``where``) no se miden: van
    a ``Report.excluded_cells`` y su conteo a ``Report.exclusions``.
    """
    if reference not in ("a", "b"):
        raise CompareError(f"reference debe ser 'a' o 'b', no {reference!r}")
    th = mapping.thresholds
    common, only_a, only_b = common_periods(a, b, mapping)
    cells: list[Cell] = []
    unmatched: list[Unmatched] = []
    summary: dict[str, dict[str, Any]] = {}
    counts = {id(ce): 0 for ce in mapping.cell_exclusions}
    excluded_cells: list[ExcludedCell] = []
    for al in align(a, b, mapping, set(common)):
        vm = al.var
        n_excluded_before = len(excluded_cells)
        mine_excl = [ce for ce in mapping.cell_exclusions if ce.variable == vm.name]

        def excluded(
            key: tuple[str, Index], va: float | None, vb: float | None
        ) -> bool:
            ce = next((c for c in mine_excl if c.matches(vm.a.dims, *key)), None)
            if ce is None:
                return False
            counts[id(ce)] += 1
            excluded_cells.append(ExcludedCell(vm.name, *key, va, vb, ce.why))
            return True

        mine = []
        for (period, index), va, vb in al.pairs:
            if excluded((period, index), va, vb):
                continue
            d = rel_diff(va, vb, vb if reference == "b" else va, vm.tol.abs_floor)
            mine.append(Cell(vm.name, period, index, va, vb, d, d <= vm.tol.rel))
        cells.extend(mine)
        miss_b = [k for k, v in al.missing_in_b.items() if not excluded(k, v, None)]
        miss_a = [k for k, v in al.missing_in_a.items() if not excluded(k, None, v)]
        unmatched += [Unmatched(vm.name, p, i, "b") for p, i in miss_b]
        unmatched += [Unmatched(vm.name, p, i, "a") for p, i in miss_a]
        s = summarize(
            [c.d for c in mine],
            [c.passed for c in mine],
            [(c.period, c.index) for c in mine],
            th,
        )
        s["missing_in_a"] = len(miss_a)
        s["missing_in_b"] = len(miss_b)
        s["excluded"] = len(excluded_cells) - n_excluded_before
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
    overall["excluded"] = len(excluded_cells)
    exclusions = [
        Exclusion(
            ce.variable,
            {d: sorted(e) for d, e in ce.where.items()}
            | ({} if ce.periods is None else {"period": sorted(ce.periods)}),
            ce.why,
            counts[id(ce)],
        )
        for ce in mapping.cell_exclusions
    ]
    return Report(
        reference,
        th,
        cells,
        unmatched,
        summary,
        overall,
        dict(mapping.excluded),
        common,
        only_a,
        only_b,
        exclusions,
        excluded_cells,
    )
