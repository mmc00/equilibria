"""The CONVERT files of MANAGE's 2015 solve: names, levels and GAMS's own rows.

With ``convertMCP on`` the recorded run wrote the scalar model of its 2015 solve
twice (data/manage/gha_gtap9/gams_mcp_baseline_y2015_nlpec-ipopt/):
``gha2015_scalar.gms`` at the start point and ``gha2015_solution.gms`` at the
solution, as rows ``eN.. <expr> =E= <expr>;`` over columns ``x1..xN``, with the
levels as ``xN.l = v;`` (a zero level is not written). ``gha2015_dict.txt``
names every row and column (``e1  e_px(a-pdr,2015)``).

Everything here is in GAMS's solve units and GAMS's row scaling (each row is
divided by the 2014 level of its variable, ``model.inc:1198-1217``): the rows
are GAMS's own residuals, not equilibria's.

``ConvertRows`` translates each row to Python. The expressions use only
``xN``, numbers, ``+ - * / **``, ``eps`` and the intrinsics ``ncpvupow`` and
``sigmoid`` (``intrinsics.py``). Measured on 2026-11-27: at the start point the
largest residual is 2.33 in ``e_paf(c-coa,a-i_s)`` and 0.68 in
``e_nshrGDP(h-hhold)``, the values the recorded run reports; at the solution
every row is below the floor ``test_convert.py`` records.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from equilibria.templates.manage.intrinsics import ncp_vu_pow, sigmoid

Name = tuple[str, tuple[str, ...]]

_DICT_LINE = re.compile(r"^\s+([ex])(\d+)\s+(\w+)(?:\((.*)\))?\s*$")
_LEVEL = re.compile(r"^x(\d+)\.l\s*=\s*([^;]+);")
_XVAR = re.compile(r"\bx(\d+)\b")


@dataclass(frozen=True)
class ConvertNames:
    """Row and column names, in CONVERT's order (``rows[0]`` is ``e1``)."""

    rows: tuple[Name, ...]
    cols: tuple[Name, ...]


def _index(text: str | None) -> tuple[str, ...]:
    if not text:
        return ()
    return tuple(p.strip().strip('"').strip("'") for p in text.split(","))


def read_dict(path: Path) -> ConvertNames:
    """The names of ``gha2015_dict.txt``: ``(family, index)`` per row and column,
    the year included as the last index (``e_dummy`` has none)."""
    rows: dict[int, Name] = {}
    cols: dict[int, Name] = {}
    for line in Path(path).read_text().splitlines():
        m = _DICT_LINE.match(line)
        if m:
            kind, n, fam, idx = m.groups()
            (rows if kind == "e" else cols)[int(n)] = (fam, _index(idx))
    if sorted(rows) != list(range(1, len(rows) + 1)) or sorted(cols) != list(
        range(1, len(cols) + 1)
    ):
        raise ValueError(f"{path}: rows or columns are not numbered 1..N")
    return ConvertNames(
        tuple(rows[i] for i in sorted(rows)), tuple(cols[i] for i in sorted(cols))
    )


def read_levels(path: Path) -> dict[int, float]:
    """``{column: level}`` of a CONVERT ``.gms``; absent columns are at 0."""
    out: dict[int, float] = {}
    for line in Path(path).read_text().splitlines():
        m = _LEVEL.match(line)
        if m:
            out[int(m.group(1))] = float(m.group(2))
    return out


def _drop_year(idx: tuple[str, ...]) -> tuple[str, ...]:
    return idx[:-1] if idx and idx[-1].isdigit() and len(idx[-1]) == 4 else idx


def instance(names: ConvertNames) -> dict[str, frozenset[tuple[str, ...]]]:
    """The rows of each equation family, without the year."""
    out: dict[str, set[tuple[str, ...]]] = {}
    for fam, idx in names.rows:
        out.setdefault(fam, set()).add(_drop_year(idx))
    return {k: frozenset(v) for k, v in out.items()}


def levels_by_family(
    names: ConvertNames, levels: Mapping[int, float]
) -> dict[str, dict[tuple[str, ...], float]]:
    """``{family: {index without the year: level}}`` of every column."""
    out: dict[str, dict[tuple[str, ...], float]] = {}
    for j, (fam, idx) in enumerate(names.cols, start=1):
        out.setdefault(fam, {})[_drop_year(idx)] = levels.get(j, 0.0)
    return out


class ConvertRows:
    """GAMS's rows of a CONVERT scalar model, evaluated in Python."""

    def __init__(self, sources: Sequence[str], ncols: int) -> None:
        self.sources = tuple(sources)
        self.ncols = ncols
        env: dict[str, Any] = {"_vu": ncp_vu_pow, "_sg": sigmoid}
        code = "\n".join(f"r{n} = lambda x: {src}" for n, src in enumerate(sources))
        exec(compile(code, "<convert rows>", "exec"), env)  # noqa: S102
        self._rows: list[Callable[[list[float]], float]] = [
            env[f"r{n}"] for n in range(len(sources))
        ]

    @classmethod
    def from_gms(cls, path: Path) -> ConvertRows:
        text = Path(path).read_text()
        body = text.split("\n* set non-default levels", 1)[0]
        defs = re.findall(r"^e(\d+)\.\.(.*?);", body, flags=re.S | re.M)
        if [int(n) for n, _ in defs] != list(range(1, len(defs) + 1)):
            raise ValueError(f"{path}: rows are not e1..eN in order")
        ncols = max(int(n) for n in _XVAR.findall(body))
        return cls([_translate(expr) for _, expr in defs], ncols)

    def residuals(self, levels: Mapping[int, float]) -> list[float]:
        """``lhs - rhs`` of every row at ``levels`` (``{column: level}``, absent
        columns at 0); ``nan`` where GAMS would hit a domain error."""
        x = [0.0] * (self.ncols + 1)
        for j, v in levels.items():
            x[j] = v
        out: list[float] = []
        for row in self._rows:
            try:
                out.append(float(row(x)))
            # a negative base to a fractional power gives a complex number,
            # float() of it a TypeError: GAMS's vcPower domain error
            except (ZeroDivisionError, OverflowError, ValueError, TypeError):
                out.append(math.nan)
        return out


def _translate(expr: str) -> str:
    lhs, rhs = re.split(r"=E=", " ".join(expr.split()), maxsplit=1)
    src = f"({lhs}) - ({rhs})"
    src = _XVAR.sub(r"x[\1]", src)
    src = re.sub(r"\bncpvupow\(", "_vu(", src)
    src = re.sub(r"\bsigmoid\(", "_sg(", src)
    src = re.sub(r"\beps\b", "0.0", src)
    if re.search(r"[A-Za-z_]\w*\(", src.replace("_vu(", "").replace("_sg(", "")):
        raise ValueError(f"unknown function in CONVERT row: {expr.strip()[:80]}")
    return src
