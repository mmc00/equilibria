"""Summary of a solved period: Walras, equation residuals and the worst equations.

The residual of an equation is ``|body - rhs|``; an inequality only counts when
it is violated. Residuals are absolute, in the model's own units, as the solver
sees them.

A variable "on its floor" has a positive lower bound and a value within
``FLOOR_RTOL`` of it. Prices carry such floors (``blocks/gtap/floors.py``:
``max(1e-8, 1e-3 * init)``) only to keep the solver away from zero; a solve
that ends with code 1 and a price there has found a corner the model does not
mean, so it is flagged. A zero bound is left alone: a quantity at zero is
usually a real corner (an absent flow).
"""

from __future__ import annotations

import logging
import statistics
from dataclasses import dataclass, field
from typing import Any

from rich import box
from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from equilibria.version import __version__

# Matrix palette of the approved mockup (dev-tools plans/research/report-mockup).
_G, _G2, _G3, _WARN, _RED = "#00ff41", "#00c832", "#008f11", "#d4ff00", "#ff3b3b"

_log = logging.getLogger(__name__)

#: Relative distance to the lower bound under which a variable is on its floor.
FLOOR_RTOL = 1e-6


@dataclass(frozen=True)
class WorstEquation:
    equation: str
    residual: float


@dataclass(frozen=True)
class SolveSummary:
    period: str | None
    code: int | None
    walras: float | None
    n_equations: int
    max_residual: float
    mean_residual: float
    median_residual: float
    worst: list[WorstEquation] = field(default_factory=list)
    unevaluated: int = 0
    on_floor: list[str] = field(default_factory=list)


def _in_period(index: Any, period: str | None) -> bool:
    if period is None:
        return True
    if isinstance(index, tuple):
        return bool(index) and index[-1] == period
    return index == period


def _residual(con) -> float:
    from pyomo.environ import value

    body = float(value(con.body))
    lo = None if con.lower is None else float(value(con.lower))
    up = None if con.upper is None else float(value(con.upper))
    if con.equality:
        target = lo if lo is not None else up
        assert target is not None  # an equality always has its right-hand side
        return abs(body - target)
    r = 0.0
    if lo is not None and body < lo:
        r = lo - body
    if up is not None and body > up:
        r = max(r, body - up)
    return r


def _walras(m, period: str | None) -> float | None:
    from pyomo.environ import value

    w = getattr(m, "walras", None)
    if w is None:
        return None
    if not w.is_indexed():
        return float(value(w))
    for idx in w:
        if _in_period(idx, period):
            return float(value(w[idx]))
    return None


def _on_floor(m, period: str | None) -> list[str]:
    from pyomo.environ import Var

    names = []
    for v in m.component_data_objects(Var, descend_into=True):
        lb = v.lb
        if (
            lb is None
            or lb <= 0
            or v.value is None
            or not _in_period(v.index(), period)
        ):
            continue
        if v.value <= lb * (1.0 + FLOOR_RTOL):
            names.append(v.name)
    return names


def summary(
    m,
    results: dict[str, dict[str, Any]] | None = None,
    *,
    period: str | None = "shock",
    top: int = 5,
    console: Console | None = None,
) -> SolveSummary:
    """Print the end-of-solve panel for ``period`` and return its numbers.

    ``results`` is what ``solve_multiperiod`` returns (its ``code`` for the
    period goes in the header). ``period=None`` takes every active equation,
    for models without periods. Equations whose value cannot be evaluated are
    counted in ``unevaluated``, not dropped silently.
    """
    from pyomo.environ import Constraint

    if period is not None and not any(
        _in_period(c.index(), period)
        for c in m.component_data_objects(Constraint, active=True)
    ):
        if any(
            _in_period(c.index(), period) for c in m.component_data_objects(Constraint)
        ):
            # Not falling back to another period: that would print the wrong
            # period's numbers under this one's name.
            raise ValueError(
                f"period {period!r} has no active equations: the solve "
                "deactivates a period's equations once it moves on to the next "
                "one; call summary() right after that period is solved"
            )
        period = None  # a model without periods

    rows: list[tuple[float, str]] = []
    unevaluated = 0
    for con in m.component_data_objects(Constraint, active=True, descend_into=True):
        if not _in_period(con.index(), period):
            continue
        try:
            rows.append((_residual(con), con.name))
        except (ValueError, ArithmeticError, TypeError):
            unevaluated += 1

    res = [r for r, _ in rows]
    rows.sort(key=lambda row: row[0], reverse=True)
    code = None
    if results is not None and period is not None and period in results:
        code = results[period].get("code")
    s = SolveSummary(
        period=period,
        code=code,
        walras=_walras(m, period),
        n_equations=len(rows),
        max_residual=max(res, default=0.0),
        mean_residual=statistics.fmean(res) if res else 0.0,
        median_residual=statistics.median(res) if res else 0.0,
        worst=[WorstEquation(name, r) for r, name in rows[:top]],
        unevaluated=unevaluated,
        on_floor=_on_floor(m, period),
    )
    _to_log(s)
    (console or Console()).print(_panel(s))
    return s


def _to_log(s: SolveSummary) -> None:
    """The panel's numbers as plain text, for a log file of an unattended run."""
    worst = (
        f", worst {s.worst[0].equation} {_e(s.worst[0].residual)}" if s.worst else ""
    )
    _log.info(
        "%s: code %s, %d equations, max residual %s, mean %s, median %s, walras %s%s",
        s.period or "model",
        s.code if s.code is not None else "-",
        s.n_equations,
        _e(s.max_residual),
        _e(s.mean_residual),
        _e(s.median_residual),
        _e(s.walras),
        worst,
    )
    if s.code not in (None, 1):
        _log.warning("%s: solve ended with code %s", s.period or "model", s.code)
    if s.on_floor:
        _log.warning(
            "%s: %d variables on their floor: %s",
            s.period or "model",
            len(s.on_floor),
            ", ".join(s.on_floor),
        )
    if s.unevaluated:
        _log.warning(
            "%s: %d equations could not be evaluated",
            s.period or "model",
            s.unevaluated,
        )


def _e(x: float | None) -> str:
    return "-" if x is None else f"{x:.0e}"


def _table(title: str) -> Table:
    return Table(
        box=box.SQUARE,
        border_style=_G3,
        header_style=f"bold {_G}",
        expand=True,
        title=title,
        title_style=f"bold {_G}",
        style=_G2,
    )


def _panel(s: SolveSummary) -> Panel:
    ok = s.code in (None, 1)
    head = Text.assemble(
        (">_ equilibria", f"bold {_G}"),
        (f"  ::  period {s.period or '-'}", _G3),
        (f"  ::  code {s.code if s.code is not None else '-'}", _G3),
        (f"  ::  {s.n_equations:,} equations", _G3),
    )

    r = _table("[ RESIDUALS ]")
    for h in ("MAX", "MEAN", "MEDIAN", "WALRAS"):
        r.add_column(h, justify="right")
    r.add_row(
        _e(s.max_residual), _e(s.mean_residual), _e(s.median_residual), _e(s.walras)
    )

    w = _table("[ WORST EQUATIONS ]")
    w.add_column("EQUATION")
    w.add_column("RESIDUAL", justify="right")
    for eq in s.worst:
        # Text, not str: rich would read an index such as [shock] as markup
        w.add_row(Text(eq.equation), _e(eq.residual))

    parts: list[Any] = [head, r, w]
    if s.on_floor:
        shown = ", ".join(s.on_floor[:5]) + (" ..." if len(s.on_floor) > 5 else "")
        parts.append(
            Text.assemble(
                ("[WARN] ", f"bold {_WARN}"),
                (f"{len(s.on_floor)} on their floor: ", _WARN),
                (shown, _WARN),
            )
        )
    else:
        parts.append(
            Text.assemble(("[ OK ] ", f"bold {_G}"), ("no price on its floor", _G2))
        )
    if s.unevaluated:
        parts.append(
            Text(
                f"[WARN] {s.unevaluated} equations could not be evaluated",
                style=f"bold {_RED}",
            )
        )
    title = "SOLVE COMPLETE" if ok else f"SOLVE FAILED (code {s.code})"
    return Panel(
        Group(*parts),
        box=box.DOUBLE,
        border_style=_G if ok else _RED,
        padding=(0, 1),
        title=f"[bold {_G if ok else _RED}] {title} [/]",
        subtitle=f"[{_G3}] equilibria v{__version__} [/]",
    )
