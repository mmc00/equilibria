"""``equilibria doctor``: print the dependency checks; exit 1 if a solve cannot run."""

from __future__ import annotations

import click
from rich import box
from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from equilibria.doctor.dependencies import MISSING, OK, check_dependencies

# Same palette as equilibria.report.summary.
_G, _G2, _G3, _WARN, _RED = "#00ff41", "#00c832", "#008f11", "#d4ff00", "#ff3b3b"
_TAG = {OK: ("[ OK ]", _G), MISSING: ("[MISS]", _RED)}


@click.command("doctor")
def doctor_command() -> None:
    """Check that PATH, its license and the solver packages are installed."""
    checks = check_dependencies()
    t = Table(
        box=box.SQUARE,
        border_style=_G3,
        header_style=f"bold {_G}",
        expand=True,
        style=_G2,
    )
    t.add_column("CHECK")
    t.add_column("STATUS", justify="center")
    t.add_column(
        "DETAIL", overflow="fold"
    )  # fold, not "…": paths and URLs are copied from here
    for c in checks:
        tag, color = _TAG.get(c.status, ("[WARN]", _WARN))
        detail = Text(c.detail)
        if c.fix:
            detail.append(f"\nfix: {c.fix}", style=_G3)
        t.add_row(Text(c.name), Text(tag, style=f"bold {color}"), detail)

    missing = [c for c in checks if c.status == MISSING]
    title, color = ("ALL SET", _G) if not missing else (f"{len(missing)} MISSING", _RED)
    Console(soft_wrap=False).print(
        Panel(
            Group(Text(">_ equilibria doctor", style=f"bold {_G}"), t),
            box=box.DOUBLE,
            border_style=color,
            title=f"[bold {color}] {title} [/]",
            padding=(0, 1),
        )
    )
    if missing:
        raise SystemExit(1)
