"""Linea de comandos ``equilibria``: un grupo de subcomandos (hoy, ``compare``)."""

from __future__ import annotations

import click

from equilibria.compare.cli import compare_command


@click.group()
def main() -> None:
    """equilibria: herramientas de linea de comandos."""


main.add_command(compare_command)
