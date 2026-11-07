"""Linea de comandos ``equilibria``: un grupo de subcomandos (``compare``, ``doctor``)."""

from __future__ import annotations

import click

from equilibria.compare.cli import compare_command
from equilibria.doctor.cli import doctor_command


@click.group()
def main() -> None:
    """equilibria: herramientas de linea de comandos."""


main.add_command(compare_command)
main.add_command(doctor_command)
