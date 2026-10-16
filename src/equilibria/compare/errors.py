"""Errores de :mod:`equilibria.compare`: una sola familia.

:class:`CompareError` es la base (subclase de ``ValueError``); :class:`MappingError`
la especializa para un mapeo incompleto o incoherente.
"""

from __future__ import annotations


class CompareError(ValueError):
    """Una fuente, un mapeo o una llamada a ``compare`` no se puede usar tal cual."""


class MappingError(CompareError):
    """El mapeo esta incompleto o es incoherente."""
