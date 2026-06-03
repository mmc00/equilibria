"""Shared helpers for t-indexed Pyomo GTAP models."""
from __future__ import annotations


def t0(model) -> str:
    """Return the base-period label from a built model.

    Centralizes the 'base' literal so future t_set conventions don't
    require a sweep across call sites.
    """
    return next(iter(model.t0))
