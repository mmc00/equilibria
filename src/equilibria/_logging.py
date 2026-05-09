"""Logging configuration helpers for equilibria.

This module follows the standard Python recommendation for library logging
(https://docs.python.org/3/howto/logging.html#configuring-logging-for-a-library):

  * The library installs a ``NullHandler`` on the root ``equilibria`` logger.
    This silences the "No handlers could be found" warning when the consumer
    has not configured logging.
  * Each module obtains its own logger via ``logging.getLogger(__name__)``,
    yielding a hierarchical namespace (``equilibria.<package>.<module>``).
  * The library never calls ``logging.basicConfig`` or attaches user-visible
    handlers — that is the consumer's responsibility.

For scripts and notebooks, :func:`setup_logging` provides a one-line opt-in
that configures a sensible default handler on the ``equilibria`` logger only.
"""

from __future__ import annotations

import logging
import sys
from typing import TextIO

_LIBRARY_LOGGER_NAME = "equilibria"
_DEFAULT_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"
_DEFAULT_DATEFMT = "%H:%M:%S"


def _install_null_handler() -> None:
    """Attach a ``NullHandler`` to the library root logger (idempotent)."""
    root = logging.getLogger(_LIBRARY_LOGGER_NAME)
    if not any(isinstance(h, logging.NullHandler) for h in root.handlers):
        root.addHandler(logging.NullHandler())


def setup_logging(
    level: int | str = logging.INFO,
    *,
    stream: TextIO | None = None,
    fmt: str = _DEFAULT_FORMAT,
    datefmt: str = _DEFAULT_DATEFMT,
) -> logging.Logger:
    """Configure a stream handler on the ``equilibria`` logger.

    Intended for scripts, notebooks, and CLI entry points. Library code
    should never call this — it should only call ``logging.getLogger(__name__)``.

    Parameters
    ----------
    level:
        Log level (e.g. ``logging.INFO``, ``"DEBUG"``).
    stream:
        Output stream. Defaults to ``sys.stderr``.
    fmt, datefmt:
        Standard ``logging.Formatter`` arguments.

    Returns
    -------
    The configured ``equilibria`` logger.
    """
    logger = logging.getLogger(_LIBRARY_LOGGER_NAME)
    logger.setLevel(level)

    handler = logging.StreamHandler(stream if stream is not None else sys.stderr)
    handler.setFormatter(logging.Formatter(fmt=fmt, datefmt=datefmt))
    logger.addHandler(handler)
    logger.propagate = False
    return logger
