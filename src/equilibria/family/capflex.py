"""capFlex's ``risk[r]`` (``rorg/rore(r)`` at the benchmark) with no Pyomo model.

The values are read from the GAMS reference GDX by Pyomo's own reader
(``_capflex_risk_from_gdx``), so they are bitwise the ones ``build_block_model`` stamps.
There is no twin-solve fallback and no ``risk = 1``: without a GDX the run stops."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any


def _unsupported(cause: str) -> NotImplementedError:
    return NotImplementedError(
        f"family backend: savf_flag='capFlex' needs rore/rorg from a GAMS GDX: {cause}"
    )


def require_ref_gdx(ref_gdx: Any) -> None:
    """``NotImplementedError`` unless ``ref_gdx`` names an existing file (cheap, so the
    driver can refuse capFlex before any work)."""
    if ref_gdx is None:
        raise _unsupported("no ref_gdx was given")
    if not Path(str(ref_gdx)).exists():
        raise _unsupported(f"{ref_gdx} does not exist")


def capflex_risk(params: Any, ref_gdx: Any) -> dict[str, float]:
    """``{region: rorg/rore(r)}`` from the reference GDX; ``NotImplementedError`` when
    the GDX cannot give it."""
    require_ref_gdx(ref_gdx)
    try:
        importlib.import_module("_diff_core")  # scripts/gtap, on sys.path in parity env
    except ImportError as exc:
        raise _unsupported(f"_diff_core is not importable ({exc})") from exc
    from equilibria.templates.gtap.gtap_block_model import _capflex_risk_from_gdx

    risk = _capflex_risk_from_gdx(ref_gdx, [str(r) for r in params.sets.r])
    if not risk:
        raise _unsupported(f"{ref_gdx} has no rorg('base') or a region's rore")
    return risk


def stamp_capflex_risk(params: Any, closure: Any, ref_gdx: Any) -> None:
    """Set ``params._capflex_risk`` under capFlex (before any cache key is taken, so
    capFix and capFlex records never share a key). Under any other closure it removes a
    ``_capflex_risk`` left by an earlier capFlex run, so that run's value cannot move the
    capFix keys or results."""
    if getattr(closure, "savf_flag", "capFix") != "capFlex":
        if hasattr(params, "_capflex_risk"):
            del params._capflex_risk
        return
    risk = capflex_risk(params, ref_gdx)
    held = getattr(params, "_capflex_risk", None)
    if held is not None and held != risk:
        raise ValueError(
            f"params._capflex_risk is stale: {held} differs from the GDX's {risk}"
        )
    params._capflex_risk = risk
