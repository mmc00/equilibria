"""Compatibility shim — the real module now lives in the installed package.

Moved to ``equilibria.solver._closure_patches`` together with the PATH C-API
solve that calls it: it is part of the MCP squaring, so it has to ship inside
the wheel. Kept here so ``scripts/gtap`` consumers that do
``from _closure_patches import ...`` (nl_compare.py, _parity_datasets.py) keep
working with sys.path-based loading.
"""

from __future__ import annotations

from equilibria.solver._closure_patches import *  # noqa: F401,F403
from equilibria.solver._closure_patches import (  # noqa: F401
    apply_squareness_patches,
    structural_matching,
)
