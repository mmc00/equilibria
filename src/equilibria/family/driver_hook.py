"""Opt-in hook: route the newton_tr driver's F and J through ``HybridNLP``.

``EQUILIBRIA_GTAP_FAMILY_EVAL=1`` turns it on; unset, the driver keeps its PyomoNLP.
The family recording is made once per (params, closure) and reused across periods.
"""

from __future__ import annotations

import os
import sys
from typing import Any

ENV = "EQUILIBRIA_GTAP_FAMILY_EVAL"

_CACHE: dict[tuple, tuple[Any, Any]] = {}


def _recording(params: Any, closure: Any, residual_region: str | None):
    from equilibria.backends.family_backend import FamilyBackend
    from equilibria.templates.gtap.gtap_block_model import compose_block_state

    key = (
        id(params),
        None if closure is None else closure.model_dump_json(),
        residual_region,
    )
    if key not in _CACHE:
        model, state = compose_block_state(
            params, params.sets, closure, residual_region
        )
        backend = FamilyBackend()
        backend.build(model)
        _CACHE[key] = (backend, state)
    return _CACHE[key]


def maybe_wrap(nlp: Any, params: Any, closure: Any, method: str) -> Any:
    """``nlp`` itself unless the hook is on and the solver is newton_tr."""
    if os.environ.get(ENV) != "1" or method != "newton_tr":
        return nlp
    if bool(getattr(closure, "if_sub", False)):
        raise NotImplementedError(
            f"{ENV}=1: ifSUB=1 is not supported by the family backend"
        )
    from equilibria.family.hybrid import HybridNLP

    region = getattr(nlp.pyomo_model(), "_residual_region", None)
    backend, state = _recording(params, closure, region)
    h = HybridNLP(nlp, backend, state)
    print(
        f"[family-eval] period={h.period} served={h.family_rows.size} rows, "
        f"fallback={sorted(h.fallback)}",
        file=sys.stderr,
        flush=True,
    )
    return h
