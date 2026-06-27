"""Four-way classification of a cascade-tool result. GOVERNING PRINCIPLE: no error
branch maps to clean. clean = 'measured, found nothing'; an error maps to
dirty / stop / continue-but-visible / continue-but-record."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

CONTINUE = "continue"
EXPLAIN_STOP = "explain_stop"
VACUOUS_CONTINUE = "vacuous_continue"   # nl_compare #23 vacuity — visible, not clean
UPSTREAM_STOP = "upstream_stop"         # no_convergence — seed layers would be noise
TOOL_BROKEN_CONTINUE = "tool_broken_continue"
BLOCKING_STOP = "blocking_stop"         # unknown/gdx_not_found — safe-default blocking


@dataclass
class LayerResult:
    name: str
    status: str
    error_kind: Optional[str]
    headline: str
    action: str
    exit_code: int
    raw: dict


def classify(name: str, payload: dict, exit_code: int) -> LayerResult:
    status = payload.get("status", "error")
    kind = (payload.get("meta") or {}).get("error_kind")
    headline = payload.get("headline", "")
    if status == "clean":
        action = CONTINUE
    elif status == "dirty":
        action = EXPLAIN_STOP
    elif status == "error" and kind == "no_common_constraints":
        action = VACUOUS_CONTINUE
    elif status == "error" and kind == "no_convergence":
        action = UPSTREAM_STOP
    elif status == "error" and kind == "exception":
        action = TOOL_BROKEN_CONTINUE
    else:
        # error with unknown/other kind (incl. gdx_not_found, or a status the tool
        # invented) -> blocking by safe default. NEVER clean.
        action = BLOCKING_STOP
    return LayerResult(name, status, kind, headline, action, exit_code, payload)
