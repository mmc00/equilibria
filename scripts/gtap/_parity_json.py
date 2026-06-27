#!/usr/bin/env python
"""Shared JSON-output contract for the parity-debug cascade tools.

PHASE 1 of the parity-orchestrator work (see CLAUDE.md "cascade de tools").
The six cascade tools (probe, drift_test, nl_compare, diff_mcp_pairing,
validate_reference, diff_calibration) all emit a SINGLE pure-JSON object to
stdout — no text mode, no `--json` flag. The orchestrator (Phase 2) reads the
`status` field as the binary signal: which cascade layer explains the gap.

Common schema, identical across all six tools:

    {
      "tool":     "probe",
      "dataset":  "gtap7_5x5",
      "period":   "shock" | "base" | "check" | null,
      "status":   "clean" | "dirty" | "error",
      "headline": "one sentence: what it found / confirmed clean",
      "violations": [
        {"entity": "eq_pvaeq", "index": ["USA","Svces"],
         "metric": "residual", "value": 2.83e-01}
      ],
      "meta": {}
    }

Contract:
  - status: the binary signal the orchestrator reads.
      clean = this layer does NOT explain the gap
      dirty = it DOES explain the gap
      error = the tool could not run
  - violations: sorted by |value| DESCENDING; empty list when clean.
  - metric: names WHAT this tool measures (residual, drift_rel, jac_coef_diff,
            marginal_sign, ...) so scales are never confused across layers.
  - period: normalised to base|check|shock|null. Anything off that axis
            (altertax, multi-phase) goes into meta with period=null.
  - meta: free-form, tool-specific.
  - On failure the tool STILL emits JSON with status="error" and the reason in
    headline — NEVER a raw traceback to stdout. Errors are JSON too.

To guarantee stdout is exactly one JSON line no matter what the build/solve
prints underneath, wrap the whole tool body in `stdout_to_stderr()` (a context
manager that redirects sys.stdout -> stderr AND routes the logging module to
stderr) and only let the final `emit(...)` write to the real stdout.

Usage skeleton (every tool follows this):

    from _parity_json import stdout_to_stderr, emit, run_tool

    def _run(args):
        # ... do the work, returns (status, headline, violations, meta, period)
        return dict(status=..., headline=..., violations=[...], meta={...},
                    period=...)

    def main():
        args = ap.parse_args()
        run_tool("probe", args.dataset, lambda: _run(args),
                 period_hint=...)   # period_hint optional; _run may override

`run_tool` runs the callable inside stdout_to_stderr(), catches any exception,
and guarantees a single JSON line on the real stdout in every path.
"""
from __future__ import annotations

import contextlib
import io
import json
import logging
import math
import sys
import traceback
from typing import Any, Callable, Optional

# Periods the orchestrator iterates over. Anything else -> None (goes to meta).
_CANONICAL_PERIODS = {"base", "check", "shock"}


def normalize_period(period: Optional[str]) -> Optional[str]:
    """Map a raw period/phase to the canonical axis or None.

    The orchestrator groups results by period, so this MUST be one of
    {base, check, shock} or None. Off-axis values (altertax, "base+shock",
    a phase list) collapse to None; the tool should stash the real value in
    meta.
    """
    if period is None:
        return None
    p = str(period).strip().lower()
    return p if p in _CANONICAL_PERIODS else None


def make_violation(entity: str, index: Any, metric: str, value: float) -> dict:
    """Build one violation entry.

    `index` is normalised to a JSON-friendly shape:
      - None        -> []
      - list/tuple  -> [str, ...]   (the common single-entity case)
      - dict        -> {k: [str,...]} passed through (e.g. a Jacobian CELL that
                       carries the sets of BOTH the eq and the var side)
      - scalar      -> [str(scalar)]
    """
    if index is None:
        idx: Any = []
    elif isinstance(index, dict):
        idx = {k: ([str(x) for x in v] if isinstance(v, (list, tuple))
                   else [str(v)]) for k, v in index.items()}
    elif isinstance(index, (list, tuple)):
        idx = [str(x) for x in index]
    else:
        idx = [str(index)]
    return {
        "entity": str(entity),
        "index": idx,
        "metric": str(metric),
        "value": float(value),
    }


def _sort_violations(violations: list[dict]) -> list[dict]:
    """Sort by |value| descending; tolerate non-numeric values (sink last)."""
    def _key(v: dict):
        try:
            return -abs(float(v.get("value", 0.0)))
        except (TypeError, ValueError):
            return float("inf")
    return sorted(violations, key=_key)


def build_payload(
    *,
    tool: str,
    dataset: Optional[str],
    period: Optional[str],
    status: str,
    headline: str,
    violations: Optional[list[dict]] = None,
    meta: Optional[dict] = None,
) -> dict:
    """Assemble the common-schema dict (sorted violations, normalised period)."""
    viols = _sort_violations(list(violations or []))
    return {
        "tool": tool,
        "dataset": dataset,
        "period": normalize_period(period),
        "status": status,
        "headline": headline,
        "violations": viols,
        "meta": meta or {},
    }


def _json_safe(obj):
    """Recursively replace non-finite floats (inf/-inf/nan) so the output is
    STRICT JSON (json.dumps would otherwise emit `Infinity`/`NaN`, which most
    parsers reject). A non-finite value becomes the string "inf"/"-inf"/"nan";
    callers that need the magnitude should expose it in a separate numeric field.
    """
    if isinstance(obj, float):
        if math.isinf(obj):
            return "inf" if obj > 0 else "-inf"
        if math.isnan(obj):
            return "nan"
        return obj
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    return obj


def emit(payload: dict, *, stream=None) -> None:
    """Write the payload as exactly one STRICT-JSON line to the REAL stdout.

    `stream` defaults to the genuine stdout captured at import time, so this
    works even while inside stdout_to_stderr() (which has swapped sys.stdout).
    allow_nan=False + _json_safe guarantee the line is parseable everywhere.
    """
    out = stream if stream is not None else _REAL_STDOUT
    out.write(json.dumps(_json_safe(payload), default=str, allow_nan=False) + "\n")
    out.flush()


# The genuine stdout, captured once at import — survives sys.stdout swapping.
_REAL_STDOUT = sys.stdout


class _StderrWriter(io.TextIOBase):
    """A thin proxy that forwards everything to the (current) sys.stderr."""

    def write(self, s):  # noqa: D401
        return sys.stderr.write(s)

    def flush(self):
        return sys.stderr.flush()


@contextlib.contextmanager
def stdout_to_stderr():
    """Redirect ALL stdout writes (and the logging module) to stderr.

    Inside this context:
      - bare `print(...)` and any library that writes to sys.stdout land on
        stderr (so stdout stays clean for the final JSON);
      - the root logging handler is pointed at stderr too (pyomo, equilibria,
        solver chatter).

    The real stdout is preserved in `_REAL_STDOUT` for `emit()` to use.
    """
    saved_stdout = sys.stdout
    root = logging.getLogger()
    saved_handlers = root.handlers[:]
    saved_level = root.level
    # Route logging to stderr for the duration.
    stderr_handler = logging.StreamHandler(sys.stderr)
    root.handlers = [stderr_handler]
    sys.stdout = _StderrWriter()
    try:
        yield
    finally:
        sys.stdout = saved_stdout
        root.handlers = saved_handlers
        root.setLevel(saved_level)


def run_tool(
    tool: str,
    dataset: Optional[str],
    work: Callable[[], dict],
    *,
    period_hint: Optional[str] = None,
) -> int:
    """Run `work()` under stdout capture and guarantee one JSON line out.

    `work` returns a dict with keys: status, headline, violations, meta,
    and optionally period (overrides period_hint). Any exception is turned
    into a status="error" payload with the message in headline and the
    traceback in meta. Returns a process exit code:
      0  -> status clean
      1  -> status dirty
      2  -> status error
    """
    payload: dict
    try:
        with stdout_to_stderr():
            result = work()
        period = result.get("period", period_hint)
        payload = build_payload(
            tool=tool,
            dataset=dataset,
            period=period,
            status=result["status"],
            headline=result["headline"],
            violations=result.get("violations"),
            meta=result.get("meta"),
        )
    except Exception as exc:  # noqa: BLE001 — errors must surface as JSON, not a crash
        # error_kind discriminates a genuine tool crash ("exception") from a
        # domain error the tool returns deliberately (e.g. "no_convergence");
        # the latter is set in the returned dict's meta by the tool itself.
        payload = build_payload(
            tool=tool,
            dataset=dataset,
            period=normalize_period(period_hint),
            status="error",
            headline=f"{type(exc).__name__}: {exc}",
            violations=[],
            meta={"error_kind": "exception",
                  "traceback": traceback.format_exc()},
        )
    emit(payload)
    return {"clean": 0, "dirty": 1, "error": 2}.get(payload["status"], 2)
