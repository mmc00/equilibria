"""Subprocess runner + per-period sweep for the cascade orchestrator."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Callable, Optional

from cascade_layers import LAYER_SPECS, build_cmd
from cascade_classify import (
    classify, LayerResult, CONTINUE,
    EXPLAIN_STOP, UPSTREAM_STOP, BLOCKING_STOP,
)

_STOP_ACTIONS = {EXPLAIN_STOP, UPSTREAM_STOP, BLOCKING_STOP}


def run_layer(argv: list[str], timeout: float) -> tuple[dict, int]:
    """Run a cascade tool; return (parsed JSON payload, exit code).

    Each tool guarantees one strict-JSON line on stdout. If stdout is not parseable
    (a hard crash before run_tool, or a timeout), synthesize an exception payload so
    the classifier still sees a structured 'tool broken' signal."""
    try:
        proc = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return ({"status": "error", "headline": "tool timed out",
                 "meta": {"error_kind": "exception", "reason": "timeout"},
                 "violations": []}, 124)
    line = (proc.stdout or "").strip().splitlines()
    payload = None
    for ln in reversed(line):  # the JSON is the last stdout line by contract
        try:
            payload = json.loads(ln)
            break
        except json.JSONDecodeError:
            continue
    if payload is None:
        return ({"status": "error", "headline": "non-JSON stdout from tool",
                 "meta": {"error_kind": "exception",
                          "stderr_tail": (proc.stderr or "")[-500:]},
                 "violations": []}, proc.returncode)
    return payload, proc.returncode


def sweep_period(dataset: str, period: str, gdx: Optional[Path], *,
                 stop: bool = True, timeout: float = 600.0,
                 runner: Callable[[list[str], float], tuple[dict, int]] = run_layer
                 ) -> list[LayerResult]:
    results: list[LayerResult] = []
    for layer in LAYER_SPECS:
        argv = build_cmd(layer, dataset, period, gdx)
        if argv is None:
            results.append(LayerResult(
                layer.name, "skipped", None,
                f"tool has no '{period}'", CONTINUE, 0, {}))
            continue
        payload, code = runner(argv, timeout)
        res = classify(layer.name, payload, code)
        results.append(res)
        if stop and res.action in _STOP_ACTIONS:
            break
    return results
