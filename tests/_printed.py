"""Compare what a notebook or a doc page prints with what it printed on ``main``.

The golden (``tests/examples/golden/pr3c_outputs_main.json``) holds every stdout line
of the old-driver run. The core driver prints its solve differently (codes
``"optimal"`` instead of ``1``; no Pyomo bound warnings on the Johansen path), so
the comparison keeps only the results: the solver chatter, the ``codes:`` line,
Pyomo's warnings, the residual boxes and residual values are dropped; every other line must have the same text and the
same numbers, a number within ``rel`` (another CPU moves the last bits).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

GOLDEN = json.loads(
    (
        Path(__file__).parent / "examples" / "golden" / "pr3c_outputs_main.json"
    ).read_text(encoding="utf-8")
)

_CHATTER = (
    "structural matching:",
    "Building Jacobian structure",
    "expression-driven sparsity detection",
    "Jacobian structure built",
    "codes:",
    "'residual'",
    # Pyomo's W1002: the old driver's Johansen path set pwmg to 0, below its bound
    "WARNING (W1002)",
    "outside the bounds",
    "pyomo.readthedocs.io/en/stable/errors.html",
)
_BOX = ("╔", "║", "╚")
_NUMBER = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")


def results(lines: list[str]) -> list[str]:
    """The lines that carry a result: not empty, not chatter, not a residual box."""
    return [
        line
        for line in lines
        if line.strip()
        and not line.lstrip().startswith(_BOX)
        and not any(c in line for c in _CHATTER)
    ]


def mismatches(got: list[str], want: list[str], *, rel: float = 1e-9) -> list[str]:
    """Where ``results(got)`` leaves ``results(want)``: a line missing or extra, a
    line whose text differs, or a number off by more than ``rel`` (plus 1e-12
    absolute, for values printed near zero)."""
    g, w = results(got), results(want)
    if len(g) != len(w):
        return [f"{len(g)} result lines, golden has {len(w)}:\n{g}\n{w}"]
    out = []
    for i, (a, b) in enumerate(zip(g, w, strict=True)):
        if _NUMBER.sub("#", a) != _NUMBER.sub("#", b):
            out.append(f"line {i}: {a!r} != {b!r}")
            continue
        for x, y in zip(_NUMBER.findall(a), _NUMBER.findall(b), strict=True):
            fx, fy = float(x), float(y)
            if abs(fx - fy) > 1e-12 + rel * max(abs(fx), abs(fy)):
                out.append(f"line {i}: {x} != {y} in {b!r}")
    return out
