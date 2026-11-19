"""Compare what a notebook or a doc page prints with what it printed on ``main``.

The golden (``tests/examples/golden/pr3c_outputs_main.json``) holds every stdout line
of the old-driver run. The core driver prints its solve differently (codes
``"optimal"`` instead of ``1``; no Pyomo bound warnings on the Johansen path), so
the comparison keeps only the results. Dropped: the solver chatter, the ``codes:``
line (each notebook asserts its codes itself), Pyomo's warnings, the body of the
residual panel, and the notebooks' "largest differences" table: its rows are the
near-zero cells (``qxs`` ~1e-5) that move in the 4th digit between Linux and the
Mac, sorted by differences that tie at 0.000% (the CI listed other cells), while
the headline (cells, % within, largest difference) does not move. Kept, with the solve code read as the old one: a printed code,
a ``{'code': ..., 'residual': ...}`` dict (its residual value masked) and the
panel's title, which says whether the solve succeeded. Every kept line must have
the same text and the same numbers, a number within ``rel`` (another CPU moves the
last bits).
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
    # Pyomo's W1002: the old driver's Johansen path set pwmg to 0, below its bound
    "WARNING (W1002)",
    "outside the bounds",
    "pyomo.readthedocs.io/en/stable/errors.html",
)
# the residual panel's body and footer (the footer carries the version); its
# title line (╔) is a result
_BOX = ("║", "╚")
_PANEL_TITLE = re.compile(r"^╔═*\s*(.*?)\s*═*╗$")
# a row of the "largest differences" table ("qxs   post  USA,SER,USA ...") and its
# header ("variable   period  cell ...")
_TABLE_ROW = re.compile(r"^\S+\s+(?:pre|post|period)\s{2}")
_RESIDUAL = re.compile(r"('residual': )[^,}]+")
_NUMBER = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")


def _as_old(line: str) -> str:
    """A kept line with the core's solve code read as PATH's 1, the residual value
    masked and the panel title without its padding."""
    if line == "optimal":
        return "1"
    title = _PANEL_TITLE.match(line)
    if title:
        return "panel: " + title.group(1).replace("code optimal", "code 1")
    return _RESIDUAL.sub(r"\1R", line.replace("'code': 'optimal'", "'code': 1"))


def results(lines: list[str]) -> list[str]:
    """The lines that carry a result: not empty, not chatter, not the panel's body."""
    return [
        _as_old(line)
        for line in lines
        if line.strip()
        and not line.lstrip().startswith(_BOX)
        and not _TABLE_ROW.match(line)
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
