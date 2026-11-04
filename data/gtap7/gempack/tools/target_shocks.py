"""RunGTAP's ``target% X from file F`` shocks as plain shocks (TBL53).

Only the RunGTAP GUI resolves ``target%``; GEMSIM and gtapuv7.exe do not. The
file is a SHOCKSv7 ``*_HAT``: ``TFD_HAT = 100*[1 - TFD_L]/TFD_L``, the shock
that removes the tax, with ``TFD_L = VDFP/VDFB`` the power of the tax
(C:/runGTAP375/shocksv7.tab; for nus333, 1/(1 + HAT/100) equals VDFP/VDFB in
every cell). ``target% X`` sets the ad valorem rate to X%, i.e. the power to
1 + X/100, so the shock to the power is known before solving:

    100 * [(1 + X/100) * (1 + HAT/100) - 1]

No closure swap is needed: the tax is exogenous and its target is a level the
data gives. ``check`` confirms it on the solution: in the updated data, the
power of every shocked cell is 1 + X/100 (maintainer, 2026-11-03).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

ENCODING = "latin-1"
TOL = 1e-5  # power of the tax in the updated data, Gragg-extrapolated

_SHOCK = re.compile(
    r"^(?P<lead>[ \t]*)Shock[ \t]+(?P<var>\w+)\((?P<idx>[^)]*)\)[ \t]*=[ \t]*"
    r"target%[ \t]+(?P<x>[-+\d.eE]+)[ \t]+from[ \t]+file[ \t]+(?P<file>[\w.\-]+)[ \t]*;"
    r"[ \t]*(?=\r?$)",  # the Burfisher .exp files are CRLF
    re.IGNORECASE | re.MULTILINE,
)
_DIMS = re.compile(r"^\s*([\d\s]+?)\s+real\s+row_order\s*;", re.IGNORECASE)
# the base flows whose ratio is the power of each purchase tax
POWER = {
    "tfd": ("VDFP", "VDFB"), "tfm": ("VMFP", "VMFB"),
    "tgd": ("VDGP", "VDGB"), "tgm": ("VMGP", "VMGB"),
    "tpd": ("VDPP", "VDPB"), "tpm": ("VMPP", "VMPB"),
    "tid": ("VDIP", "VDIB"), "tim": ("VMIP", "VMIB"),
}  # fmt: skip


class TargetError(ValueError):
    """A target% shock that cannot be translated faithfully."""


@dataclass(frozen=True)
class Target:
    var: str
    cell: tuple[str, ...]
    target: float  # the ad valorem rate, in %


def read_shk(path: Path) -> dict[tuple[str, ...], float]:
    """{(element, element[, element]): value} of a SHOCKSv7 ``.shk``.

    The names come from the file itself (``! %2=AGR MFG SER`` column headers,
    ``! %1=AGR`` row labels, ``(%1,%2,"USA")`` for the third axis), and the
    labelled cells must cover the declared dimensions exactly, in order.
    """
    dims: list[int] = []
    cells: dict[tuple[str, ...], float] = {}
    third: str | None = None
    cols: list[str] | None = None
    for line in path.read_text(encoding=ENCODING).splitlines():
        m = _DIMS.match(line)
        if m and not dims:
            dims = [int(x) for x in m.group(1).split()]
            continue
        m = re.search(r'\(%1,%2,"(\w+)"\)', line)
        if m:
            third = m.group(1)
            continue
        m = re.match(r"^\s*!\s*%2=(.*)$", line)
        if m:
            cols = m.group(1).split()
            continue
        m = re.search(r"!\s*%1=(\w+)", line)
        if not (m and cols):
            continue
        values = [float(t) for t in line.split("!", 1)[0].split()]
        if len(values) != len(cols):
            raise TargetError(f"{path.name}: a row with {len(values)} values, {len(cols)} columns")
        for col, v in zip(cols, values, strict=True):
            key = (m.group(1), col) + ((third,) if len(dims) == 3 else ())
            if key in cells:
                raise TargetError(f"{path.name}: cell {key} twice")
            cells[key] = v
    expected = 1
    for d in dims:
        expected *= d
    if len(dims) not in (2, 3) or len(cells) != expected:
        raise TargetError(f"{path.name}: {len(cells)} labelled cells for dimensions {dims}")
    return cells


def _cells(idx: str, cells: dict[tuple[str, ...], float]) -> list[tuple[tuple[str, ...], float]]:
    """The cells of ``cells`` that ``idx`` covers: ``"USA"`` fixes an axis, a set name spans it."""
    parts = [p.strip() for p in idx.split(",")]
    if len(parts) != len(next(iter(cells))):
        raise TargetError(f"({idx}): {len(parts)} indices for a {len(next(iter(cells)))}-D file")
    fixed = [p.strip('"') if p.startswith('"') else None for p in parts]
    for axis, name in enumerate(fixed):
        if name is not None and name not in {k[axis] for k in cells}:
            raise TargetError(f"({idx}): {name} is not in the file")
    return [(k, v) for k, v in cells.items()
            if all(f is None or f == c for f, c in zip(fixed, k, strict=True))]  # fmt: skip


def translate(exp: str, package: Path) -> tuple[str, list[Target]]:
    """``exp`` with every ``target%`` shock written out cell by cell, and the targets.

    An .exp without ``target%`` comes back unchanged, with no targets.
    """
    targets: list[Target] = []
    eol = "\r\n" if "\r\n" in exp else "\n"

    def one(m: re.Match[str]) -> str:
        var, x = m.group("var"), float(m.group("x"))
        if var.lower() not in POWER:
            raise TargetError(f"target% on {var}: only purchase taxes {sorted(POWER)}")
        cells = _cells(m.group("idx"), read_shk(package / m.group("file")))
        lead = m.group("lead")
        lines = [
            f"{lead}! Translated from `target% {m.group('x')} from file {m.group('file')}`: "
            f"the rate set to {x:g}%, shock = 100*[(1 + {x:g}/100)*(1 + HAT/100) - 1].",
            f"{lead}! Original: {m.group(0).strip()}",
        ]
        for cell, hat in cells:
            shock = 100 * ((1 + x / 100) * (1 + hat / 100) - 1) + 0.0
            names = ", ".join(f'"{c}"' for c in cell)
            lines.append(f"{lead}Shock {var}({names}) = {shock:.9g};")
            targets.append(Target(var.lower(), cell, x))
        return eol.join(lines)

    return _SHOCK.sub(one, exp), targets


def check(updated: Path, targets: list[Target]) -> list[str]:
    """One message per target whose power in ``updated`` is not 1 + X/100."""
    from equilibria.babel.har.reader import read_har

    har = read_har(updated)
    msgs = []
    for t in targets:
        num, den = (har[h] for h in POWER[t.var])
        pos = tuple(e.index(c) for e, c in zip(num.set_elements, t.cell, strict=True))
        power = float(num.array[pos]) / float(den.array[pos])
        if abs(power - (1 + t.target / 100)) > TOL:
            msgs.append(f"{t.var}{t.cell}: power {power:.7f} in {updated.name}, "
                        f"target {1 + t.target / 100:.7f}")  # fmt: skip
    return msgs
