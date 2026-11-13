"""The social accounting matrix of simple_open (Hosoe's stdcge layout).

A square CSV: the first row and column are the account labels. Besides goods and
factors, the SAM has six named accounts (indirect tax, tariff, household,
government, investment, rest of the world). Goods and factors are read from the
SAM, never named in code: the factors are the accounts that pay the household,
the goods are every other account.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

BALANCE_TOL = 1e-9


@dataclass(frozen=True)
class SpecialAccounts:
    """Labels of the six accounts that are neither goods nor factors."""

    idt: str = "IDT"
    trf: str = "TRF"
    hoh: str = "HOH"
    gov: str = "GOV"
    inv: str = "INV"
    ext: str = "EXT"

    def labels(self) -> tuple[str, ...]:
        return (self.idt, self.trf, self.hoh, self.gov, self.inv, self.ext)


@dataclass(frozen=True)
class SimpleOpenSAM:
    accounts: tuple[str, ...]
    matrix: np.ndarray
    goods: tuple[str, ...]
    factors: tuple[str, ...]
    special: SpecialAccounts

    def cell(self, row: str, col: str) -> float:
        """SAM(row, col): what ``col`` pays ``row``."""
        return float(self.matrix[self.accounts.index(row), self.accounts.index(col)])


def load_sam(
    path: str | Path, *, special: SpecialAccounts | None = None
) -> SimpleOpenSAM:
    """Read the SAM, check it balances, and find its goods and factors."""
    special = special or SpecialAccounts()
    with Path(path).open(newline="") as fh:
        rows = [r for r in csv.reader(fh) if r]
    header = tuple(c.strip() for c in rows[0][1:])
    labels = tuple(r[0].strip() for r in rows[1:])
    if labels != header:
        raise ValueError(f"{path}: row labels {labels} differ from columns {header}")
    missing = [s for s in special.labels() if s not in header]
    if missing:
        raise ValueError(f"{path}: the SAM has no account {missing}")
    matrix = np.array([[float(v) for v in r[1:]] for r in rows[1:]])
    gaps = matrix.sum(axis=1) - matrix.sum(axis=0)
    for name, gap in zip(header, gaps, strict=True):
        if abs(gap) > BALANCE_TOL * max(1.0, abs(matrix).max()):
            raise ValueError(
                f"{path}: account {name} does not balance: "
                f"row total - column total = {gap:g}"
            )
    hoh = header.index(special.hoh)
    rest = [a for a in header if a not in special.labels()]
    factors = tuple(a for a in rest if matrix[hoh, header.index(a)] > 0)
    goods = tuple(a for a in rest if a not in factors)
    return SimpleOpenSAM(header, matrix, goods, factors, special)
