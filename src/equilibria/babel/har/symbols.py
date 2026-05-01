from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class HeaderArray:
    """Represents a single header array from a GEMPACK HAR file."""

    name: str
    coeff_name: str
    long_name: str
    array: np.ndarray
    set_names: list[str]
    set_elements: list[list[str]]

    @property
    def rank(self) -> int:
        return self.array.ndim

    @property
    def shape(self) -> tuple[int, ...]:
        return self.array.shape
