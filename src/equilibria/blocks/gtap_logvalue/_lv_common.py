"""Shared helpers for the log-value GTAP blocks: calibrated-point access (_get/_has,
same semantics as the port's), the CES demand form, and a seed-array builder."""

from __future__ import annotations

from typing import Any

import numpy as np


def _get(sol: dict[str, Any], name: str, idx: tuple[str, ...]) -> float:
    d = sol.get(name)
    if d is None:
        return 0.0
    if isinstance(d, dict):
        return float(d.get(idx, 0.0) or 0.0)
    return float(d)


def _has(sol: dict[str, Any], name: str, idx: tuple[str, ...]) -> bool:
    d = sol.get(name)
    if not isinstance(d, dict) or idx not in d:
        return False
    v = d[idx]
    return v == v and v != 0.0  # finite (not NaN) and non-zero


def _ces_input(y, prices, alphas, sigma, gamma, i):
    """CES demand for input i — verbatim from the port's _ces_input."""
    if sigma == 1:
        prod_term = 1.0
        for a, p in zip(alphas, prices, strict=True):
            prod_term = prod_term * (a / p) ** a
        return y / (gamma * prod_term) * (alphas[i] / prices[i])
    if sigma == 0:
        return y * alphas[i] / gamma
    c = (1.0 / gamma) * sum(
        (a**sigma) * (p ** (1.0 - sigma)) for a, p in zip(alphas, prices, strict=True)
    ) ** (1.0 / (1.0 - sigma))
    return (y / gamma) * ((alphas[i] * gamma * c) / prices[i]) ** sigma


def seed_array(sol: dict[str, Any], name: str, dims, setmap) -> np.ndarray:
    """np.ndarray seed for `name` over its (block-set) domain, from sol[name]."""
    axes = [setmap[d] for d in dims]
    arr = np.ones([len(ax) for ax in axes], dtype=float)
    d = sol.get(name, {})
    if isinstance(d, dict):
        idx_of = [{m: k for k, m in enumerate(ax)} for ax in axes]
        for key, val in d.items():
            if len(key) != len(axes):
                continue
            try:
                pos = tuple(idx_of[j][key[j]] for j in range(len(axes)))
            except KeyError:
                continue
            if val == val:  # skip NaN
                arr[pos] = val
    return arr
