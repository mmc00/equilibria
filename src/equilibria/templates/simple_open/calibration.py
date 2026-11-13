"""Calibration of simple_open from its SAM, formula by formula as stdcge.gms.

Lines 60-131 of Hosoe's stdcge (gamslib SEQ=276). Arrays are ordered by
``goods`` (and ``factors``); two-dimensional ones are (factor, sector) for
``beta_VA``/``FD`` and (good, sector) for ``a_X``/``XI``. Sectors and goods are
the same accounts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from equilibria.templates.simple_open.sam import SimpleOpenSAM


@dataclass(frozen=True)
class Calibration:
    goods: tuple[str, ...]
    factors: tuple[str, ...]
    values: dict[str, Any]
    base: dict[str, Any]


def _positive(sam: SimpleOpenSAM, what: str, flows: np.ndarray, label: str) -> None:
    for good, v in zip(sam.goods, flows, strict=True):
        if v <= 0:
            raise ValueError(
                f"good {good}: {what} is {v:g}; stdcge needs a positive {label} "
                "flow for every good (the CES/CET share would be 0 or undefined)"
            )


def calibrate(
    sam: SimpleOpenSAM, *, sigma: float = 2.0, psi: float = 2.0
) -> Calibration:
    """Every parameter and base level of stdcge, from the SAM alone."""
    s = sam.special
    goods, factors = sam.goods, sam.factors
    n = len(goods)

    def vec(row: str | None, col: str | None, keys: tuple[str, ...]) -> np.ndarray:
        return np.array([sam.cell(row or k, col or k) for k in keys], dtype=float)

    Td0 = sam.cell(s.gov, s.hoh)
    Tz0 = vec(s.idt, None, goods)
    Tm0 = vec(s.trf, None, goods)
    F0 = np.array([[sam.cell(h, j) for j in goods] for h in factors])
    Y0 = F0.sum(axis=0)
    X0 = np.array([[sam.cell(i, j) for j in goods] for i in goods])
    Z0 = Y0 + X0.sum(axis=0)
    M0 = vec(s.ext, None, goods)
    _positive(sam, "imports (EXT row)", M0, "import")
    tauz = Tz0 / Z0
    taum = Tm0 / M0
    Xp0 = vec(None, s.hoh, goods)
    FF = np.array([sam.cell(s.hoh, h) for h in factors])
    Xg0 = vec(None, s.gov, goods)
    Xv0 = vec(None, s.inv, goods)
    E0 = vec(None, s.ext, goods)
    _positive(sam, "exports (EXT column)", E0, "export")
    Q0 = Xp0 + Xg0 + Xv0 + X0.sum(axis=1)
    D0 = (1 + tauz) * Z0 - E0
    Sp0 = sam.cell(s.inv, s.hoh)
    Sg0 = sam.cell(s.inv, s.gov)
    Sf = sam.cell(s.inv, s.ext)

    sig = np.full(n, sigma)
    ps = np.full(n, psi)
    eta = (sig - 1) / sig
    phi = (ps + 1) / ps

    alpha = Xp0 / Xp0.sum()
    beta = F0 / F0.sum(axis=0)
    b = Y0 / np.prod(F0**beta, axis=0)
    ax = X0 / Z0
    ay = Y0 / Z0
    mu = Xg0 / Xg0.sum()
    lam = Xv0 / (Sp0 + Sg0 + Sf)
    arm = (1 + taum) * M0 ** (1 - eta) + D0 ** (1 - eta)
    deltam = (1 + taum) * M0 ** (1 - eta) / arm
    deltad = D0 ** (1 - eta) / arm
    gamma = Q0 / (deltam * M0**eta + deltad * D0**eta) ** (1 / eta)
    cet = E0 ** (1 - phi) + D0 ** (1 - phi)
    xie = E0 ** (1 - phi) / cet
    xid = D0 ** (1 - phi) / cet
    theta = Z0 / (xie * E0**phi + xid * D0**phi) ** (1 / phi)
    ssp = Sp0 / FF.sum()
    ssg = Sg0 / (Td0 + Tz0.sum() + Tm0.sum())
    taud = Td0 / FF.sum()

    values: dict[str, Any] = {
        "b_VA": b, "beta_VA": beta, "a_X": ax, "a_VA": ay,
        "theta_T": theta, "xi_E": xie, "xi_D": xid, "psi_T": ps, "tau_z": tauz,
        "gamma_A": gamma, "delta_M": deltam, "delta_D": deltad, "sigma_A": sig,
        "tau_m": taum, "pWe": np.ones(n), "pWm": np.ones(n), "Sf": Sf,
        "alpha_H": alpha, "ssp": ssp, "FF": FF,
        "tau_d": taud, "mu_G": mu, "ssg": ssg, "lambda_V": lam,
    }  # fmt: skip
    ones = np.ones(n)
    base: dict[str, Any] = {
        "VA": Y0, "FD": F0, "PVA": ones, "WF": np.ones(len(factors)),
        "XI": X0, "Z": Z0, "PZ": ones,
        "E": E0, "D": D0, "PD": ones,
        "Q": Q0, "M": M0, "PQ": ones,
        "PE": ones, "PM": ones, "EPS": 1.0,
        "XP": Xp0, "SP": Sp0, "UU": float(np.prod(Xp0**alpha)),
        "TD": Td0, "TZ": Tz0, "TM": Tm0, "XG": Xg0, "SG": Sg0,
        "XV": Xv0, "WALRAS": 0.0,
    }  # fmt: skip
    return Calibration(goods, factors, values, base)
