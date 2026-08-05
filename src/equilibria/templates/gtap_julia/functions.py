"""CES and CDE demand functions, ported verbatim from the Julia
ComputableGeneralEquilibriumHelpers package (ces.jl / cde.jl).

These are the algebraic core reused by every gtap_julia equation. Kept
byte-faithful to the Julia branches (σ==1, σ==0, σ>0, σ<0; the α==0 zeroing)
so the ported model matches the Julia oracle to machine precision.
"""

from __future__ import annotations

import numpy as np


def ces(y, p, alpha, sigma, gamma):
    """Vector of input demands for output ``y`` under CES with prices ``p``,
    distribution parameters ``alpha``, elasticity ``sigma``, scale ``gamma``.

    Mirrors Julia ``ces(y, p, α, σ, γ)``.
    """
    p = np.asarray(p, dtype=float)
    alpha = np.asarray(alpha, dtype=float)

    if sigma == 1:
        # Cobb-Douglas: y / (γ·∏(α/p)^α) · (α/p)
        ratio = alpha / p
        return y / (gamma * np.prod(ratio**alpha)) * ratio
    if sigma == 0:
        # Leontief: y·α/γ
        return y * alpha / gamma
    if sigma > 0:
        c = (1.0 / gamma) * np.sum((alpha**sigma) * (p ** (1.0 - sigma))) ** (
            1.0 / (1.0 - sigma)
        )
        to_ret = (y / gamma) * ((alpha * gamma * c) / p) ** sigma
        to_ret = np.asarray(to_ret, dtype=float)
        to_ret[alpha == 0] = 0.0
        return to_ret
    # sigma < 0: cells with α==0 use σ_adj=1 to avoid a negative-power blowup
    sigma_adj = np.full(len(p), sigma, dtype=float)
    sigma_adj[alpha == 0] = 1.0
    c = (1.0 / gamma) * np.sum((alpha**sigma_adj) * (p ** (1.0 - sigma))) ** (
        1.0 / (1.0 - sigma)
    )
    return (y / gamma) * ((alpha * gamma * c) / p) ** sigma_adj


def cde(alpha, beta, e, u, p, c):
    """CDE demand system: returns ``[demands…, sum_of_shares]``.

    Mirrors Julia ``cde(α, β, e, u, p, c)``.
    """
    alpha = np.asarray(alpha, dtype=float)
    beta = np.asarray(beta, dtype=float)
    e = np.asarray(e, dtype=float)
    p = np.asarray(p, dtype=float)

    weight = beta * u ** (e * (1.0 - alpha)) * (1.0 - alpha)
    denom = np.sum(weight * (p / c) ** (1.0 - alpha))
    x = weight * (p / c) ** (-alpha) / denom
    i = np.sum(beta * u ** ((1.0 - alpha) * e) * (p / c) ** (1.0 - alpha))
    return np.concatenate([x, [i]])
