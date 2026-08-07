"""Native log-value calibration seeds — closed-form value shares.

The Python transcription of GlobalTradeAnalysisProjectModelV7.jl
``generate_calibration_inputs.jl`` CAL-I..VII: the σ_* seeds the inverted-closure
calibration solve consumes. Each σ is a pure VALUE SHARE of the benchmark, so it is
exact closed-form algebra (no solve) and must match the Julia calibrated point to
1e-9 (gate: tests/templates/gtap_logvalue/test_derived_params.py).

``GTAPParameters`` supplies sets and the bilateral value flows (vcif), keyed
(src, comm, dest). Factor value-added flows are read RAW from basedata.har (EVFP) via
``read_har`` — GTAPParameters does not carry EVFP, and the agent-price factor payment
is NOT reconstructible as evfb+ftrv-fbep in these data (evfb+ftrv-fbep = 52842 ≠ raw
EVFP 34003 for land/food/usa; Julia's σ_vff = raw EVFP share, verified 0.06992). Output
is emitted in the PORT order (σ_qxs → (comm, src, dest); σ_vff → (factor, act, reg)),
lower-cased, so the 1e-9 gate is apples-to-apples.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from equilibria.babel.har import read_har


def _lc(*xs: str) -> tuple[str, ...]:
    return tuple(x.lower() for x in xs)


def _raw_evfp(dataset_dir: Path, sets: Any) -> dict[tuple, float]:
    """Raw agent-price factor payment EVFP[factor,act,reg] from basedata.har.

    The HAR EVFP axes are (endw, acts, reg); emit keyed (factor, act, reg) lower-cased.
    """
    har = read_har(str(dataset_dir / "basedata.har"))
    ha = har.get("EVFP")
    if ha is None:
        raise ValueError(f"EVFP header missing in {dataset_dir / 'basedata.har'}")
    arr = np.asarray(ha.array)
    endw, acts, regs = list(sets.f), list(sets.a), list(sets.r)
    out: dict[tuple, float] = {}
    for fi, f in enumerate(endw):
        for ai, a in enumerate(acts):
            for ri, r in enumerate(regs):
                out[_lc(f, a, r)] = float(arr[fi, ai, ri])
    return out


def derived_shares(
    params: Any, sets: Any, dataset_dir: Path | str | None = None
) -> dict[str, dict[tuple, float]]:
    b = params.benchmark
    regs = list(sets.r)
    comms = list(sets.i)
    acts = list(sets.a)
    endw = list(sets.f)
    if dataset_dir is None:
        dataset_dir = getattr(params, "dataset_dir", None)
    dataset_dir = Path(dataset_dir) if dataset_dir else None
    out: dict[str, dict[tuple, float]] = {}

    # CAL-I: σ_qxs[c,s,d] = vcif[c,s,d] / Σ_s vcif[c,s,d]  (bilateral import value share).
    # GTAPParameters keys vcif as (src, comm, dest); emit (comm, src, dest) lower-cased.
    sig: dict[tuple, float] = {}
    for c in comms:
        for d in regs:
            tot = sum(float(b.vcif.get((s, c, d), 0.0) or 0.0) for s in regs)
            if tot <= 0.0:
                continue
            for s in regs:
                v = float(b.vcif.get((s, c, d), 0.0) or 0.0)
                if v != 0.0:
                    sig[_lc(c, s, d)] = v / tot
    out["sigma_qxs"] = sig

    # CAL-II: σ_vff[f,a,r] = evfp[f,a,r] / Σ_f evfp[f,a,r]  (factor value share in VA).
    # EVFP read raw from basedata.har (see module docstring), keyed (factor, act, reg).
    if dataset_dir is None:
        raise ValueError("derived_shares needs dataset_dir to read raw EVFP for σ_vff")
    evfp = _raw_evfp(dataset_dir, sets)
    sig = {}
    for a in acts:
        for r in regs:
            tot = sum(evfp.get(_lc(f, a, r), 0.0) for f in endw)
            if tot <= 0.0:
                continue
            for f in endw:
                v = evfp.get(_lc(f, a, r), 0.0)
                if v != 0.0:
                    sig[_lc(f, a, r)] = v / tot
    out["sigma_vff"] = sig

    return out
