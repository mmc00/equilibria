"""High-level shock helpers for GTAP parameter containers.

These mirror the shocks the GAMS reference scripts apply on the .l levels
of tax variables, but operate directly on a `GTAPParameters` instance so
the model can be (re)built afterwards without re-loading the GDX.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Iterable, Literal, Optional

from equilibria.templates.gtap.gtap_parameters import GTAPParameters

ShockMode = Literal["tm_pct", "pct", "set"]


def apply_tariff_shock(
    params: GTAPParameters,
    value: float,
    *,
    mode: ShockMode = "tm_pct",
    commodities: Optional[Iterable[str]] = None,
    sources: Optional[Iterable[str]] = None,
    destinations: Optional[Iterable[str]] = None,
    inplace: bool = False,
) -> GTAPParameters:
    """Apply an import-tariff shock to `params.taxes.imptx`.

    The container is indexed by `(source_region, commodity, destination_region)`
    and represents the import-tax wedge `tm`. The shock is applied to every
    matching key; pass `commodities` / `sources` / `destinations` to restrict
    the shock to a subset.

    Modes:
      * ``"tm_pct"`` (default, GAMS-equivalent): scales the *power* of the
        tariff by ``(1 + value)``: ``tm_new = (1 + tm_old) * (1 + value) - 1``.
        Matches GAMS ``tm.fx = tm.l * (1 + value)``.
      * ``"pct"``: scales the rate itself: ``tm_new = tm_old * (1 + value)``.
      * ``"set"``: replaces the rate: ``tm_new = value``.

    The diagonal `(r, i, r)` is always skipped because domestic sales carry
    no import tariff; multiplying the power formula on a stored zero would
    inject ``value`` as a real tariff.

    The legacy alias `params.taxes.rtms` (kept in sync by `GTAPParameters`)
    is updated alongside `imptx`.

    Returns the (possibly new) `GTAPParameters`. Pass ``inplace=True`` to
    mutate the input instead of deep-copying.
    """

    target = params if inplace else deepcopy(params)
    imptx = target.taxes.imptx
    rtms = target.taxes.rtms

    comm_filter = set(commodities) if commodities is not None else None
    src_filter = set(sources) if sources is not None else None
    dst_filter = set(destinations) if destinations is not None else None

    for key in list(imptx.keys()):
        if len(key) != 3:
            continue
        source, commodity, dest = key
        if source == dest:
            continue
        if comm_filter is not None and commodity not in comm_filter:
            continue
        if src_filter is not None and source not in src_filter:
            continue
        if dst_filter is not None and dest not in dst_filter:
            continue

        current = float(imptx[key])
        if mode == "tm_pct":
            updated = (1.0 + current) * (1.0 + float(value)) - 1.0
        elif mode == "pct":
            updated = current * (1.0 + float(value))
        elif mode == "set":
            updated = float(value)
        else:  # pragma: no cover - guarded by Literal
            raise ValueError(f"Unknown shock mode: {mode!r}")

        imptx[key] = updated
        if key in rtms:
            rtms[key] = updated

    return target


__all__ = ["apply_tariff_shock", "ShockMode"]
