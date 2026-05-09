"""High-level shock helpers for GTAP parameter containers.

`apply_shock` is the generic parent: it can shock *any* registered
container on a `GTAPParameters` instance — taxes, technical change,
endowments, elasticities — using a uniform ``target`` / ``mode`` /
``filters`` API.

`apply_tariff_shock` is a thin tariff-specific wrapper kept for backward
compatibility and ergonomics; new tax types only need to be added to the
`_REGISTRY` below.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, Literal, Optional, Tuple

from equilibria.templates.gtap.gtap_parameters import GTAPParameters

ShockMode = Literal["pct", "power", "set", "add", "mul", "tm_pct"]


@dataclass(frozen=True)
class _ContainerSpec:
    """Metadata describing how to locate and shock a parameter container.

    ``path`` is a dotted path from `GTAPParameters` to the dict-like
    container. ``dim_names`` labels each tuple-key axis with a filter
    name (e.g. ``("sources", "commodities", "destinations")``); pass any
    of these as keyword filters to `apply_shock`. ``skip_diagonal`` is a
    pair of dim positions that must differ; ``aliases`` are sibling
    container paths kept in sync after every write.
    """

    path: str
    dim_names: Tuple[str, ...]
    skip_diagonal: Optional[Tuple[int, int]] = None
    aliases: Tuple[str, ...] = field(default_factory=tuple)


_REGISTRY: Dict[str, _ContainerSpec] = {
    # Taxes -----------------------------------------------------------------
    "taxes.imptx": _ContainerSpec(
        path="taxes.imptx",
        dim_names=("sources", "commodities", "destinations"),
        skip_diagonal=(0, 2),
        aliases=("taxes.rtms",),
    ),
    "taxes.rtxs": _ContainerSpec(
        path="taxes.rtxs",
        dim_names=("sources", "commodities", "destinations"),
        skip_diagonal=(0, 2),
    ),
    "taxes.rto": _ContainerSpec(
        path="taxes.rto",
        dim_names=("regions", "sectors"),
    ),
    "taxes.rtf": _ContainerSpec(
        path="taxes.rtf",
        dim_names=("regions", "factors", "sectors"),
    ),
    "taxes.rtfd": _ContainerSpec(
        path="taxes.rtfd",
        dim_names=("regions", "commodities", "sectors"),
    ),
    "taxes.rtfi": _ContainerSpec(
        path="taxes.rtfi",
        dim_names=("regions", "commodities", "sectors"),
    ),
    "taxes.rtpd": _ContainerSpec(
        path="taxes.rtpd",
        dim_names=("regions", "commodities", "sectors"),
    ),
    "taxes.rtpi": _ContainerSpec(
        path="taxes.rtpi",
        dim_names=("regions", "commodities", "sectors"),
    ),
    "taxes.rtgd": _ContainerSpec(
        path="taxes.rtgd",
        dim_names=("regions", "commodities"),
    ),
    "taxes.rtgi": _ContainerSpec(
        path="taxes.rtgi",
        dim_names=("regions", "commodities"),
    ),
    # Add productivity / endowment targets here once the corresponding
    # `GTAPParameters` containers are wired up — e.g. `calibrated.aoall`,
    # `benchmark.evom`. The registry itself is the only thing to extend.
}


def list_shock_targets() -> list[str]:
    """Return every target name registered for `apply_shock`."""
    return sorted(_REGISTRY)


def _resolve_container(params: GTAPParameters, path: str):
    obj = params
    for part in path.split("."):
        obj = getattr(obj, part)
    return obj


def _apply_op(current: float, value: float, mode: ShockMode) -> float:
    if mode == "pct":
        return current * (1.0 + value)
    if mode == "power" or mode == "tm_pct":
        return (1.0 + current) * (1.0 + value) - 1.0
    if mode == "set":
        return value
    if mode == "add":
        return current + value
    if mode == "mul":
        return current * value
    raise ValueError(f"Unknown shock mode: {mode!r}")


def apply_shock(
    params: GTAPParameters,
    target: str,
    value: float,
    *,
    mode: ShockMode = "pct",
    inplace: bool = False,
    predicate: Optional[Callable[[tuple], bool]] = None,
    **filters: Optional[Iterable[str]],
) -> GTAPParameters:
    """Apply a generic shock to any registered parameter container.

    Args:
        params: Calibrated `GTAPParameters` to shock.
        target: Registered container name, e.g. ``"taxes.imptx"``,
            ``"taxes.rtf"``, ``"calibrated.aoall"``. See
            `list_shock_targets()` for the full list.
        value: Shock magnitude. Interpretation depends on ``mode``.
        mode: One of:

            * ``"pct"`` — scale rate: ``new = old * (1 + value)``
            * ``"power"`` — scale power: ``new = (1 + old) * (1 + value) - 1``
              (canonical for tariff/tax shocks à la GAMS ``tm.fx = tm.l*1.1``)
            * ``"set"`` — replace: ``new = value``
            * ``"add"`` — add: ``new = old + value``
            * ``"mul"`` — multiply: ``new = old * value``
            * ``"tm_pct"`` — alias of ``"power"`` (legacy)
        inplace: Mutate ``params`` instead of deep-copying.
        predicate: Optional ``(key) -> bool`` used as a final filter
            after the named dim filters resolve.
        **filters: Per-dimension restrictions. Valid filter names depend
            on the target's registered dim names — e.g. ``commodities=``
            for any target with a commodity axis, ``sources=`` for trade
            tax targets, ``regions=`` / ``factors=`` / ``sectors=`` for
            others. Unknown filter names raise ``TypeError``.

    Returns the (possibly new) `GTAPParameters`. Raises ``ValueError`` for
    unknown ``target`` or ``mode``.
    """

    if target not in _REGISTRY:
        raise ValueError(
            f"Unknown shock target: {target!r}. "
            f"Available: {list_shock_targets()}"
        )
    spec = _REGISTRY[target]

    unknown = set(filters) - set(spec.dim_names)
    if unknown:
        raise TypeError(
            f"Unknown filter(s) for target {target!r}: {sorted(unknown)}. "
            f"Valid filters: {list(spec.dim_names)}"
        )

    out = params if inplace else deepcopy(params)
    container = _resolve_container(out, spec.path)
    aliases = [_resolve_container(out, p) for p in spec.aliases]

    resolved_filters: list[Optional[set]] = [
        set(filters[name]) if filters.get(name) is not None else None
        for name in spec.dim_names
    ]

    for key in list(container.keys()):
        if not isinstance(key, tuple) or len(key) != len(spec.dim_names):
            continue
        if spec.skip_diagonal is not None:
            i, j = spec.skip_diagonal
            if key[i] == key[j]:
                continue
        if any(
            allowed is not None and component not in allowed
            for component, allowed in zip(key, resolved_filters)
        ):
            continue
        if predicate is not None and not predicate(key):
            continue

        updated = _apply_op(float(container[key]), float(value), mode)
        container[key] = updated
        for alias in aliases:
            if key in alias:
                alias[key] = updated

    return out


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
    """Tariff-specific wrapper around `apply_shock` for ``taxes.imptx``.

    Equivalent to::

        apply_shock(params, "taxes.imptx", value, mode=mode,
                    commodities=..., sources=..., destinations=...,
                    inplace=inplace)

    The default ``mode="tm_pct"`` matches GAMS ``tm.fx = tm.l * (1 + value)``
    (power scaling). The diagonal `(r, i, r)` is skipped automatically and
    the legacy alias `params.taxes.rtms` is kept in sync — both behaviours
    are encoded in the registry entry for ``taxes.imptx``.
    """

    return apply_shock(
        params,
        "taxes.imptx",
        value,
        mode=mode,
        inplace=inplace,
        commodities=commodities,
        sources=sources,
        destinations=destinations,
    )


__all__ = [
    "ShockMode",
    "apply_shock",
    "apply_tariff_shock",
    "list_shock_targets",
]
