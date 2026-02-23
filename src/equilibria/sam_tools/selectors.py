"""Selector and key parsing helpers for SAM transformations."""

from __future__ import annotations

from typing import Any


def norm_text(value: Any) -> str:
    """Normalize one selector token by trimming surrounding whitespace."""
    return str(value).strip()


def norm_text_lower(value: Any) -> str:
    """Return lower-case normalized selector text."""
    return norm_text(value).lower()


def parse_key_spec(spec: Any, field_name: str) -> tuple[str, str]:
    """Parse one exact key spec into ``(category, element)``."""
    if isinstance(spec, (list, tuple)) and len(spec) == 2:
        return (norm_text(spec[0]), norm_text(spec[1]))

    if isinstance(spec, dict):
        cat = spec.get("cat")
        elem = spec.get("elem")
        if cat is None or elem is None:
            raise ValueError(f"{field_name} dict requires 'cat' and 'elem'")
        return (norm_text(cat), norm_text(elem))

    if isinstance(spec, str):
        text = norm_text(spec)
        for sep in (".", ":", "/"):
            if sep in text:
                left, right = text.split(sep, 1)
                return (norm_text(left), norm_text(right))

    raise ValueError(f"Invalid key spec for {field_name}: {spec!r}")


def parse_selector(spec: Any) -> tuple[str, str]:
    """Parse wildcard selectors such as ``AG.*`` or ``*.agr``."""
    if spec is None:
        return ("*", "*")

    if isinstance(spec, (list, tuple)) and len(spec) == 2:
        return (norm_text(spec[0]), norm_text(spec[1]))

    if isinstance(spec, dict):
        return (norm_text(spec.get("cat", "*")), norm_text(spec.get("elem", "*")))

    if isinstance(spec, str):
        text = norm_text(spec)
        if text in {"*", "*.*", "*:*"}:
            return ("*", "*")
        for sep in (".", ":", "/"):
            if sep in text:
                left, right = text.split(sep, 1)
                return (norm_text(left), norm_text(right))
        return (norm_text(text), "*")

    raise ValueError(f"Invalid selector spec: {spec!r}")


def matches_selector(key: tuple[str, str], selector: tuple[str, str]) -> bool:
    """Return ``True`` when key matches category/element wildcard selector."""
    cat, elem = norm_text_lower(key[0]), norm_text_lower(key[1])
    sel_cat, sel_elem = norm_text_lower(selector[0]), norm_text_lower(selector[1])
    cat_ok = sel_cat == "*" or cat == sel_cat
    elem_ok = sel_elem == "*" or elem == sel_elem
    return cat_ok and elem_ok


def indices_for_selector(
    keys: list[tuple[str, str]],
    selector_spec: Any,
    axis_name: str,
) -> list[int]:
    """Return indices of keys selected by one wildcard expression."""
    selector = parse_selector(selector_spec)
    indices = [i for i, key in enumerate(keys) if matches_selector(key, selector)]
    if not indices:
        raise ValueError(f"Selector matched no {axis_name} keys: {selector_spec!r}")
    return indices


def index_for_key(
    keys: list[tuple[str, str]],
    key_spec: Any,
    field_name: str,
) -> int:
    """Return index of one exact key specification."""
    target = parse_key_spec(key_spec, field_name)
    target_norm = (norm_text_lower(target[0]), norm_text_lower(target[1]))
    for idx, key in enumerate(keys):
        key_norm = (norm_text_lower(key[0]), norm_text_lower(key[1]))
        if key_norm == target_norm:
            return idx
    raise ValueError(f"Key not found for {field_name}: {target}")
