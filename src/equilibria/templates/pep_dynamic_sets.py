"""
Dynamic set derivation for PEP templates from SAM data.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from equilibria.babel.gdx.reader import read_parameter_values


DEFAULT_PEP_I1_EXCLUDED_MEMBERS = ("agr",)


def _unique_preserve(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for x in items:
        if x and x not in seen:
            seen.add(x)
            out.append(x)
    return out


def _ordered_with_preferred(items: list[str], preferred: list[str]) -> list[str]:
    items_set = set(items)
    ordered = [x for x in preferred if x in items_set]
    ordered.extend([x for x in items if x not in set(ordered)])
    return ordered


def _extract_sam_keys(sam_data: dict[str, Any]) -> list[tuple[str, str, str, str]]:
    """Extract SAM 4D keys as uppercase tuples from GDX or Excel-backed sam_data."""
    sam_matrix = sam_data.get("sam_matrix")
    if isinstance(sam_matrix, dict):
        return [tuple(str(p).upper() for p in key) for key in sam_matrix]

    try:
        values = read_parameter_values(sam_data, "SAM")
        return [tuple(str(p).upper() for p in key) for key in values]
    except Exception:
        return []


def normalize_i1_excluded_members(
    members: Iterable[str] | None = None,
) -> tuple[str, ...]:
    """Normalize runtime I1 exclusions to a stable lower-case tuple."""
    source = DEFAULT_PEP_I1_EXCLUDED_MEMBERS if members is None else members
    normalized: list[str] = []
    seen: set[str] = set()
    for member in source:
        label = str(member).strip().lower()
        if not label or label in seen:
            continue
        seen.add(label)
        normalized.append(label)
    return tuple(normalized)


def apply_i1_set_membership_overrides(
    sets: dict[str, list[str]],
    *,
    i1_excluded_members: Iterable[str] | None = None,
) -> dict[str, list[str]]:
    """Rebuild I1 from I using runtime-configured exclusions."""
    resolved = {key: list(value) for key, value in sets.items()}
    excluded = set(normalize_i1_excluded_members(i1_excluded_members))
    resolved["I1"] = [member for member in resolved.get("I", []) if member not in excluded]
    return resolved


def derive_dynamic_sets_from_sam(
    sam_data: dict[str, Any],
    *,
    i1_excluded_members: Iterable[str] | None = None,
) -> dict[str, list[str]]:
    """
    Build model sets dynamically from SAM support.

    Returns lower-case labels for compatibility with existing templates.
    """
    keys = _extract_sam_keys(sam_data)

    row_by_cat: dict[str, list[str]] = {}
    for row_cat, row_elem, _col_cat, _col_elem in keys:
        row_by_cat.setdefault(row_cat, []).append(row_elem)

    j_upper = _unique_preserve(row_by_cat.get("J", []))
    i_upper = _unique_preserve(row_by_cat.get("I", []))
    l_upper = _unique_preserve(row_by_cat.get("L", []))
    k_upper = _unique_preserve(row_by_cat.get("K", []))

    ag_raw_upper = _unique_preserve(row_by_cat.get("AG", []))
    non_agents_upper = {"TD", "TI", "TM", "TX"} | set(l_upper) | set(k_upper)
    ag_upper = [x for x in ag_raw_upper if x not in non_agents_upper]

    def lower(seq: list[str]) -> list[str]:
        return [x.lower() for x in seq]

    ag = lower(ag_upper)
    h = [x for x in ag if x not in {"firm", "gvt", "row"}]
    f = [x for x in ag if x not in set(h) and x not in {"gvt", "row"}]
    agng = [x for x in ag if x != "gvt"]
    agd = [x for x in ag if x != "row"]
    i = lower(i_upper)
    j = lower(j_upper)
    l = lower(l_upper)
    k = lower(k_upper)

    # Keep stable benchmark-oriented ordering when members are present.
    j = _ordered_with_preferred(j, ["agr", "ind", "ser", "adm"])
    i = _ordered_with_preferred(i, ["agr", "food", "othind", "ser", "adm"])
    l = _ordered_with_preferred(l, ["usk", "sk"])
    k = _ordered_with_preferred(k, ["cap", "land"])
    ag = _ordered_with_preferred(ag, ["hrp", "hup", "hrr", "hur", "firm", "gvt", "row"])
    h = [x for x in ag if x not in {"firm", "gvt", "row"}]
    f = [x for x in ag if x not in set(h) and x not in {"gvt", "row"}]
    agng = [x for x in ag if x != "gvt"]
    agd = [x for x in ag if x != "row"]

    sets = {
        "H": h,
        "F": f,
        "K": k,
        "L": l,
        "J": j,
        "I": i,
        "AG": ag,
        "AGNG": agng,
        "AGD": agd,
    }
    return apply_i1_set_membership_overrides(
        sets,
        i1_excluded_members=i1_excluded_members,
    )
