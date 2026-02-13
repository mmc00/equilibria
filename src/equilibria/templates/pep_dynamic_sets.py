"""
Dynamic set derivation for PEP templates from SAM data.
"""

from __future__ import annotations

from typing import Any

from equilibria.babel.gdx.reader import read_parameter_values


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


def derive_dynamic_sets_from_sam(sam_data: dict[str, Any]) -> dict[str, list[str]]:
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
    non_agents_upper = {"TD", "TI", "TM"} | set(l_upper) | set(k_upper)
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

    return {
        "H": h,
        "F": f,
        "K": k,
        "L": l,
        "J": j,
        "I": i,
        "I1": [x for x in i if x != "agr"],
        "AG": ag,
        "AGNG": agng,
        "AGD": agd,
    }
