"""MIP to SAM structural transformations."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from equilibria.sam_tools.models import Sam
from equilibria.sam_tools.sam_transforms import ensure_key
from equilibria.sam_tools.selectors import norm_text, norm_text_lower


DEFAULT_VA_FACTOR_SHARES = {"L": 0.65, "K": 0.35}

DEFAULT_FACTOR_TO_HOUSEHOLD_SHARES = {
    "L": {"hh": 0.95, "gvt": 0.05},
    "K": {"hh": 0.50, "firm": 0.45, "gvt": 0.05},
}

DEFAULT_TAX_RATES = {
    "production_tax": 0.10,
    "import_tariff": 0.05,
    "direct_tax": 0.15,
}


def _get_va_aggregate_row(sam: Sam) -> tuple[str, str] | None:
    """Find the VA aggregate row in the SAM."""
    for cat, elem in sam.row_keys:
        elem_lower = norm_text_lower(elem)
        if "va" in elem_lower and "aggregate" in elem_lower:
            return (cat, elem)
    return None


def _get_fd_columns(sam: Sam) -> list[tuple[str, str]]:
    """Get final demand columns (those with FD category or specific labels)."""
    fd_cols: list[tuple[str, str]] = []
    for cat, elem in sam.col_keys:
        cat_lower = norm_text_lower(cat)
        elem_lower = norm_text_lower(elem)
        if cat_lower == "fd" or elem_lower in {"hh", "gov", "gvt", "inv", "exp", "export"}:
            fd_cols.append((cat, elem))
    return fd_cols


def _get_sector_columns(sam: Sam) -> list[tuple[str, str]]:
    """Get sector (J) columns from SAM."""
    return [(cat, elem) for cat, elem in sam.col_keys if norm_text_lower(cat) == "j"]


def _get_commodity_keys(sam: Sam) -> list[tuple[str, str]]:
    """Get commodity (I) keys from SAM."""
    return [(cat, elem) for cat, elem in sam.row_keys if norm_text_lower(cat) == "i"]


def normalize_mip_accounts(sam: Sam, op: dict[str, Any]) -> dict[str, Any]:
    """
    Convert RAW MIP labels to initial PEP structure (J, I, VA, FD).

    Strategy: Keep it simple - just rename categories, don't restructure yet.
    The MIP already has the right structure (rows are commodities/VA/IMP,
    columns are sectors/final demand).
    """
    df = sam.to_dataframe()
    raw_keys = list(df.index)

    # Identify sector labels (those that appear in both rows and columns as commodities)
    sector_labels = set()
    for cat, elem in raw_keys:
        elem_lower = norm_text_lower(elem)
        # Skip special rows
        if any(x in elem_lower for x in ["va", "imp", "aggregate", "total"]):
            continue
        # Check if appears as column too
        if ("RAW", elem) in df.columns or any(norm_text_lower(e) == elem_lower for c, e in df.columns):
            sector_labels.add(norm_text_lower(elem))

    # Build new keys with proper categories
    new_keys: list[tuple[str, str]] = []
    old_to_new_row: dict[tuple[str, str], tuple[str, str]] = {}
    old_to_new_col: dict[tuple[str, str], tuple[str, str]] = {}

    # Process rows
    for cat, elem in raw_keys:
        elem_lower = norm_text_lower(elem)

        if "va" in elem_lower and "aggregate" in elem_lower:
            new_key = ("VA", "aggregate")
        elif "imp" in elem_lower and "total" in elem_lower:
            new_key = ("IMP", "total")
        elif elem_lower in sector_labels:
            new_key = ("I", norm_text(elem))  # Commodities
        else:
            new_key = ("I", norm_text(elem))  # Treat unknown as commodity

        old_to_new_row[(cat, elem)] = new_key
        if new_key not in new_keys:
            new_keys.append(new_key)

    # Process columns
    for cat, elem in df.columns:
        elem_lower = norm_text_lower(elem)

        if elem_lower in sector_labels:
            new_key = ("J", norm_text(elem))  # Sectors
        elif any(fd in elem_lower for fd in ["hh", "hogar", "gov", "gob", "inv", "inversion", "exp", "export"]):
            new_key = ("FD", norm_text(elem))  # Final demand
        else:
            # Could be a sector we haven't seen
            new_key = ("J", norm_text(elem))

        old_to_new_col[(cat, elem)] = new_key
        if new_key not in new_keys:
            new_keys.append(new_key)

    # Build new matrix - just copy values with new labels
    n = len(new_keys)
    new_matrix = np.zeros((n, n), dtype=float)
    key_index = {key: idx for idx, key in enumerate(new_keys)}

    for i, row_key in enumerate(raw_keys):
        for j, col_key in enumerate(df.columns):
            value = float(df.iloc[i, j])
            if abs(value) <= 1e-14:
                continue

            new_row = old_to_new_row.get(row_key)
            new_col = old_to_new_col.get(col_key)

            if new_row and new_col and new_row in key_index and new_col in key_index:
                new_matrix[key_index[new_row], key_index[new_col]] += value

    # Create new dataframe
    multi_index = pd.MultiIndex.from_tuples(new_keys)
    new_df = pd.DataFrame(new_matrix, index=multi_index, columns=multi_index)
    sam.replace_dataframe(new_df)

    return {
        "raw_accounts": len(raw_keys),
        "normalized_accounts": len(new_keys),
        "commodities": len([k for k in new_keys if k[0] == "I"]),
        "sectors": len([k for k in new_keys if k[0] == "J"]),
        "final_demand": len([k for k in new_keys if k[0] == "FD"]),
    }


def disaggregate_va_to_factors(sam: Sam, op: dict[str, Any]) -> dict[str, Any]:
    """
    Split VA aggregate row into L (labor) and K (capital) using shares.

    Before:
        ("VA", "aggregate") → ("J", "agr") = 100

    After:
        ("L", "labor") → ("J", "agr") = 65   # 100 * 0.65
        ("K", "capital") → ("J", "agr") = 35 # 100 * 0.35
    """
    shares = op.get("va_factor_shares", DEFAULT_VA_FACTOR_SHARES)

    # Validate shares sum to 1.0
    total_share = sum(shares.values())
    if abs(total_share - 1.0) > 1e-6:
        raise ValueError(f"va_factor_shares must sum to 1.0, got {total_share}")

    df = sam.to_dataframe()

    # Find VA row
    va_key = _get_va_aggregate_row(sam)
    if va_key is None:
        return {"factors_created": [], "total_va": 0.0, "error": "VA aggregate row not found"}

    if va_key not in df.index:
        return {"factors_created": [], "total_va": 0.0, "error": f"VA key {va_key} not in index"}

    # Create L and K rows
    l_key = ("L", "labor")
    k_key = ("K", "capital")

    ensure_key(sam, l_key)
    ensure_key(sam, k_key)

    df = sam.to_dataframe()
    total_va = 0.0

    # Distribute VA to L and K for each sector
    for col_key in sam.col_keys:
        if norm_text_lower(col_key[0]) != "j":
            continue

        va_value = float(df.loc[va_key, col_key])
        if abs(va_value) <= 1e-14:
            continue

        # Distribute to factors
        df.loc[l_key, col_key] = va_value * shares.get("L", 0.0)
        df.loc[k_key, col_key] = va_value * shares.get("K", 0.0)

        # Clear original VA
        df.loc[va_key, col_key] = 0.0
        total_va += va_value

    sam.replace_dataframe(df)

    return {
        "factors_created": ["L", "K"],
        "total_va": total_va,
        "shares": shares,
    }


def create_factor_income_distribution(sam: Sam, op: dict[str, Any]) -> dict[str, Any]:
    """
    Route factor income to institutions (households, firms, government).

    Before:
        ("L", "labor") → ("J", "agr") = 65  # Only production cost

    After:
        ("L", "labor") → ("J", "agr") = 65           # Keep production cost
        ("AG", "hh") → ("L", "labor") = 61.75        # 65 * 0.95 to households
        ("AG", "gvt") → ("L", "labor") = 3.25        # 65 * 0.05 taxes
    """
    factor_shares = op.get("factor_to_household_shares", DEFAULT_FACTOR_TO_HOUSEHOLD_SHARES)

    df = sam.to_dataframe()
    total_distributed = 0.0
    distribution_by_factor: dict[str, float] = {}

    # Get factor keys (L, K)
    factor_keys = [(cat, elem) for cat, elem in sam.row_keys if norm_text_lower(cat) in {"l", "k"}]

    for factor_key in factor_keys:
        factor_cat = factor_key[0]
        factor_total = 0.0

        # Calculate total factor income from all sectors
        for col_key in sam.col_keys:
            if norm_text_lower(col_key[0]) == "j":
                factor_total += float(df.loc[factor_key, col_key])

        if abs(factor_total) <= 1e-14:
            continue

        # Get distribution for this factor
        shares = factor_shares.get(factor_cat, {})

        # Create institution accounts and distribute
        for institution, share in shares.items():
            inst_key = ("AG", norm_text_lower(institution))
            ensure_key(sam, inst_key)

            df = sam.to_dataframe()
            income = factor_total * share

            # Add income from factor to institution
            df.loc[inst_key, factor_key] += income
            total_distributed += income

        distribution_by_factor[factor_cat] = factor_total

    sam.replace_dataframe(df)

    return {
        "total_distributed": total_distributed,
        "distribution_by_factor": distribution_by_factor,
        "institutions_created": list({inst for shares in factor_shares.values() for inst in shares.keys()}),
    }


def create_household_expenditure(sam: Sam, op: dict[str, Any]) -> dict[str, Any]:
    """
    Convert final demand "HH" from MIP to AG.hh → I flows.

    Before:
        ("I", "agr") → ("FD", "HH") = 20  # Final demand temporary

    After:
        ("AG", "hh") → ("I", "agr") = 20  # Household consumption
    """
    df = sam.to_dataframe()

    # Find HH final demand column
    hh_fd_key = None
    for cat, elem in sam.col_keys:
        elem_lower = norm_text_lower(elem)
        if "hh" in elem_lower or "hogar" in elem_lower:
            hh_fd_key = (cat, elem)
            break

    if hh_fd_key is None:
        return {"converted": 0, "total_expenditure": 0.0, "error": "HH final demand not found"}

    # Ensure AG.hh exists
    hh_key = ("AG", "hh")
    ensure_key(sam, hh_key)

    df = sam.to_dataframe()
    total_expenditure = 0.0

    # Transfer I → FD.HH flows to AG.hh → I
    for row_key in sam.row_keys:
        if norm_text_lower(row_key[0]) != "i":
            continue

        value = float(df.loc[row_key, hh_fd_key])
        if abs(value) <= 1e-14:
            continue

        # Move from (I, FD.HH) to (AG.hh, I)
        df.loc[row_key, hh_fd_key] = 0.0
        df.loc[hh_key, row_key] += value
        total_expenditure += value

    sam.replace_dataframe(df)

    return {
        "converted": 1,
        "total_expenditure": total_expenditure,
    }


def create_government_flows(sam: Sam, op: dict[str, Any]) -> dict[str, Any]:
    """
    Consolidate government flows (fiscal revenues + expenditure).

    Creates:
    - ("AG", "ti") → ("AG", "gvt"): Indirect taxes
    - ("AG", "tm") → ("AG", "gvt"): Import tariffs
    - ("AG", "gvt") → ("I", *): Government consumption
    """
    tax_rates = op.get("tax_rates", DEFAULT_TAX_RATES)

    df = sam.to_dataframe()

    # Ensure accounts exist
    gvt_key = ("AG", "gvt")
    ti_key = ("AG", "ti")
    tm_key = ("AG", "tm")

    for key in [gvt_key, ti_key, tm_key]:
        ensure_key(sam, key)

    df = sam.to_dataframe()

    # Calculate indirect taxes on production
    production_tax_rate = tax_rates.get("production_tax", 0.10)
    total_ti = 0.0

    for col_key in sam.col_keys:
        if norm_text_lower(col_key[0]) == "j":
            # Tax on sector output (sum of commodity inputs)
            sector_output = 0.0
            for row_key in sam.row_keys:
                if norm_text_lower(row_key[0]) == "i":
                    sector_output += float(df.loc[row_key, col_key])

            if sector_output > 1e-14:
                tax = sector_output * production_tax_rate
                df.loc[ti_key, col_key] += tax
                total_ti += tax

    # Calculate import tariffs
    import_tariff_rate = tax_rates.get("import_tariff", 0.05)
    total_tm = 0.0

    imp_key = None
    for cat, elem in sam.row_keys:
        if norm_text_lower(cat) == "imp":
            imp_key = (cat, elem)
            break

    if imp_key is not None and imp_key in df.index:
        for col_key in sam.col_keys:
            if norm_text_lower(col_key[0]) == "i":
                import_value = float(df.loc[imp_key, col_key])
                if import_value > 1e-14:
                    tariff = import_value * import_tariff_rate
                    df.loc[tm_key, col_key] += tariff
                    total_tm += tariff

    # Route tax revenues to government
    if total_ti > 1e-14:
        df.loc[ti_key, gvt_key] += total_ti

    if total_tm > 1e-14:
        df.loc[tm_key, gvt_key] += total_tm

    # Transfer government consumption from FD
    gov_fd_key = None
    for cat, elem in sam.col_keys:
        elem_lower = norm_text_lower(elem)
        if "gov" in elem_lower or "gob" in elem_lower or "gobierno" in elem_lower:
            gov_fd_key = (cat, elem)
            break

    total_gov_consumption = 0.0
    if gov_fd_key is not None:
        for row_key in sam.row_keys:
            if norm_text_lower(row_key[0]) != "i":
                continue

            value = float(df.loc[row_key, gov_fd_key])
            if abs(value) <= 1e-14:
                continue

            df.loc[row_key, gov_fd_key] = 0.0
            df.loc[gvt_key, row_key] += value
            total_gov_consumption += value

    sam.replace_dataframe(df)

    return {
        "total_ti": total_ti,
        "total_tm": total_tm,
        "total_gov_consumption": total_gov_consumption,
        "tax_rates": tax_rates,
    }


def create_row_account(sam: Sam, op: dict[str, Any]) -> dict[str, Any]:
    """
    Create rest-of-world (ROW) account for foreign trade.

    Before:
        ("IMP", "total") → ("I", "agr") = 15  # Imports
        ("I", "agr") → ("FD", "EXP") = 10   # Exports

    After:
        ("AG", "row") → ("I", "agr") = 15   # Import supply
        ("I", "agr") → ("AG", "row") = 10   # Export demand (moved to X later)
    """
    df = sam.to_dataframe()

    # Ensure ROW account exists
    row_key = ("AG", "row")
    ensure_key(sam, row_key)

    df = sam.to_dataframe()
    total_imports = 0.0
    total_exports = 0.0

    # Handle imports: IMP → I becomes AG.row → I
    imp_key = None
    for cat, elem in sam.row_keys:
        if norm_text_lower(cat) == "imp":
            imp_key = (cat, elem)
            break

    if imp_key is not None and imp_key in df.index:
        for col_key in sam.col_keys:
            if norm_text_lower(col_key[0]) == "i":
                value = float(df.loc[imp_key, col_key])
                if abs(value) <= 1e-14:
                    continue

                df.loc[imp_key, col_key] = 0.0
                df.loc[row_key, col_key] += value
                total_imports += value

    # Handle exports: I → FD.EXP becomes I → AG.row
    exp_fd_key = None
    for cat, elem in sam.col_keys:
        elem_lower = norm_text_lower(elem)
        if "exp" in elem_lower or "export" in elem_lower:
            exp_fd_key = (cat, elem)
            break

    if exp_fd_key is not None:
        for row_key in sam.row_keys:
            if norm_text_lower(row_key[0]) != "i":
                continue

            value = float(df.loc[row_key, exp_fd_key])
            if abs(value) <= 1e-14:
                continue

            df.loc[row_key, exp_fd_key] = 0.0
            df.loc[row_key, row_key] += value  # Will be moved to X.* later
            total_exports += value

    sam.replace_dataframe(df)

    return {
        "total_imports": total_imports,
        "total_exports": total_exports,
    }


def create_investment_account(sam: Sam, op: dict[str, Any]) -> dict[str, Any]:
    """
    Create savings-investment closure.

    Creates:
    - ("AG", *) → ("OTH", "inv"): Savings from institutions
    - ("I", *) → ("OTH", "inv"): Investment demand
    """
    df = sam.to_dataframe()

    # Ensure investment account exists
    inv_key = ("OTH", "inv")
    ensure_key(sam, inv_key)

    df = sam.to_dataframe()

    # Transfer investment demand from FD
    inv_fd_key = None
    for cat, elem in sam.col_keys:
        elem_lower = norm_text_lower(elem)
        if "inv" in elem_lower or "inversion" in elem_lower:
            inv_fd_key = (cat, elem)
            break

    total_investment = 0.0
    if inv_fd_key is not None:
        for row_key in sam.row_keys:
            if norm_text_lower(row_key[0]) != "i":
                continue

            value = float(df.loc[row_key, inv_fd_key])
            if abs(value) <= 1e-14:
                continue

            df.loc[row_key, inv_fd_key] = 0.0
            df.loc[row_key, inv_key] += value
            total_investment += value

    # Calculate institutional savings (residual balancing)
    # For each institution (AG.*), savings = income - expenditure
    total_savings = 0.0
    for row_key in sam.row_keys:
        if norm_text_lower(row_key[0]) != "ag":
            continue

        # Skip tax accounts
        elem_lower = norm_text_lower(row_key[1])
        if elem_lower in {"ti", "tm", "tx", "td"}:
            continue

        # Income = sum of row
        income = float(df.loc[row_key, :].sum())

        # Expenditure = sum of column
        col_idx = sam.col_keys.index(row_key) if row_key in sam.col_keys else -1
        if col_idx >= 0:
            expenditure = float(df.iloc[:, col_idx].sum())
        else:
            expenditure = 0.0

        savings = income - expenditure
        if abs(savings) > 1e-14:
            df.loc[row_key, inv_key] += savings
            total_savings += savings

    sam.replace_dataframe(df)

    return {
        "total_investment": total_investment,
        "total_savings": total_savings,
        "balance_diff": abs(total_investment - total_savings),
    }
