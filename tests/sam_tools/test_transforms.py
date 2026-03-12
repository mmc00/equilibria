from __future__ import annotations

import numpy as np
import pandas as pd

from equilibria.sam_tools.models import Sam
from equilibria.sam_tools.sam_transforms import (
    align_ti_to_gvt_j_on_sam,
    clear_agent_self_transfers_on_sam,
    collapse_margin_account_on_sam,
    collapse_tx_account_into_ti_on_sam,
    convert_exports_to_x_on_sam,
    create_x_block_on_sam,
    move_k_to_ji_on_sam,
    move_l_to_ji_on_sam,
    move_margin_to_i_margin_on_sam,
    move_nonmargin_i_to_ji_on_sam,
    move_row_factor_inflows_to_owning_agents_on_sam,
    move_tx_to_ti_on_i_on_sam,
    normalize_pep_accounts_on_sam,
)


def _build_pep_like_matrix() -> Sam:
    keys = [
        ("RAW", "A-AGR"),
        ("RAW", "C-AGR"),
        ("K", "cap"),
        ("L", "usk"),
        ("J", "agr"),
        ("I", "agr"),
        ("I", "ser"),
        ("AG", "ti"),
        ("AG", "gvt"),
        ("AG", "tx"),
        ("AG", "hrp"),
        ("AG", "hrr"),
        ("X", "agr"),
        ("MARG", "MARG"),
        ("AG", "row"),
    ]
    matrix = np.zeros((len(keys), len(keys)), dtype=float)
    df = pd.DataFrame(matrix, index=pd.MultiIndex.from_tuples(keys), columns=pd.MultiIndex.from_tuples(keys))
    sam = Sam(dataframe=df)
    return sam


def test_move_margin_to_i_margin_relocates_values() -> None:
    sam = _build_pep_like_matrix()
    df = sam.to_dataframe()
    df.loc[("MARG", "MARG"), ("I", "agr")] = 10.0
    df.loc[("I", "agr"), ("AG", "row")] = 5.0
    sam.replace_dataframe(df)

    result = move_margin_to_i_margin_on_sam(sam, {"margin_commodity": "ser"})
    assert result["moved_total"] == 10.0
    assert sam.to_dataframe().loc[("MARG", "MARG"), ("I", "agr")] == 0.0
    assert sam.to_dataframe().loc[("I", "ser"), ("I", "agr")] == 10.0


def test_move_k_to_ji_reallocates_capital() -> None:
    sam = _build_pep_like_matrix()
    df = sam.to_dataframe()
    df.loc[("K", "cap"), ("I", "agr")] = 20.0
    df.loc[("J", "agr"), ("I", "agr")] = 30.0
    sam.replace_dataframe(df)

    result = move_k_to_ji_on_sam(sam, {"commodity_to_sector": {"agr": "agr"}})
    assert result["moved_total"] == 20.0
    assert sam.to_dataframe().loc[("K", "cap"), ("I", "agr")] == 0.0
    assert sam.to_dataframe().loc[("J", "agr"), ("I", "agr")] == 50.0


def test_move_l_to_ji_reallocates_labor() -> None:
    sam = _build_pep_like_matrix()
    df = sam.to_dataframe()
    df.loc[("L", "usk"), ("I", "agr")] = 10.0
    df.loc[("J", "agr"), ("I", "agr")] = 5.0
    sam.replace_dataframe(df)

    result = move_l_to_ji_on_sam(sam, {"commodity_to_sector": {"agr": "agr"}})
    assert result["moved_total"] == 10.0
    assert sam.to_dataframe().loc[("L", "usk"), ("I", "agr")] == 0.0
    assert sam.to_dataframe().loc[("J", "agr"), ("I", "agr")] == 15.0


def test_move_tx_to_ti_moves_tax_on_commodity() -> None:
    sam = _build_pep_like_matrix()
    df = sam.to_dataframe()
    df.loc[("AG", "tx"), ("I", "agr")] = 7.0
    sam.replace_dataframe(df)
    result = move_tx_to_ti_on_i_on_sam(sam)
    assert result["moved_total"] == 7.0
    assert sam.to_dataframe().loc[("AG", "tx"), ("I", "agr")] == 0.0
    assert sam.to_dataframe().loc[("AG", "ti"), ("I", "agr")] == 7.0


def test_collapse_margin_account_moves_row_and_clears_column() -> None:
    sam = _build_pep_like_matrix()
    df = sam.to_dataframe()
    df.loc[("MARG", "MARG"), ("I", "agr")] = 10.0
    df.loc[("I", "ser"), ("MARG", "MARG")] = 10.0
    sam.replace_dataframe(df)

    result = collapse_margin_account_on_sam(sam, {"margin_commodity": "ser"})

    assert result["moved_total"] == 10.0
    assert result["cleared_column_total"] == 10.0
    assert sam.to_dataframe().loc[("MARG", "MARG"), ("I", "agr")] == 0.0
    assert sam.to_dataframe().loc[("I", "ser"), ("MARG", "MARG")] == 0.0
    assert sam.to_dataframe().loc[("I", "ser"), ("I", "agr")] == 10.0


def test_collapse_tx_account_merges_row_and_column_into_ti() -> None:
    sam = _build_pep_like_matrix()
    df = sam.to_dataframe()
    df.loc[("AG", "tx"), ("I", "agr")] = 7.0
    df.loc[("AG", "gvt"), ("AG", "tx")] = 7.0
    sam.replace_dataframe(df)

    result = collapse_tx_account_into_ti_on_sam(sam)

    assert result["row_moved_total"] == 7.0
    assert result["column_moved_total"] == 7.0
    assert sam.to_dataframe().loc[("AG", "tx"), ("I", "agr")] == 0.0
    assert sam.to_dataframe().loc[("AG", "gvt"), ("AG", "tx")] == 0.0
    assert sam.to_dataframe().loc[("AG", "ti"), ("I", "agr")] == 7.0
    assert sam.to_dataframe().loc[("AG", "gvt"), ("AG", "ti")] == 7.0


def test_align_ti_to_gvt_j_reassigns_columns() -> None:
    sam = _build_pep_like_matrix()
    df = sam.to_dataframe()
    df.loc[("AG", "ti"), ("J", "agr")] = 12.0
    sam.replace_dataframe(df)
    result = align_ti_to_gvt_j_on_sam(sam)
    assert result["moved_total"] == 12.0
    assert sam.to_dataframe().loc[("AG", "ti"), ("J", "agr")] == 0.0
    assert sam.to_dataframe().loc[("AG", "gvt"), ("J", "agr")] == 12.0


def test_move_nonmargin_i_to_ji_moves_invalid_i_to_i_support() -> None:
    sam = _build_pep_like_matrix()
    df = sam.to_dataframe()
    df.loc[("I", "agr"), ("I", "agr")] = 8.0
    df.loc[("J", "agr"), ("I", "agr")] = 2.0
    sam.replace_dataframe(df)

    result = move_nonmargin_i_to_ji_on_sam(
        sam,
        {"commodity_to_sector": {"agr": "agr"}, "margin_commodities": ["ser"]},
    )

    assert result["moved_total"] == 8.0
    assert sam.to_dataframe().loc[("I", "agr"), ("I", "agr")] == 0.0
    assert sam.to_dataframe().loc[("J", "agr"), ("I", "agr")] == 10.0


def test_move_row_factor_inflows_to_owning_agents_reallocates_by_weights() -> None:
    sam = _build_pep_like_matrix()
    df = sam.to_dataframe()
    df.loc[("L", "usk"), ("AG", "row")] = 12.0
    df.loc[("AG", "hrp"), ("L", "usk")] = 3.0
    df.loc[("AG", "hrr"), ("L", "usk")] = 1.0
    sam.replace_dataframe(df)

    result = move_row_factor_inflows_to_owning_agents_on_sam(sam)

    assert result["moved_total"] == 12.0
    assert sam.to_dataframe().loc[("L", "usk"), ("AG", "row")] == 0.0
    assert sam.to_dataframe().loc[("AG", "hrp"), ("AG", "row")] == 9.0
    assert sam.to_dataframe().loc[("AG", "hrr"), ("AG", "row")] == 3.0


def test_clear_agent_self_transfers_zeroes_only_requested_agents() -> None:
    sam = _build_pep_like_matrix()
    df = sam.to_dataframe()
    df.loc[("AG", "gvt"), ("AG", "gvt")] = 11.0
    df.loc[("AG", "row"), ("AG", "row")] = 4.0
    df.loc[("AG", "hrp"), ("AG", "hrp")] = 2.0
    sam.replace_dataframe(df)

    result = clear_agent_self_transfers_on_sam(sam, {"agents": ["gvt", "row"]})

    assert result["removed_total"] == 15.0
    assert sam.to_dataframe().loc[("AG", "gvt"), ("AG", "gvt")] == 0.0
    assert sam.to_dataframe().loc[("AG", "row"), ("AG", "row")] == 0.0
    assert sam.to_dataframe().loc[("AG", "hrp"), ("AG", "hrp")] == 2.0


def test_normalize_pep_accounts_builds_j_i_structure() -> None:
    sam = _build_pep_like_matrix()
    raw_df = sam.to_dataframe()
    raw_df.loc[("RAW", "A-AGR"), ("RAW", "C-AGR")] = 5.0
    sam.replace_dataframe(raw_df)
    result = normalize_pep_accounts_on_sam(sam)

    assert result["pep_accounts"] >= 6
    assert ("J", "agr") in sam.row_keys
    assert ("I", "agr") in sam.row_keys


def test_normalize_pep_accounts_maps_savings_rows_to_investment() -> None:
    keys = [
        ("RAW", "A-AGR"),
        ("RAW", "C-AGR"),
        ("RAW", "HRP"),
        ("RAW", "ROW"),
        ("RAW", "S-HH"),
        ("RAW", "S-ROW"),
    ]
    df = pd.DataFrame(
        np.zeros((len(keys), len(keys)), dtype=float),
        index=pd.MultiIndex.from_tuples(keys),
        columns=pd.MultiIndex.from_tuples(keys),
    )
    df.loc[("RAW", "S-HH"), ("RAW", "HRP")] = 10.0
    df.loc[("RAW", "S-ROW"), ("RAW", "ROW")] = 5.0
    sam = Sam(dataframe=df)

    normalize_pep_accounts_on_sam(sam)
    normalized = sam.to_dataframe()

    assert normalized.loc[("OTH", "INV"), ("AG", "hrp")] == 10.0
    assert normalized.loc[("OTH", "INV"), ("AG", "row")] == 5.0


def test_normalize_pep_accounts_maps_commodity_inv_and_vstk_flows() -> None:
    keys = [
        ("RAW", "C-AGR"),
        ("RAW", "INV"),
        ("RAW", "VSTK"),
    ]
    df = pd.DataFrame(
        np.zeros((len(keys), len(keys)), dtype=float),
        index=pd.MultiIndex.from_tuples(keys),
        columns=pd.MultiIndex.from_tuples(keys),
    )
    df.loc[("RAW", "C-AGR"), ("RAW", "INV")] = 12.0
    df.loc[("RAW", "C-AGR"), ("RAW", "VSTK")] = 3.0
    sam = Sam(dataframe=df)

    normalize_pep_accounts_on_sam(sam)
    normalized = sam.to_dataframe()

    assert normalized.loc[("I", "agr"), ("OTH", "INV")] == 12.0
    assert normalized.loc[("I", "agr"), ("OTH", "VSTK")] == 3.0
