from __future__ import annotations

import numpy as np
import pandas as pd

from equilibria.sam_tools.models import Sam
from equilibria.sam_tools.sam_transforms import (
    align_ti_to_gvt_j_on_sam,
    convert_exports_to_x_on_sam,
    create_x_block_on_sam,
    move_k_to_ji_on_sam,
    move_l_to_ji_on_sam,
    move_margin_to_i_margin_on_sam,
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


def test_align_ti_to_gvt_j_reassigns_columns() -> None:
    sam = _build_pep_like_matrix()
    df = sam.to_dataframe()
    df.loc[("AG", "ti"), ("J", "agr")] = 12.0
    sam.replace_dataframe(df)
    result = align_ti_to_gvt_j_on_sam(sam)
    assert result["moved_total"] == 12.0
    assert sam.to_dataframe().loc[("AG", "ti"), ("J", "agr")] == 0.0
    assert sam.to_dataframe().loc[("AG", "gvt"), ("J", "agr")] == 12.0


def test_normalize_pep_accounts_builds_j_i_structure() -> None:
    sam = _build_pep_like_matrix()
    raw_df = sam.to_dataframe()
    raw_df.loc[("RAW", "A-AGR"), ("RAW", "C-AGR")] = 5.0
    sam.replace_dataframe(raw_df)
    result = normalize_pep_accounts_on_sam(sam)

    assert result["pep_accounts"] >= 6
    assert ("J", "agr") in sam.row_keys
    assert ("I", "agr") in sam.row_keys
