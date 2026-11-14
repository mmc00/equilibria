from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from equilibria.sam_tools.models import Sam
from equilibria.sam_tools.sam_transforms import (
    net_reciprocal_cells_on_sam,
    reroute_cell_via_on_sam,
)

HH, GVT, ROW, FIRM = ("AG", "hh"), ("AG", "gvt"), ("AG", "row"), ("AG", "firm")


def _balanced_sam() -> Sam:
    """Four agents with balanced circular flows plus ROW -> HH and HH <-> FIRM."""
    keys = [HH, GVT, ROW, FIRM]
    df = pd.DataFrame(
        np.zeros((4, 4)),
        index=pd.MultiIndex.from_tuples(keys),
        columns=pd.MultiIndex.from_tuples(keys),
    )
    df.loc[HH, ROW] = 30.0  # remittance: ROW pays HH
    df.loc[ROW, HH] = 30.0
    df.loc[GVT, HH] = 35.0
    df.loc[HH, GVT] = 20.0
    df.loc[HH, FIRM] = 50.0
    df.loc[FIRM, HH] = 35.0
    df.loc[FIRM, GVT] = 30.0
    df.loc[GVT, FIRM] = 15.0
    return Sam(dataframe=df)


def test_fixture_is_balanced() -> None:
    assert _balanced_sam().balance_status()["is_balanced"]


def test_reroute_moves_cell_through_via_and_keeps_balance() -> None:
    sam = _balanced_sam()
    totals_before = sam.to_dataframe().sum(axis=1)
    result = reroute_cell_via_on_sam(sam, {"row": HH, "col": ROW, "via": GVT})
    df = sam.to_dataframe()
    assert result["moved"] == 30.0
    assert df.loc[HH, ROW] == 0.0
    assert df.loc[GVT, ROW] == 30.0
    assert df.loc[HH, GVT] == 50.0
    assert result["balanced_before"] and result["balanced_after"]
    totals_after = df.sum(axis=1)
    assert totals_after[HH] == totals_before[HH]
    assert totals_after[GVT] == totals_before[GVT] + 30.0


def test_reroute_partial_value() -> None:
    sam = _balanced_sam()
    result = reroute_cell_via_on_sam(
        sam, {"row": HH, "col": ROW, "via": GVT, "value": 10.0}
    )
    assert result["cell_after"] == 20.0
    assert sam.balance_status()["is_balanced"]


def test_reroute_rejects_via_equal_to_row_or_missing_account() -> None:
    sam = _balanced_sam()
    with pytest.raises(ValueError):
        reroute_cell_via_on_sam(sam, {"row": HH, "col": ROW, "via": HH})
    with pytest.raises(ValueError):
        reroute_cell_via_on_sam(sam, {"row": HH, "col": ROW, "via": ("AG", "nope")})


def test_net_reciprocal_cells_keeps_net_flow_and_balance() -> None:
    sam = _balanced_sam()
    result = net_reciprocal_cells_on_sam(sam, {"a": HH, "b": FIRM})
    df = sam.to_dataframe()
    assert result["netted"] == 35.0
    assert df.loc[HH, FIRM] == 15.0
    assert df.loc[FIRM, HH] == 0.0
    assert result["balanced_after"]
