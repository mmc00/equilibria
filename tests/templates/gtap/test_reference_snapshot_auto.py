"""EQUILIBRIA_GTAP_REFERENCE_SNAPSHOT=auto seeds from the recorded 9x10 GAMS reference.

The reference is data/gtap7/9x10/gams_mcp_gtap_ifsub1_fbep-zero/out.gdx (comp.gms,
the GAMS distribution's own dataset); the seed is its base period.
"""

from __future__ import annotations

import contextlib
import io
from types import SimpleNamespace
from typing import Any, cast

import pytest

from equilibria.sources import dataset_dir, reference
from equilibria.templates.gtap import GTAPParameters
from equilibria.templates.gtap.gtap_model_equations import GTAPModelEquations
from equilibria.templates.gtap.gtap_parity_pipeline import (
    GTAPVariableSnapshot,
    standard_gtap_reference_gdx,
)

REF = (
    reference("9x10", "gams", form="mcp", mode="gtap", ifsub=1, fbep="zero") / "out.gdx"
)


@pytest.fixture(scope="module")
def sets():
    d = dataset_dir("9x10")
    p = GTAPParameters()
    with contextlib.redirect_stdout(io.StringIO()):
        p.load_from_har(
            basedata_path=d / "basedata.har",
            sets_path=d / "sets.har",
            default_path=d / "default.prm",
        )
    return p.sets


def test_reference_is_the_recorded_9x10_gams_result():
    assert standard_gtap_reference_gdx() == REF


def test_auto_seeds_the_reference_base(sets, monkeypatch):
    monkeypatch.setenv("EQUILIBRIA_GTAP_REFERENCE_SNAPSHOT", "auto")
    # only self.sets is read: a stand-in instead of building the whole model
    equations = cast(Any, SimpleNamespace(sets=sets))
    snap = GTAPModelEquations._load_reference_snapshot_from_env(equations)
    base = GTAPVariableSnapshot.from_gdx(REF, sets, period="base")
    assert snap is not None
    for name in ("xet", "xw", "pe", "pet", "yi"):
        assert getattr(snap, name) == getattr(base, name), name
    # margin prices only on routes with margin flows, as the COMP export wrote them
    assert snap.pwmg == {k: v for k, v in base.pwmg.items() if base.xwmg.get(k)}
    assert len(snap.pwmg) == 700
