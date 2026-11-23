"""altertax keeps alphad/alpham at their base calibration, as GAMS does.

GAMS calibrates the Armington shares once, in the cal block (``loop(tsim$t0)``),
and the reference GDX holds them identical in base, check and shock. Python used
to rebuild eq_paa each period from the previous period's solved xd/xm/xa. For
flows near 1e-9 those quantities carry the solver's absolute tolerance, and
with sigmam = 0.95 the eq_paa row raises pa to the power 1/(1 - 0.95) = 20: in
gtap7_15x10 ifSUB=0 alpham came out 82% off and pa[MEX,Rice,gov] 23.7 against
GAMS's 1.0 (shock 99.45%).
"""

from __future__ import annotations

import pytest
from tests.templates.gtap import _period_snapshot as snap

pytestmark = pytest.mark.skipif(
    not snap.DATASET.exists(), reason="gtap7_3x3 dataset not present"
)


@pytest.mark.parametrize("case", ["altertax_ifsub0", "altertax_ifsub1"])
def test_altertax_does_not_rebuild_eq_paa(case, monkeypatch):
    shots = snap.take(case, monkeypatch)

    rebuilt = sorted(
        {
            name
            for shot in shots
            for name in shot["components"]
            if name.startswith("Con:_eq_paa_recal_")
        }
    )

    assert rebuilt == []
