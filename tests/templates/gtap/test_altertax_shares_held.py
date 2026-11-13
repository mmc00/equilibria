"""altertax keeps the share parameters at their base calibration, as GAMS does.

GAMS sets and/ava, gx/ax and the alphaa shares in cal.gms ("Calibration of
parameters"), included before the tsim loop; iterloop.gms does not touch them,
and the reference GDXs hold them identical in base, check and shock. Python used
to rebuild their rows every period from the previous period's solved levels
(the ``_recalibrate_*`` of the driver). Step 0 of the driver recipe deletes each
rebuild only when no cell of the altertax gates leaves the gates' tolerance
(``scripts/gtap/measure_altertax_cells.py``); ``_recalibrate_and_ava`` stays.
"""

from __future__ import annotations

import pytest
from tests.templates.gtap import _period_snapshot as snap

pytestmark = pytest.mark.skipif(
    not snap.DATASET.exists(), reason="gtap7_3x3 dataset not present"
)

# The rows each deleted recalibration rebuilt, by component-name prefix. Only the
# ones it did rebuild: _recalibrate_alphaa_gov_inv never rebuilt the government's
# eq_pg/eq_xg rows on gtap7_3x3, so a test on them could not fail.
HELD = [
    "_eq_xi_recal_",  # _recalibrate_alphaa_gov_inv (investment)
]


@pytest.mark.parametrize("prefix", HELD)
@pytest.mark.parametrize("case", ["altertax_ifsub0", "altertax_ifsub1"])
def test_altertax_does_not_rebuild(case, prefix, monkeypatch):
    shots = snap.take(case, monkeypatch)

    rebuilt = sorted(
        {
            name
            for shot in shots
            for name in shot["components"]
            if name.startswith(f"Con:{prefix}")
        }
    )

    assert rebuilt == []
