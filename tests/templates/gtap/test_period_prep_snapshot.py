"""La preparacion de cada periodo del driver no cambia sin que alguien lo decida.

Compara la foto del modelo antes de cada solve (ver ``_period_snapshot.py``)
contra la guardada en ``tests/fixtures/period_prep_snapshot.json``.  Un cambio
INTENCIONAL en la preparacion se registra regenerando la foto:

    EQUILIBRIA_UPDATE_PERIOD_SNAPSHOT=1 uv run pytest tests/templates/gtap/test_period_prep_snapshot.py
"""

from __future__ import annotations

import os

import pytest
from tests.templates.gtap import _period_snapshot as snap

pytestmark = pytest.mark.skipif(
    not snap.DATASET.exists(), reason="gtap7_3x3 dataset not present"
)

_UPDATE = os.environ.get("EQUILIBRIA_UPDATE_PERIOD_SNAPSHOT") == "1"


@pytest.mark.parametrize("case", sorted(snap.CASES))
def test_period_preparation_matches_the_snapshot(case, monkeypatch):
    got = snap.take(case, monkeypatch)
    data = snap.load_fixture()
    if _UPDATE:
        data[case] = got
        snap.save_fixture(data)
        return
    assert case in data, f"sin foto guardada para {case}: regenerar (ver docstring)"
    problems = snap.diff(data[case], got)
    assert not problems, (
        f"la preparacion de periodos cambio en {case}:\n  "
        + "\n  ".join(problems[:40])
        + "\nSi el cambio es intencional, regenerar la foto (ver docstring)."
    )
