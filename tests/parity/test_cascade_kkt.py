import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "gtap"))

from cascade_kkt import KKT_READER, kkt_layer, read_marginals

REF = Path(
    "/Users/marmol/proyectos2/equilibria_refs/"
    "gtap7_3x3_altertax_cd/out_altertax_ifsub0.gdx"
)
pytestmark = pytest.mark.skipif(not REF.exists(), reason="durable ref GDX absent")


def test_read_marginals_returns_period_filtered_values():
    m = read_marginals(REF, "arenteq", "check")
    # keys no longer contain the period; USA present with the gdxdump value
    usa = next(v for k, v in m.items() if "USA" in k)
    assert abs(usa - 0.157338617290058) < 1e-9


def test_kkt_layer_returns_layerresult():
    res = kkt_layer("gtap7_3x3", "shock", REF)
    assert res.name == "kkt_marginals"
    assert res.status in ("clean", "dirty", "error")


def test_kkt_reader_is_pure_python():
    assert KKT_READER == "pure-python"
