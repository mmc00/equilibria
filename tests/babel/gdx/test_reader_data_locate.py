from pathlib import Path
import pytest
from equilibria.babel.gdx.reader import read_gdx, _section_at_offset

REF = Path("/Users/marmol/proyectos2/equilibria_refs/"
           "gtap7_3x3_altertax_cd/out_altertax_ifsub0.gdx")
pytestmark = pytest.mark.skipif(not REF.exists(), reason="durable ref GDX absent")


def test_section_starts_after_marker():
    data = read_gdx(str(REF))
    raw = REF.read_bytes()
    sym = next(s for s in data["symbols"] if s["name"] == "arenteq")
    off = sym["data_offset"]
    # marker is 7 bytes: 0x06 + b"_DATA_"
    assert raw[off:off + 7] == b"\x06_DATA_"
    section = _section_at_offset(raw, off)
    # section begins with the dim byte (arenteq dim=2)
    assert section[0] == 2
