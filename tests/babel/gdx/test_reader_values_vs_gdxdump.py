import re
import subprocess
from pathlib import Path

import pytest

from equilibria.babel.gdx.reader import read_gdx, read_equation_values

GDXDUMP = "/Library/Frameworks/GAMS.framework/Versions/53/Resources/gdxdump"
REF = Path("/Users/marmol/proyectos2/equilibria_refs/"
           "gtap7_3x3_altertax_cd/out_altertax_ifsub0.gdx")
pytestmark = pytest.mark.skipif(
    not REF.exists() or not Path(GDXDUMP).exists(), reason="ref or gdxdump absent")


def _gdxdump_marginals(name):
    out = subprocess.run([GDXDUMP, str(REF), f"Symb={name}"],
                         capture_output=True, text=True).stdout
    vals = {}
    for m in re.finditer(r"((?:'[^']*'\.)+)M\s+([-\d.eE+]+)", out):
        key = tuple(p.strip("'") for p in m.group(1).rstrip(".").split("."))
        vals[key] = float(m.group(2))
    return vals


@pytest.mark.parametrize("name", ["arenteq"])
def test_equation_marginals_match_gdxdump(name):
    data = read_gdx(str(REF))
    ours = read_equation_values(data, name)
    truth = _gdxdump_marginals(name)
    assert truth, "gdxdump produced no marginals to compare"
    for key, m in truth.items():
        ours_key = next((k for k in ours if all(p in k for p in key)), None)
        assert ours_key is not None, f"missing {key}"
        assert abs(ours[ours_key]["marginal"] - m) < 1e-9
