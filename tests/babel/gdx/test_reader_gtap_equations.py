import math
from pathlib import Path

import pytest

from equilibria.babel.gdx.reader import read_gdx, read_equation_values

REF = Path("/Users/marmol/proyectos2/equilibria_refs/"
           "gtap7_3x3_altertax_cd/out_altertax_ifsub0.gdx")
pytestmark = pytest.mark.skipif(not REF.exists(), reason="durable ref GDX absent")


def _symbol(data, name):
    return next((s for s in data["symbols"] if s["name"] == name), None)


def test_arenteq_classified_as_equation():
    data = read_gdx(str(REF))
    sym = _symbol(data, "arenteq")
    assert sym is not None and sym["type_name"] == "equation"


def test_many_equations_found():
    data = read_gdx(str(REF))
    eqs = [s["name"] for s in data["symbols"] if s["type_name"] == "equation"]
    assert len(eqs) >= 50, f"found {len(eqs)}"  # gdxdump sees 103


@pytest.mark.xfail(reason="_DATA_ var/equ decoder rewritten in Task 3; the "
                          "symbol-table fix alone does not decode values yet")
def test_arenteq_marginals_present_with_period_dim():
    data = read_gdx(str(REF))
    vals = read_equation_values(data, "arenteq")
    assert vals
    key = next(k for k in vals if ("USA" in k and "check" in k))
    assert math.isfinite(vals[key]["marginal"])
    assert abs(vals[key]["marginal"] - 0.157338617290058) < 1e-9
