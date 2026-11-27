"""The cascade's solve of each period starts at the GAMS point (MAR-252).

drift_test and seed_and_solve ask whether Python stays at the GAMS point, so the
solve must start there. The shock's plan seeds its free cells from Python's
solved check (``seed_from(prior)``), which overwrites the GAMS seed: the layers'
solver puts the seed back just before the solve.
"""

import importlib.util
import math
import sys
from pathlib import Path

import pytest

from equilibria.sources import find_reference

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "gtap"))
_spec = importlib.util.spec_from_file_location(
    "seed_and_solve", ROOT / "scripts" / "gtap" / "seed_and_solve.py"
)
assert _spec is not None and _spec.loader is not None
SS = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(SS)

_FOLDER = find_reference(
    "gtap7_3x3", "gams", form="mcp", mode="altertax", ifsub=0, fbep="header"
)


def _seed_levels(gdx: Path, period: str) -> dict[tuple[str, tuple], float]:
    """The levels drift_test compares against: build_seeded plus the derived cascade."""
    from pyomo.environ import value as V

    _p, _gc, m0 = SS.build_seeded("gtap7_3x3", gdx, period, "altertax", False)

    def _setv(comp, key, val):
        try:
            comp[key].set_value(val)
            return 1
        except Exception:
            return 0

    SS._cascade_derived_seed_mp(m0, V, _setv, period)
    return SS.period_levels(m0, period)


@pytest.mark.skipif(_FOLDER is None, reason="GAMS MCP altertax ref absent")
@pytest.mark.needs_path
@pytest.mark.needs_gdxdump  # the seed reads the GAMS GDX through it
@pytest.mark.parametrize("period", ["check", "shock"])
def test_the_solve_of_the_period_starts_at_the_gams_point(period, monkeypatch):
    import equilibria.templates.gtap.recipe as recipe

    assert _FOLDER is not None
    gdx = _FOLDER / "out.gdx"
    real = recipe._by_backend
    start: dict[tuple[str, tuple], float] = {}

    def spy(model):
        rows = model.row_names()
        active = [
            t for t in model.periods if any(model.active_rows(r, t) for r in rows)
        ]
        if active == [period]:
            for name in model.var_names():
                fixed = model.fixed_cells(name, period)
                for k, v in model.values(name, period).items():
                    if k not in fixed:
                        start[(name, k)] = v
        return real(model)

    monkeypatch.setattr(recipe, "_by_backend", spy)
    seed = _seed_levels(gdx, period)
    SS._solve_gtap("gtap7_3x3", gdx, False, "altertax")

    assert start, f"no solve of {period} was seen"
    off = sorted(
        (abs(v - seed[c]) / max(1.0, abs(seed[c])), c, seed[c], v)
        for c, v in start.items()
        if c in seed
        and not math.isnan(v)
        and abs(v - seed[c]) > 1e-12 * max(1.0, abs(seed[c]))
    )
    assert not off, (
        f"{len(off)} free cell(s) of {period} do not start at the GAMS seed; "
        f"largest {off[-3:]}"
    )
