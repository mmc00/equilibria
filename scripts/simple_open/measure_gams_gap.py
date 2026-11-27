"""How far simple_open lands from GAMS (Hosoe's stdcge), per backend and period.

Runs the tariff abolition (base, check, shock) on Pyomo and on the family backend
and compares every cell with GAMS's GDX (``Mapping.builtin("stdcge")``). The
distance is the compare metric, ``|a - b| / max(|b|, abs_floor)``; the table shows
the largest per period, the variable it falls on and the next four, plus Walras
and the equivalent variation. It measures; it sets no tolerance.

    uv sync --dev --extra family
    uv run python scripts/simple_open/measure_gams_gap.py
"""

from __future__ import annotations

import time
from collections import defaultdict
from typing import Any

from equilibria.babel.gdx.reader import read_gdx, read_parameter_values
from equilibria.compare import Mapping, compare, from_dict, from_gdx
from equilibria.core import Exercise
from equilibria.sources import reference
from equilibria.templates.simple_open import SimpleOpenEconomy
from equilibria.templates.simple_open.template import equivalent_variation

PERIODS = ("base", "check", "shock")
SHOCK = {"tau_m": 0.0}


def _period(model: Any, period: str) -> dict[str, dict[tuple, float]]:
    cells: dict[str, dict[tuple, float]] = defaultdict(dict)
    for name, index in model.cells(period):
        cells[name][index] = model.value(name, index, period)
    return cells


def measure(backend: str) -> None:
    gams = reference("hosoe_2x2", "gams", form="nlp", experiment="tariff0")
    template = SimpleOpenEconomy(backend=backend)
    ex = Exercise(template, shock=SHOCK)
    t0 = time.perf_counter()
    results = ex.run()
    took = (time.perf_counter() - t0) / 60
    model = ex.model
    codes = ", ".join(f"{t} {results[t]['code']}" for t in PERIODS)
    print(f"\n== {backend}: {codes} ({took:.2f} min)")
    print(
        f"{'period':7} {'cells':>5} {'passed':>6} {'max d':>10}  worst variable (cell)"
    )
    for t in PERIODS:
        report = compare(
            from_dict(_period(model, t), period=t),
            from_gdx(gams / f"{t}.gdx", period=t),
            Mapping.builtin("stdcge"),
            reference="b",
        )
        o = report.overall
        print(
            f"{t:7} {o['cells']:>5} {o['passed']:>6} {o['max']:>10.2e}  "
            f"{o['max_variable']} {o['max_cell']}"
        )
        top = sorted(report.summary.items(), key=lambda kv: -(kv[1]["max"] or 0.0))[1:5]
        print(
            "        next: " + ", ".join(f"{v} {s['max'] or 0.0:.1e}" for v, s in top)
        )
    for t in ("check", "shock"):
        print(f"|WALRAS| {t}: {abs(model.value('WALRAS', (), t)):.2e}")
    ev_gams = read_parameter_values(read_gdx(gams / "shock.gdx"), "EV")[()]
    ev = equivalent_variation(template.calibration(), model)
    print(f"EV: ours {ev:.10g}, GAMS {ev_gams:.10g}, |diff| {abs(ev - ev_gams):.2e}")


if __name__ == "__main__":
    for b in ("pyomo", "family"):
        measure(b)
