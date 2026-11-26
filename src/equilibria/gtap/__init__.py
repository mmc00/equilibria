"""The GTAP door: everything to load, build, shock and read a GTAP exercise.

It only gathers the existing pieces (ADR-0012): you still build each one and see it::

    from equilibria.gtap import ClosureConfig, Exercise, GTAP, gtap_static, load, results

    ex = Exercise(
        GTAP(load("nus333"), ClosureConfig.gtap_default(), residual_region="ROW"),
        gtap_static("gtap", solve_check=True),
        shock={"imptx": {("USA", "MFG", "ROW"): 10.0}},
    )
    ex.run()
    results(ex)  # EV, real GDP and trade by region; .tables["qo"] and the rest

Every name is the same object as in its own module (``ClosureConfig`` is
``templates.gtap.gtap_contract.GTAPClosureConfig``), except ``load`` and ``results``,
which are new. The old import paths keep working.
"""

from __future__ import annotations

from equilibria.blocks.gtap.overwrite import overwrite
from equilibria.core.exercise import Exercise
from equilibria.gtap._load import load
from equilibria.gtap._results import GTAPResults, results
from equilibria.report import summary
from equilibria.templates.gtap.gtap_contract import GTAPClosureConfig as ClosureConfig
from equilibria.templates.gtap.recipe import gtap_static
from equilibria.templates.gtap.template import GTAP
from equilibria.templates.gtap.welfare_regional import regional_ev

__all__ = [
    "ClosureConfig",
    "Exercise",
    "GTAP",
    "GTAPResults",
    "gtap_static",
    "load",
    "overwrite",
    "regional_ev",
    "results",
    "summary",
]
