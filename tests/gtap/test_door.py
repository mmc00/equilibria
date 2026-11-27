"""``equilibria.gtap`` re-exports the existing GTAP pieces, the same objects, and the
old import paths keep working."""

from __future__ import annotations

import equilibria.gtap as gtap


def test_same_objects() -> None:
    from equilibria.blocks.gtap.overwrite import overwrite
    from equilibria.core.exercise import Exercise
    from equilibria.gtap._load import load
    from equilibria.gtap._results import GTAPResults, results
    from equilibria.report import summary
    from equilibria.templates.gtap.gtap_contract import GTAPClosureConfig
    from equilibria.templates.gtap.recipe import gtap_static
    from equilibria.templates.gtap.template import GTAP
    from equilibria.templates.gtap.welfare_regional import regional_ev

    want = {
        "ClosureConfig": GTAPClosureConfig,
        "Exercise": Exercise,
        "GTAP": GTAP,
        "GTAPResults": GTAPResults,
        "gtap_static": gtap_static,
        "load": load,
        "overwrite": overwrite,
        "regional_ev": regional_ev,
        "results": results,
        "summary": summary,
    }
    assert sorted(gtap.__all__) == sorted(want)
    for name, obj in want.items():
        assert getattr(gtap, name) is obj, name


def test_old_imports_still_work() -> None:
    """The five imports of the quickstart before MAR-249."""
    from equilibria import load_bundled
    from equilibria.core.exercise import Exercise
    from equilibria.templates.gtap.gtap_contract import GTAPClosureConfig
    from equilibria.templates.gtap.recipe import gtap_static
    from equilibria.templates.gtap.template import GTAP

    assert all((load_bundled, Exercise, GTAPClosureConfig, gtap_static, GTAP))
