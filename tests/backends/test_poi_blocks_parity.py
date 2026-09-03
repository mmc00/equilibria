"""Real GTAP blocks build against the POI backend, and produce the same equations
as the Pyomo path.

This is the assumption Fase 0 exists to test: the block bodies are written in Pyomo
syntax, and if an adapter can stand in for the model object they should run
unmodified against PyOptInterface. If a block turns out to need editing, that is a
finding to report — the blocks are not to be changed to accommodate the backend.
"""

from __future__ import annotations

import pytest

from tests.backends._poi_fixtures import load_ipopt_or_skip, tiny_params


def _full_model():
    """The seven-block ``EquilibriaModel`` on the 3x3 dataset.

    A single block is not a self-contained model: ``IncomeBlock`` declares the 16
    variables it owns but writes equations over ``pf``, ``xf`` and others belonging
    to the factor and production blocks. Pyomo fails on a lone block for exactly the
    same reason, so the composed model is the honest unit to build.
    """
    from equilibria.backends.poi_backend import build_gtap_equilibria_model

    params = tiny_params()
    return build_gtap_equilibria_model(params, list(params.sets.r)[-1])


def test_income_block_builds_on_poi_without_editing_it():
    """Real block equations produce POI constraints through the adapter.

    ``eq_facty`` is the specific check: a genuine ``IncomeBlock`` equation combining
    a product, a division and a ``sum`` filtered on a parameter — the shapes that
    would expose an adapter too thin to carry real equations.
    """
    load_ipopt_or_skip()

    from equilibria.backends.poi_backend import PoiBackend

    backend = PoiBackend()
    backend.build(_full_model())

    assert backend.constraints, "the blocks produced no constraints at all"
    bases = {key.split("[")[0] for key in backend.constraints}
    assert "eq_facty" in bases, f"eq_facty missing; got {sorted(bases)[:10]}"


def _normalize_pyomo_name(name: str) -> str:
    """``eq_facty_con[USA]`` -> ``eq_facty[USA]``.

    The Pyomo backend suffixes constraints with ``_con``; the POI backend does not.
    Comparing the two raw would report every row as a difference on formatting
    alone, hiding whichever rows genuinely diverge.
    """
    base, sep, rest = name.partition("[")
    if base.endswith("_con"):
        base = base[:-4]
    return f"{base}{sep}{rest}"


def test_both_backends_produce_the_same_equations():
    """Gate 1: POI and Pyomo build the same set of constraint cells.

    Cell names are compared rather than values. A backend that generated extra or
    missing rows could not square the system the same way, which would make any
    later numeric comparison meaningless — so this has to hold before build time or
    Jacobian density mean anything.

    Both sides are built from the same ``EquilibriaModel``, so a difference here
    belongs to the backend rather than to two separately assembled models.
    """
    load_ipopt_or_skip()

    from pyomo.environ import Constraint

    from equilibria.backends.poi_backend import PoiBackend
    from equilibria.backends.pyomo_backend import PyomoBackend

    pyomo_backend = PyomoBackend()
    pyomo_backend.build(_full_model())
    pyomo_names = {
        _normalize_pyomo_name(c.name)
        for c in pyomo_backend.pyomo_model.component_data_objects(
            Constraint, active=True
        )
    }

    poi_backend = PoiBackend()
    poi_backend.build(_full_model())
    poi_names = set(poi_backend.constraints)

    missing_in_poi = sorted(pyomo_names - poi_names)
    extra_in_poi = sorted(poi_names - pyomo_names)

    assert not missing_in_poi, (
        f"{len(missing_in_poi)} cells Pyomo builds and POI does not: "
        f"{missing_in_poi[:15]}"
    )
    assert not extra_in_poi, (
        f"{len(extra_in_poi)} cells POI builds and Pyomo does not: "
        f"{extra_in_poi[:15]}"
    )
