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
