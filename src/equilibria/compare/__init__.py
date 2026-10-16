"""equilibria.compare: compara dos soluciones del mismo modelo, celda por celda, en niveles.

Las fuentes (GAMS ``.gdx``, un modelo Pyomo resuelto o un dict) producen una
:class:`Solution`; el usuario declara la correspondencia en un :class:`Mapping`;
:func:`compare` alinea, mide y devuelve un :class:`Report`::

    from equilibria.compare import Mapping, compare, from_gdx, from_pyomo

    report = compare(
        from_gdx("gams.gdx", period_dim=-1),
        from_pyomo(model, period_dim=-1),
        Mapping.from_yaml("map.yaml"),
        reference="a",
    )
    print(report.summary)
"""

from equilibria.compare.mapping import Mapping, MappingError, Tolerance
from equilibria.compare.report import Cell, Report, Unmatched, compare
from equilibria.compare.solution import Solution
from equilibria.compare.sources import from_dict, from_gdx, from_pyomo

__all__ = [
    "Cell",
    "Mapping",
    "MappingError",
    "Report",
    "Solution",
    "Tolerance",
    "Unmatched",
    "compare",
    "from_dict",
    "from_gdx",
    "from_pyomo",
]
