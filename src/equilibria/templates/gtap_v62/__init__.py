"""GTAP CGE Template — Version 6.2 (Hertel/Itakura/McDougall 2003).

Implements the standard GTAP model v6.2 in Pyomo. Parallel to
``equilibria.templates.gtap`` (Standard GTAP 7) but rolled back to the
v6.2 structure documented in Corong et al (2017) Table 1:

- No ACT/COMM split (single index ``i`` for sectors and commodities)
- No intermediate bundle (Leontief implicit between VA and intermediates)
- No MAKE transformation (diagonal make matrix)
- Government and trade-margins demand are Cobb-Douglas (no ESUBG/ESUBS)
- Factor markets at commodity level (no ``tinc(e,a,r)``; uses ``toi(i,r)``)
- Investment as a producing sector (cgds), not an explicit agent
- SLUG binary flag for factor mobility (no ENDOWFLAG matrix)

Reference: ``C:\\runGTAP375\\gtap.tab`` (GTAP Model Version 6.2, Sept 2003).

Validation oracle: ``gtap.exe`` v6.2 in ``C:\\runGTAP375\\`` running the
TAB on BOOK3X3 / NUS333 datasets.
"""

from equilibria.templates.gtap_v62.gtap_v62_parameters import (
    GTAPv62BenchmarkValues,
    GTAPv62Elasticities,
    GTAPv62Parameters,
)
from equilibria.templates.gtap_v62.gtap_v62_sets import GTAPv62Sets

__all__ = [
    "GTAPv62Sets",
    "GTAPv62Parameters",
    "GTAPv62Elasticities",
    "GTAPv62BenchmarkValues",
]
