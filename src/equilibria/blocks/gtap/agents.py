"""Etiquetas de los agentes de demanda final de GTAP.

Son vocabulario del dominio — los cuatro literales con los que el dataset
indexa la demanda final — y los leen las tres capas: `blocks/`, `solver/` y
`templates/`. Vivian en `templates/gtap/gtap_parameters.py`, lo que obligaba a
`blocks/gtap/_derived_params.py` a importar HACIA ARRIBA (blocks -> templates)
con imports diferidos dentro de funciones para romper el ciclo.

Viven aqui, en la capa baja que los necesita; `gtap_parameters` los re-exporta
para no romper a nadie (es donde los buscan los ~10 consumidores actuales).
"""

from __future__ import annotations

GTAP_HOUSEHOLD_AGENT = "hhd"
GTAP_GOVERNMENT_AGENT = "gov"
GTAP_INVESTMENT_AGENT = "inv"
GTAP_MARGIN_AGENT = "tmg"

#: Orden canonico del set de agentes agregados (actividades + demanda final).
GTAP_FINAL_DEMAND_AGENTS = (
    GTAP_HOUSEHOLD_AGENT,
    GTAP_GOVERNMENT_AGENT,
    GTAP_INVESTMENT_AGENT,
    GTAP_MARGIN_AGENT,
)

__all__ = [
    "GTAP_FINAL_DEMAND_AGENTS",
    "GTAP_GOVERNMENT_AGENT",
    "GTAP_HOUSEHOLD_AGENT",
    "GTAP_INVESTMENT_AGENT",
    "GTAP_MARGIN_AGENT",
]
