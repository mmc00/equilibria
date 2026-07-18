"""PEP index sets for the Pyomo model.

Wraps the calibrated ``PEPModelState.sets`` (the single source of truth produced by
``PEPModelCalibrator``) so the Pyomo builder never redefines set members. Adds the one
derived set the equations need: ``I1 = I \\ {walras_i}`` (all commodities except the
Walras/redundant-market commodity, default ``agr``) — used by the composite-good market
clearing EQ84 (agr is dropped and handled by the WALRAS/LEON slack).

Default pep2 members (from PEPModelState):
  H  = [hrp, hup, hrr, hur]         households
  F  = [firm]                        firms
  K  = [cap, land]                   capital types
  L  = [usk, sk]                     labor types
  J  = [agr, ind, ser, adm]          production sectors
  I  = [agr, food, othind, ser, adm] commodities   (note: J ≠ I — PEP make matrix)
  AG = [hrp, hup, hrr, hur, firm, gvt, row]  all agents
  AGNG = AG \\ {gvt}                  non-government agents
  AGD  = AG \\ {row}                  domestic agents
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class PEPSets:
    """Index sets for the PEP Pyomo model, sourced from a calibrated PEPModelState."""

    H: list[str] = field(default_factory=list)
    F: list[str] = field(default_factory=list)
    K: list[str] = field(default_factory=list)
    L: list[str] = field(default_factory=list)
    J: list[str] = field(default_factory=list)
    I: list[str] = field(default_factory=list)
    AG: list[str] = field(default_factory=list)
    AGNG: list[str] = field(default_factory=list)
    AGD: list[str] = field(default_factory=list)
    walras_i: str = "agr"

    @property
    def I1(self) -> list[str]:
        """Commodities except the Walras/redundant-market commodity (EQ84 domain)."""
        return [i for i in self.I if i != self.walras_i]

    @classmethod
    def from_state(cls, state: Any) -> "PEPSets":
        s = state.sets
        I = list(s["I"])
        walras_i = "agr" if "agr" in I else (I[0] if I else "agr")
        return cls(
            H=list(s["H"]), F=list(s["F"]), K=list(s["K"]), L=list(s["L"]),
            J=list(s["J"]), I=I,
            AG=list(s["AG"]), AGNG=list(s["AGNG"]), AGD=list(s["AGD"]),
            walras_i=walras_i,
        )
