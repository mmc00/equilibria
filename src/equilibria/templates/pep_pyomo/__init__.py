"""PEP CGE model rebuilt on Pyomo (parity with the cyipopt residual system).

Mirrors the GTAP template layout:
  pep_pyomo_sets        — index sets (wraps the calibrated PEPModelState.sets)
  pep_pyomo_parameters  — calibrated parameters (wraps PEPModelState blocks)
  pep_pyomo_equations   — Vars + all ~96 EQ Constraints (the model builder)
  pep_pyomo_contract    — closure / variant (with|without OBJDEF) / form (NLP|MCP)
  pep_pyomo_solver      — solve wrapper (IPOPT NLP; PATH MCP via walras⊥LEON)

Two model variants (per the GAMS lineage):
  - base:  EQ1..EQ98 + WALRAS, LEON free  (square feasibility system)
  - objdef: base + `OBJDEF: OBJ == 0` free var OBJ (the _ipopt_excel lineage)
Two solve forms:
  - NLP:  constant/zero objective over the equality system (both variants)
  - MCP:  WALRAS ⊥ LEON free-row, EQ84 over I1 (agr excluded), e fixed numeraire
"""

from __future__ import annotations
