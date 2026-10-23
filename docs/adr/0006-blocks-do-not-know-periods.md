---
status: accepted
date: 2026-10-18
---

# Blocks do not know about periods; the SP copy adds the time index

A block writes its equations for a single period, with no time index. The
multi-period model is built by reflecting the single-period (SP) model: every Var
family gets a trailing `t` index and every equation is copied once per period
(`GTAPMultiPeriodModel.build_vars` and `build_equations_all_periods` in
`templates/gtap/gtap_model_multiperiod.py`, fed by `_build_sp`, which the block
path overrides). The periods themselves are a template concern.

This keeps every block reusable whatever the time structure is. Today the periods
are fixed (`PERIODS = (base, check, shock)` in `blocks/gtap/periods.py`) and the
driver assumes that sequence; the next step on the roadmap is periods as a list
with a transition recipe between them, so a recursive dynamic model is one year
solved at a time with a dynamics block updating stocks between years. That step
needs no change to any block's equations, because they never saw `t`.

## Considered Options

- **Write `t` into every block's equations**, as GAMS does. Rejected: each block
  would carry the period structure, and changing it (a list instead of three
  fixed periods) would mean editing every block.

## Consequences

Anything period-specific lives outside the block's equations: the shock is the
`shock` cell of an instrument (ADR-0003), and an `@overwrite` closure is pruned to
the shock period after reflection (`overwrite.shock_only`, ADR-0004).

The rule covers equations, not all block code. `blocks/gtap/periods.py` lives in
`blocks/` so blocks can read the period names, and `FactorBlock.calibrate_base`
(`blocks/gtap/factor.py`) runs the base, check and shock stack and reads the
`check` period by name. Periods as a list must touch it.

## Sources

- `ROADMAP.md`, "Plan de arquitectura", "Siguiente, en este orden" item 1
- `docs/architecture/monolito_vs_bloques.md` (the multi-period reflection asks
  for its SP model by method)
