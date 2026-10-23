---
status: accepted
date: 2026-10-02
---

# The driver prepares and solves; it holds no model logic

The multi-period driver (`templates/gtap/gtap_multiperiod_driver.py`) only
orchestrates: it asks for each period to be prepared, solves it with the closure
it gets back, and moves on. How a period is prepared (freeze the other periods,
seed, recalibrate, fix and bound from the single-period reference model, holdfix)
is a **recipe**: an ordered list of steps per `(period, mode)` in
`templates/gtap/period_prep.py`, run by `PeriodPreparer.prepare(m, period,
params)`. Equations, instruments and the shock belong to blocks, not to the
driver.

Before this, each period's preparation was written by hand inside the driver, one
copy per period with an `if _gtap_mode` at every step, so the three copies drifted
and nobody could tell which differences were intended. A recipe makes the
sequence visible and testable: `tests/templates/gtap/test_period_prep_snapshot.py`
pins a snapshot of the model before every solve.

## Considered Options

- **Keep the inline per-period code and factor out helpers.** Rejected: the
  helpers already existed; what was missing was one place that states the order
  of the steps for each period and mode.

## Consequences

The driver is not there yet. It still applies the default +10 % tariff when the
model carries no shock, and rebuilds shock equations to do it (the
`_rebuild_*_shock` functions). Moving that tariff into the `ShockBlock` is the
open change `tariff-to-shock-block` (`ROADMAP.md` "Plan de arquitectura", the
shocks row: PR 2 after #113; see ADR-0003).

## Sources

- #103 (base and check prepared by recipes), #108 (the shock recipe)
- `ROADMAP.md`, "Plan de arquitectura"
