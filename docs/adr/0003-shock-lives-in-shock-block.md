---
status: accepted
date: 2026-10-19
---

# The shock lives in the ShockBlock and is written as in GEMPACK

There is one way to shock a GTAP model: write it into the `ShockBlock` of the
built model with `shocks.apply_shock(m, {instrument: {cell: pct}})`, before
solving. Each instrument is a Var that is fixed in every period, and the shock is
that its `shock` cell differs from its `check` cell, as in GAMS
(`x.fx(..., 'shock') = v`). The number is the GEMPACK percentage, as read in a
`.EXP`, and the instrument decides how it becomes a level
(`GEMPACK_KIND`: `pct`, `power`, `power_kappa`, `power_fct`); `levels=True`
takes the level directly. The driver does not receive the shock; it reads it
from the model with `shocks.shock_of`.

We write in GEMPACK's terms because the published exercises and our GEMPACK
oracle are stated that way, and converting by hand per instrument (a % of the
power `1+t` is not a % of `t`) is where errors came from. Keeping the shock in
the model, and only in the `shock` period, means base and check replicate the
benchmark whatever the shock is.

## Considered Options

- **Change `params` before building** (the old `apply_shock(params)`). Rejected
  and deleted: it put the shock in all three periods.
- **Pass the shock to the driver as arguments** (the old `lambdava_shock`,
  `qe_shock`). Rejected and removed: every new instrument needed a new argument
  and its own code path in the driver.

## Consequences

`apply_shock` is all or nothing and rejects a shock that changes nothing
(`check_shock_value`): with no shock in the model the driver would silently
apply its default +10 % tariff. That default still lives in the driver until
the open change `tariff-to-shock-block` moves it here (`ROADMAP.md` "Plan de
arquitectura", the shocks row: PR 2 after #113).

## Sources

- #113
- `docs/architecture/registro_de_bloques.md` (closing note)
- `ROADMAP.md`, "Plan de arquitectura"
