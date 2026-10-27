## ADDED Requirements

### Requirement: A model without the ShockBlock is rejected
Source: change tariff-to-shock-block, ADR-0001

`solve_multiperiod` must raise `ValueError` before loading the solver when the
model has no ShockBlock (the monolith `GTAPMultiPeriodModel`): there is nowhere
to write the shock, the default tariff included.

## MODIFIED Requirements

### Requirement: The shock lives in the ShockBlock
Source: ADR-0003, #113, change tariff-to-shock-block

A shock must be written into the `shock` cells of the ShockBlock instruments
with `shocks.apply_shock` before solving; the driver must read it from the model
(`shocks.shock_of`) and must not receive it as an argument. Both `mode="gtap"`
and `mode="altertax"` accept it.

### Requirement: Default tariff shock
Source: change tariff-to-shock-block, paper section 2.3

When the model carries no shock, the driver must write a 10% import tariff into
the `shock` cells of `imptx` with `apply_shock`, as GAMS fixes `tm`: in
`mode="gtap"` +10% on the power 1+t of every route, the domestic diagonal
included; in `mode="altertax"` the rate times 1.1. No equation is rebuilt and no
solved value is patched after the solve.

### Requirement: An imptx shock cell must be read by a live row
Source: change tariff-to-shock-block

`check_shock_cell(m, "imptx", cell)` must accept a cell when
`eq_pmeq[cell, 'shock']` is active, or, when it is not, when
`eq_xweq[cell, 'shock']` is active and reads the `imptx` Var of that cell
(ifSUB=1, macro M_PM); it must raise otherwise.

## REMOVED Requirements
