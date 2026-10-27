# templates/gtap specification

Last synced: 2026-10-27

## Purpose

The GTAP template: the GTAP v7 model built from blocks, its three periods
(`base`, `check`, `shock`) and the multi-period driver that prepares and solves
them. It exists so that a GTAP experiment is written as data (closures, shocks)
and not as edits to the model.

## Scope

`src/equilibria/templates/gtap/` and `src/equilibria/blocks/gtap/`
(`specs/areas.toml`). Only the requirements that an open or closed change has
touched are written here; the rest of the area is not yet specified.

## Requirements

### Requirement: The shock lives in the ShockBlock
Source: ADR-0003, #113

A shock must be written into the `shock` cells of the ShockBlock instruments
with `shocks.apply_shock` before solving; the driver must read it from the model
(`shocks.shock_of`) and must not receive it as an argument. A shock in the
ShockBlock is only accepted in `mode="gtap"`.

### Requirement: Default tariff shock
Source: #113 (PR 2 pending), paper section 2.3

When the model carries no shock, the driver must apply a 10% import tariff. It
does so by changing `params.taxes.imptx` and, in `mode="gtap"`, rebuilding the
`shock` rows of `eq_pmeq` and `eq_ytax[mt]` (with ifSUB=1, `eq_xweq` and
`eq_pmteq`); in `mode="altertax"` it patches `ytax[mt]` and `pm`/`pmt`/`pa` after
the solve. The `imptx` Var of the `shock` period keeps its `check` value.

### Requirement: An imptx shock cell must be read by a live row
Source: #113

`check_shock_cell(m, "imptx", cell)` must accept a cell only when
`eq_pmeq[cell, 'shock']` is active, and must raise otherwise (so with ifSUB=1
no `imptx` cell can take a shock).
