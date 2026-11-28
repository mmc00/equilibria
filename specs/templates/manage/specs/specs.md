# templates/manage specification

Last synced: 2026-11-27

## Purpose

The MANAGE-WB template: the World Bank's single-country recursive-dynamic CGE
model built from blocks, with its data reader, its calibration and its recipe.
It exists so that MANAGE runs in equilibria and gives the same numbers as its
GAMS reference (`data/manage/`).

## Scope

`src/equilibria/templates/manage/` and the door `src/equilibria/manage.py`
(`specs/areas.toml`). The shared blocks it uses live under
`src/equilibria/blocks/` and belong to their own areas. No requirement is written
yet; the open change `manage-template` adds the first ones.

## Requirements
