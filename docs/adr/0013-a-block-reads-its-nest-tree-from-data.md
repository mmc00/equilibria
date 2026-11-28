---
status: proposed
date: 2026-11-27
---

# A block reads its nest tree from data

A nested CES or CET block may take its tree (which child sits under which
parent, each parent's elasticity and form) from parameters read with the data,
instead of writing one block per level. The shape of the model then lives in
the dataset: the block's rows and columns are the cells the tree table selects,
exactly as the reference model's `$` conditions select them. Its variable
names are fields, so one block class serves several trees in the same model
(each instance still declares each name once, ADR-0005).

MANAGE-WB needs this. Its production technology is one pair of GAMS equations,
`e_xfac` and `e_pxfac` (`data/manage/gams/inc/model.inc:1253, 1307`). The tree
comes from the share table `p_alphaXfac(child,parent,activity,vintage)`. In Ghana
it has 19 parent nests, 7 levels at most, children with more than one parent,
three additive-CES nests and a Leontief branch. The household's CES sub-nests
below its CDE top level are a second tree of the same kind
(`model.inc:2303-2352`). Writing a block per level would give a different model
per dataset, and every dataset change would become a code change.

## Considered Options

- **One block per nest level, as the GTAP blocks do**
  (`blocks/gtap/production_supply.py`, a fixed xp -> nd/va tree). Rejected for
  MANAGE: its trees differ by activity and by dataset.
- **Generate block classes from the tree at build time.** Rejected: it hides
  the equations from a reader and from `uses` (ADR-0005), and ADR-0007 rules out
  a registry.

## Consequences

- A wrong row in the tree table fails no solve; it moves the numbers. Every
  instance is therefore tested against the reference's row instance (count and
  index set of each family), not only its residuals.
- The first block of this kind is `CESTree` (`src/equilibria/blocks/production/`),
  change `manage-template` (`specs/templates/manage/changes/manage-template/`).

## Sources

- `specs/templates/manage/changes/manage-template/design.md`
