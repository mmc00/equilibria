---
status: accepted
---

# One name, one declaration; each block declares what it uses

`Model.add_block` (`src/equilibria/model.py`) accepts a parameter or variable
that an earlier block already declared only if both declarations are identical
(value, domains, bounds); otherwise it raises `ValueError` naming both blocks. An
equation has exactly one owner. The block is checked whole before anything is
registered. Separately, each block lists in `Block.uses` the variables its
equations read but another block declares, and `Model.check_uses` (called by
`PyomoBackend.build`) fails if a used variable is declared by no block.

Before this, `add_block` was "first wins" in silence: a second block that declared
the same name with another initial value, floor or domain was ignored, and a
repeated equation was dropped without warning. That is the kind of bug that does
not fail a solve; it only moves the result. Declaring `uses` by hand, instead of
inferring it, makes a block's dependencies readable without building the model
and turns a missing provider into a clear error when the model is built.

## Considered Options

- **A single owner for every name, no sharing.** Rejected: GTAP shares 26
  declarations on purpose (23 parameters, `pwfact`, `ev`, `cv`), all identical.
- **Infer `uses` from the equations.** Rejected as the source of truth: what an
  equation reads depends on the closure and is only known once it is built. The
  measurement checks the lists instead:
  `tests/blocks/gtap/test_block_uses.py` measures what each equation really reads
  and requires each list to be exact.

## Sources

- #139
- `ROADMAP.md`, "Plan de arquitectura"
