---
status: accepted
---

# An exercise changes one block with @overwrite and leaves the others alone

An exercise (one notebook) that needs a different closure or an extra equation
modifies a single block with `@overwrite(BlockClass, period="shock")`
(`blocks/gtap/overwrite.py`). The hook runs after the block's own `setup` and
gets a `BlockEdit` with three verbs: `endogenous` (free one cell of an
instrument), `equation` (add a row) and `target` (a quantity target that becomes
a new instrument). Together they express a GEMPACK `swap` without a swap API.
The new closure holds **only in the shock period**, as GEMPACK's and GAMS's swaps
do, so base and check still replicate the benchmark.

Hooks are registered on the block **class** and last for the whole process, so
everything that builds blocks sees them: `build_block_model`, the driver's
single-period reference model and the calibration. Re-running the notebook cell
replaces the hook instead of stacking it. This is the surprising part, and it is
deliberate: passing a modified block list through every builder would have
touched every call site, and a builder left out would still use the standard
block.

## Considered Options

- **Subclass the block per exercise.** Rejected: the template composes fixed
  classes, so every builder would need to be told about the subclass.
- **A swap API on the closure.** Rejected: a swap often needs a new equation
  (the real wage in TBL65B, the quantity target in TBL94 and ME9A), which a
  variable swap alone cannot express.

## Consequences

With hooks active the model and seed caches are skipped (the hook is not in the
cache key), and the monolith refuses to build (`require_blocks`) rather than
ignore the hooks. `period` accepts only `"shock"`.

## Sources

- #100 (`@overwrite`, TBL65B), #105 (`b.target`, TBL94, English API),
  #107 (shock-only closure, ME9A)
