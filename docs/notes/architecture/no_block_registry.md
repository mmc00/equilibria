# Why there is no block registry

**Date:** 2026-09-24 · **Status:** decided, before the first PyPI release

`BlockRegistry`, `get_registry` and `register_block` were removed from
`blocks/base.py` and from both `__all__` (`equilibria/` and `equilibria/blocks/`).

## What they were

A `name -> block class` table with a global singleton and a decorator,
so that a block could be requested by its name as text:

```python
bloque = get_registry().create("IncomeBlock", sigma=0.8)
```

## Why they went

Their only user was an example that demonstrated them to itself. Measured before
deleting:

- **33 subclasses of `Block`** in `src/` (GTAP, gtap_logvalue, PEP, trade,
  institutions, demand, production, equilibrium). **None** carried
  `@register_block`.
- The only exception was in `examples/`: `example_04_custom_blocks.py`
  decorated its `SimpleDemand`, printed `registry.list_blocks()` and devoted a
  whole step to `registry.create("SimpleDemand")`. It was a REAL user —CI
  runs it as a test— but of a particular kind: the example existed to teach
  the registry, it did not use it for anything it needed. The block was instantiated
  directly (`SimpleDemand(name=...)`) and the text itself said
  "Use @register_block decorator (optional)". It was rewritten in this same commit
  to compose by direct import, which is what the rest of the repo does.
- Outside that example, `get_registry()` was not called from anywhere: the
  other live mentions were inside the dead code itself
  (`register_block` calling itself) or in docstring `>>>` examples.
- The repo needed a registry **three times** elsewhere, and all three times it wrote
  a module `dict` without touching this one: `_ADAPTER_REGISTRY`
  (`simulations/simulator.py`), `_MAPPING_RUNTIME_REGISTRY`
  (`simulations/runtimes.py`), `GEMPACK_KIND` (`templates/gtap/shocks.py`).

And its signature did not model the real problem. Blocks are not created with loose
parameters: `_mk_unit` (`templates/gtap/gtap_block_model.py`) instantiates them with
`sets` + `params` and threads `residual_region` / `if_sub` / `savf_flag` according to the
fields each class declares, and `_block_classes` returns them in **dependency
order**. `create(name, **kwargs)` expresses neither the order nor that passing of
context.

## The case that does matter: describing a model in YAML

The motivation for keeping it was that an economist could define a model
without writing Python. The file we would want to be able to write was sketched:

```yaml
modelo: gtap7
dataset: 9x10                # equilibria.datasets.dataset_path('gtap', ...)

closure: gtap_standard       # _closure_template_data, en gtap_contract.py
  # savf_flag: capFlex       # ...o sobrescribir campos sueltos
  # if_sub: false

region_residual: ROW
base_calibrada: true

shock:                       # -> shocks.apply_shock(m, shock), en el modelo construido
  imptx:                     # un instrumento del ShockBlock (shocks.GEMPACK_KIND)
    [USA, VegFruit, EU_28]: 10.0   # celda (exportador, bien, importador): % GEMPACK

solver:
  motor: ipopt
  periodos: [base, check, shock]
```

The values above are **measured against the repo**, not invented: `9x10`
is one of the two real keys of `dataset_path` (`['9x10', 'nus333']`) and
`imptx` —the import tariff, GEMPACK's `tms`— is one of the instruments
of the `ShockBlock`; the % reads as in the `.EXP` and the conversion to levels is
decided by the instrument. The first version of this sketch
put `dataset: gtap7_10x7` and `shock: {tipo: arancel, destino: tm}`: **neither
is accepted by anything**; the code review found it. It matters because this sketch
is the guide for future work, and a guide with plausible but false values
sends the next person to the wrong place.

And it lays bare what the YAML really lacks, which is not a block
registry: the large datasets (`gtap7_10x7`, `gtap7_15x10`...) live as
folders in `data/<model>/` and in hand-written lists inside `scripts/`, outside the
`equilibria.datasets` registry. Unifying that is the first real step.

**The sketch has no list of blocks, and that is the finding.** A GTAP user
does not choose blocks: the 7 *are* the GTAP model (plus the `ShockBlock`, which only
declares the fixed instruments), and removing one does not give another model but a broken one. They choose dataset, closure, shock and periods — and all four already
have their registry. None of them is the block one.

The YAML -> model path is, almost entirely, reading the file and calling
`build_block_model(params, sets, closure, residual_region, ...)` with pieces that already
exist. The template puts the blocks in, as now.

## When this should be reopened

When a **third party** publishes blocks in their own package (`equilibria_energia`
with their own `Block`s) without touching this repo. That is the only case where the
`@register_block` decorator is indispensable, because the block lives in code that
we do not control.

If that day comes, the registry is designed from `_block_classes` and `_mk_unit`
in `gtap_block_model.py` — which already are the real registry, written by hand and with the
dependency order that the deleted one did not model — and not by bringing this version back.

> Note (2026-10-05): the shock lives in the `ShockBlock` of the built model, only
> in the shock period: `shocks.apply_shock(m, {instrument: {cell: %}})`. The driver
> does not receive the shock; it reads it from the model (`shocks.shock_of`). The old
> `apply_shock(params)`, which changed `params` and left the shock in all 3 periods,
> was deleted.
