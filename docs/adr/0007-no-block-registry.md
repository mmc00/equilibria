---
status: accepted
date: 2026-09-24
---

# No block registry

There is no `name -> block class` registry. `BlockRegistry`, `get_registry` and
`register_block` were removed from `blocks/base.py` and from both public
`__all__` lists (`equilibria` and `equilibria.blocks`) before the first PyPI
release. A template composes its blocks by direct import; for GTAP,
`_block_classes` and `_mk_unit` in
`templates/gtap/gtap_block_model.py` are the real registry: they return the
blocks in dependency order and build each one with its sets, parameters and the
context fields it declares.

When removed, none of the 33 `Block` subclasses in `src/` used the registry; its
only user was an example that existed to demonstrate it. Its
`create(name, **kwargs)` signature could express neither dependency order nor
that context. And a user of a template does not choose blocks: the GTAP blocks
*are* GTAP, and dropping one gives a broken model, not another model. What a user
chooses (dataset, closure, shock, periods) already has its own entry point.
Where the repo did need a registry it wrote a module-level `dict`
(`_ADAPTER_REGISTRY` in `simulations/simulator.py`, `_MAPPING_RUNTIME_REGISTRY`
in `simulations/runtimes.py`, `GEMPACK_KIND` in `templates/gtap/shocks.py`).

## Consequences

Reopen this when a third party publishes blocks in its own package without
touching this repo: that is the one case where registering by decorator is
needed. Design it then from `_block_classes` and `_mk_unit`, not by restoring the
removed version.

## Sources

- `docs/architecture/registro_de_bloques.md`
