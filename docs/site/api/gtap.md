# GTAP door (`equilibria.gtap`)

The public way in to GTAP: everything to load a dataset, build, shock and read an
exercise. It only gathers the existing pieces (ADR-0012), so every name below is the
same object as in its own module (documented in {doc}`templates_gtap`), except
`load` and `results`, which are new.

```python
from equilibria.gtap import ClosureConfig, Exercise, GTAP, gtap_static, load, results

ex = Exercise(
    GTAP(load("nus333"), ClosureConfig.gtap_default(), residual_region="ROW"),
    gtap_static("gtap", solve_check=True),
    shock={"imptx": {("USA", "MFG", "ROW"): 10.0}},
)
ex.run()
results(ex)
```

The guides: {doc}`../guide/gtap_quickstart` and {doc}`../guide/gtap_closures`.

The named starting point of the closure is
{py:meth}`~equilibria.templates.gtap.gtap_contract.GTAPClosureConfig.gtap_default`,
documented with the closure class in {doc}`templates_gtap`.

```{eval-rst}
.. automodule:: equilibria.gtap

.. autodata:: equilibria.gtap.overwrite
   :no-value:

   The ``@overwrite`` registry for advanced closures (ADR-0004), the same object
   as ``equilibria.blocks.gtap.overwrite``: ``@overwrite(ShockBlock, period="shock")``
   changes one block in one period; ``overwrite.clear()`` drops the hooks of a
   previous exercise. Examples in :doc:`../guide/gtap_closures`.
```

## The door contract

Every model door follows the same contract; `DOORS` lists the doors and
`PENDING` the templates that have none yet, with the reason.

```{eval-rst}
.. automodule:: equilibria.core.door
   :members: DoorSpec, DOORS, PENDING, check_door, find_templates, coverage_problems
```
