# GTAP template

Public surface of `equilibria.templates.gtap`. The equation monolith
(`gtap_model_equations`, ~6k lines) is deliberately not auto-documented —
its extraction into `equilibria.blocks` is roadmap phase F3.

```{note}
`gtap_multiperiod_driver` (the `loop(tsim)` base→check→shock driver) is
excluded from the generated reference until its module docstring is valid
reST (unbalanced `*` emphasis breaks the strict docs build).
```

```{eval-rst}
.. automodule:: equilibria.templates.gtap.gtap_contract
   :members:

.. automodule:: equilibria.templates.gtap.gtap_parameters
   :members:

.. automodule:: equilibria.templates.gtap.gtap_sets
   :members:

.. automodule:: equilibria.templates.gtap.gtap_solver
   :members:

.. automodule:: equilibria.templates.gtap.shocks
   :members:

.. automodule:: equilibria.templates.gtap.welfare_decomp
   :members:
```
