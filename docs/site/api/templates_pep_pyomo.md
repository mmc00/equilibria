# PEP template (Pyomo)

The PEP-1-1 v2.1 port — see the
[templates overview](../guide/templates.md) for context.

```{eval-rst}
.. automodule:: equilibria.templates.pep_pyomo.pep_pyomo_sets
   :members:

.. automodule:: equilibria.templates.pep_pyomo.pep_pyomo_parameters
   :members:

.. automodule:: equilibria.templates.pep_pyomo.pep_pyomo_equations
   :members:

.. automodule:: equilibria.templates.pep_pyomo.pep_pyomo_blocks
   :members:

.. automodule:: equilibria.templates.pep_pyomo.pep_pyomo_scenarios
   :members:
```

```{note}
`pep_pyomo_solver` (the IPOPT/PATH solve driver) is excluded from the
generated reference until its module docstring is valid reST (unbalanced
`*` emphasis breaks the strict docs build).
```
