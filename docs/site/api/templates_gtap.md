# GTAP template

Public surface of `equilibria.templates.gtap`. An exercise is
`Exercise(GTAP(...), gtap_static(...), shock=...)`: the `GTAP` template builds the
model from the blocks in `equilibria.blocks.gtap`, the `gtap_static` recipe says
how to solve its periods. The old single-file model (`gtap_model_equations`) is not
documented here.

```{eval-rst}
.. automodule:: equilibria.core.exercise
   :members: Exercise

.. automodule:: equilibria.templates.gtap.template
   :members: GTAP

.. automodule:: equilibria.templates.gtap.recipe
   :members: gtap_static

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

.. automodule:: equilibria.templates.gtap.welfare_regional
   :members: regional_ev

.. automodule:: equilibria.templates.gtap.welfare_decomp
   :members:
```
