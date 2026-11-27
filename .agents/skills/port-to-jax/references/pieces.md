# The generic family pieces and the template wiring

| Piece | Where | What it does |
|---|---|---|
| `build_arrays(model, instruments, periods)` | `src/equilibria/family/generic_arrays.py` | records the blocks (`FamilyBackend`), lays every var and row out per period, fixes the instruments and makes them exogenous, takes bounds from `StateModel.lower/upper` |
| `GenericFamilyPeriodModel(mp, backend, state, periods)` | `src/equilibria/family/generic_periods.py` | the `PeriodModel` protocol, a real `residual(period)` (max over active rows) and `period_system()` |
| `newton_family(tol=1e-10)` | `src/equilibria/solver/newton_family.py` | the recipe solver: finds the active period, refuses a non-square one, calls `newton_tr` with bounds and an explicit `ftol` |

Template wiring (copy `SimpleOpenEconomy` in `src/equilibria/templates/simple_open/template.py`):

- a field `backend: Literal["pyomo", "family"]`, default `"pyomo"`;
- `build()` returns `GenericFamilyPeriodModel(mp, backend, state, periods)` from `build_arrays(self.create_model(overwrite=overwrite), INSTRUMENTS, periods)`, fresh on every call, and falls back to `super().build(...)` for Pyomo;
- `default_recipe()` passes `solver=newton_family()` on the family backend;
- steps use only the `PeriodModel` protocol (`src/equilibria/core/period_model.py`).

Tests to copy: `tests/family/test_generic_arrays.py`, `test_generic_periods.py`, `test_newton_family.py`, `test_simple_open_family.py`, `test_simple_open_gams_parity.py`.
