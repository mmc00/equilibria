"""The Pyomo bridge must emit and extract N-dimensional params & vars (F3 task 3.5).

GTAP needs 3-D and 4-D components pervasively (28 three-index vars + ``xmgm``
which is four-index; 15 three-index params + two four-index params). Before this
task the bridge capped ``_build_parameters`` / ``_build_variables`` / the
solve()-extraction loop at 2-D: a 3-D+ component fell through with an EMPTY
initialize dict (silently under-initialized) and extraction raised
``NotImplementedError``.

These tests stand up a small model with a 3-D var, a 3-D param, and a 4-D var
(the ``xmgm`` shape), and assert every cell is initialized on build and that a
3-D var round-trips through a real solve.

The extraction round-trip uses a REAL IPOPT solve (``/opt/homebrew/bin/ipopt``
is present in this worktree) rather than a direct ``value()`` read, because a
real solve exercises the exact ``solve()`` extraction path the task generalizes
(the ``NotImplementedError`` for 3-D+ lived there). The trivial constraint makes
the seed feasible so the solver leaves the init values in place, letting us
assert the extracted array equals the numpy init.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from equilibria import Model
from equilibria.backends import PyomoBackend
from equilibria.core import Parameter, Set, Variable
from equilibria.core.symbolic_equations import SymbolicEquation

# Distinct, non-trivial values per cell so a transposed fill would be caught.
_VAR3D_INIT = np.array(
    [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=float
)
# Param set equal to the var init so the identity constraint holds at the seed.
_PARAM3D_VALUES = _VAR3D_INIT.copy()
# 4-D var (the xmgm shape: domains ("R","A","I","I")); 16 distinct cells.
_VAR4D_INIT = np.arange(1.0, 17.0).reshape(2, 2, 2, 2).astype(float)


class _Var3dEqualsParam3d(SymbolicEquation):
    """One trivial constraint per (r,a,i): myvar3d[r,a,i] == myparam3d[r,a,i].

    Satisfiable at the seed (param == var init), so a real solve leaves the
    init values in place and the extracted array must equal the numpy init.
    """

    name: str = "var3d_eq_param3d"
    domains: tuple = ("R", "A", "I")
    description: str = "3-D identity constraint to make the model non-empty."

    def build_expression(self, pyomo_model: Any, indices: tuple[str, ...]) -> Any:
        r, a, i = indices
        return pyomo_model.myvar3d[r, a, i] == pyomo_model.myparam3d[r, a, i]


def _ndim_model() -> Model:
    """A Model with 3 small sets, a 3-D var+param, a 4-D var, and one constraint."""
    model = Model(name="NDimModel")
    model.add_set(Set(name="R", elements=("r1", "r2")))
    model.add_set(Set(name="A", elements=("a1", "a2")))
    model.add_set(Set(name="I", elements=("i1", "i2")))

    model.add_variable(
        Variable(name="myvar3d", value=_VAR3D_INIT.copy(), domains=("R", "A", "I"))
    )
    model.add_parameter(
        Parameter(
            name="myparam3d", value=_PARAM3D_VALUES.copy(), domains=("R", "A", "I")
        )
    )
    # 4-D var covering the xmgm case (domains ("R","A","I","I")).
    model.add_variable(
        Variable(name="myvar4d", value=_VAR4D_INIT.copy(), domains=("R", "A", "I", "I"))
    )

    model.add_equation(_Var3dEqualsParam3d())  # ty: ignore[invalid-argument-type]
    return model


@pytest.fixture
def built_backend() -> PyomoBackend:
    """A PyomoBackend with build() already run on the N-D model."""
    backend = PyomoBackend()
    backend.build(_ndim_model())
    return backend


def test_build_3d_var_initializes_every_cell(built_backend: PyomoBackend) -> None:
    """All 8 cells of the 3-D Var carry their numpy init (init_dict fully populated)."""
    from pyomo.environ import value

    pm = built_backend.pyomo_model
    sets = (("r1", "r2"), ("a1", "a2"), ("i1", "i2"))
    for ri, r in enumerate(sets[0]):
        for ai, a in enumerate(sets[1]):
            for ii, i in enumerate(sets[2]):
                got = value(pm.myvar3d[r, a, i])
                assert got is not None
                assert got == pytest.approx(_VAR3D_INIT[ri, ai, ii])


def test_build_3d_param_initializes_every_cell(built_backend: PyomoBackend) -> None:
    """All 8 cells of the 3-D Param carry their numpy value."""
    from pyomo.environ import value

    pm = built_backend.pyomo_model
    sets = (("r1", "r2"), ("a1", "a2"), ("i1", "i2"))
    for ri, r in enumerate(sets[0]):
        for ai, a in enumerate(sets[1]):
            for ii, i in enumerate(sets[2]):
                got = value(pm.myparam3d[r, a, i])
                assert got == pytest.approx(_PARAM3D_VALUES[ri, ai, ii])


def test_build_4d_var_initializes_every_cell(built_backend: PyomoBackend) -> None:
    """All 16 cells of the 4-D Var (xmgm shape) are initialized to the numpy init."""
    from pyomo.environ import value

    pm = built_backend.pyomo_model
    sets = (("r1", "r2"), ("a1", "a2"), ("i1", "i2"), ("i1", "i2"))
    for r0, e0 in enumerate(sets[0]):
        for r1, e1 in enumerate(sets[1]):
            for r2, e2 in enumerate(sets[2]):
                for r3, e3 in enumerate(sets[3]):
                    got = value(pm.myvar4d[e0, e1, e2, e3])
                    assert got is not None
                    assert got == pytest.approx(_VAR4D_INIT[r0, r1, r2, r3])


def test_extract_3d_var_roundtrips() -> None:
    """A real solve extracts the 3-D var as a (2,2,2) array matching the init.

    Uses IPOPT (present in this worktree). The identity constraint is feasible at
    the seed, so the solver leaves the init values, and the extracted array must
    equal the numpy init cell-for-cell (this proves the fill order is not
    transposed).
    """
    backend = PyomoBackend(solver="ipopt")
    backend.build(_ndim_model())
    solution = backend.solve()

    arr = solution.variables["myvar3d"]
    assert arr.shape == (2, 2, 2)
    assert np.allclose(arr, _VAR3D_INIT)

    arr4 = solution.variables["myvar4d"]
    assert arr4.shape == (2, 2, 2, 2)
    assert np.allclose(arr4, _VAR4D_INIT)
