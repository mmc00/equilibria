"""Prototype validation: the Pyomo->JAX translator reproduces a REAL GTAP equation exactly.

Builds a real GTAP model, takes real eq_nd constraint bodies, translates each to JAX, and
checks the JAX residual equals Pyomo's residual at the model's current point (byte-close).
If this holds on the real equation, the tree-translation approach (Option A) is validated
for the spec. Runs on CPU (no GPU needed — correctness is hardware-independent).
"""
import os
import sys
from pathlib import Path

import numpy as np
import pytest

# JAX in float64 so the comparison is meaningful
import jax
jax.config.update("jax_enable_x64", True)

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts" / "gtap"))

from equilibria.templates.gtap_jax.expr_to_jax import translate

_DS = None
for cand in ("gtap7_3x3", "gtap6_10x7", "gtap7_10x7"):
    if (Path("datasets") / cand).exists():
        _DS = cand
        break


def _build_model():
    from equilibria.templates.gtap import GTAPParameters
    from equilibria.templates.gtap.gtap_contract import GTAPClosureConfig
    from equilibria.templates.gtap_sparse.multiperiod import build_sparse_model_mp

    d = Path("datasets") / _DS
    p = GTAPParameters()
    p.load_from_har(
        basedata_path=d / "basedata.har", sets_path=d / "sets.har",
        default_path=d / "default.prm", baserate_path=d / "baserate.har",
    )
    rr = list(p.sets.r)[-1]
    ac = GTAPClosureConfig(
        name="base", closure_type="MCP", capital_mobility="sluggish",
        fix_endowments=False, fix_taxes=False, fix_technology=False,
        if_sub=False, savf_flag="capFix", numeraire="pnum",
    )
    m, mp, _ = build_sparse_model_mp(p, p.sets, ac, rr, base_calibrated=False, ref_gdx=None)
    return m


@pytest.mark.skipif(_DS is None, reason="no GTAP dataset available")
def test_eq_nd_jax_matches_pyomo():
    from pyomo.core.base.constraint import Constraint
    from pyomo.core.base.var import Var
    from pyomo.environ import value

    m = _build_model()

    # global variable index: id(vardata) -> slot, and the current value vector z
    vars_list = [v for v in m.component_data_objects(Var, active=True)]
    var_index = {id(v): i for i, v in enumerate(vars_list)}
    z = np.array([value(v) if v.value is not None else 1.0 for v in vars_list], dtype=float)
    import jax.numpy as jnp
    z_j = jnp.asarray(z)

    checked = 0
    for c in m.component_data_objects(Constraint, active=True):
        if not str(c.name).startswith("eq_nd"):
            continue
        # Pyomo residual = value(body) (body is lhs - rhs == 0 form)
        pyo_res = value(c.body)
        # JAX residual
        f = translate(c.body, var_index)
        jax_res = float(f(z_j))
        assert abs(pyo_res - jax_res) < 1e-9, (
            f"{c.name}: pyomo={pyo_res:.6e} jax={jax_res:.6e} diff={abs(pyo_res-jax_res):.2e}"
        )
        checked += 1
        if checked >= 20:
            break
    assert checked > 0, "no eq_nd constraints found"


@pytest.mark.skipif(_DS is None, reason="no GTAP dataset available")
def test_translated_is_differentiable():
    """The translated function must be jax-differentiable (grad works) — this is what makes
    the Jacobian free via autodiff/sparsejac."""
    from pyomo.core.base.constraint import Constraint
    from pyomo.core.base.var import Var
    from pyomo.environ import value
    import jax.numpy as jnp

    m = _build_model()
    vars_list = [v for v in m.component_data_objects(Var, active=True)]
    var_index = {id(v): i for i, v in enumerate(vars_list)}
    z = jnp.asarray([value(v) if v.value is not None else 1.0 for v in vars_list], dtype=float)

    for c in m.component_data_objects(Constraint, active=True):
        if str(c.name).startswith("eq_nd"):
            f = translate(c.body, var_index)
            g = jax.grad(f)(z)  # must not raise
            assert g.shape == z.shape
            assert jnp.any(g != 0), "gradient is all-zero — translation likely dropped the vars"
            return
