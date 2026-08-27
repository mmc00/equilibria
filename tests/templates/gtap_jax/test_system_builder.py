"""build_F assembles the per-cell translations into ONE vectorized JAX residual F(z) over
the whole squared system, and F(z) matches Pyomo's constraint residuals elementwise."""
import sys
from pathlib import Path

import numpy as np
import pytest

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts" / "gtap"))

from equilibria.templates.gtap_jax.system_builder import build_F


def _available(ds):
    return (Path("datasets") / ds).exists()


def _build_model(ds):
    from equilibria.templates.gtap import GTAPParameters
    from equilibria.templates.gtap.gtap_contract import GTAPClosureConfig
    from equilibria.templates.gtap_sparse.multiperiod import build_sparse_model_mp

    d = Path("datasets") / ds
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


def _z_in_index_order(m, var_index):
    """The variable value vector z ordered by var_index[id(v)] -> slot."""
    from pyomo.core.base.var import Var
    from pyomo.environ import value
    slot_of = var_index
    n = len(slot_of)
    z = np.empty(n)
    for v in m.component_data_objects(Var, active=True):
        s = slot_of.get(id(v))
        if s is not None:
            z[s] = value(v) if v.value is not None else 1.0
    return jnp.asarray(z)



def _pyomo_residual(cons_order):
    """Pyomo residual = body - rhs (what pynumero's evaluate_eq_constraints returns), matching
    translate_constraint. body alone omits a nonzero RHS (e.g. sum==1.0)."""
    from pyomo.environ import value
    import numpy as np
    out = []
    for c in cons_order:
        rhs = 0.0
        if c.lower is not None: rhs = float(value(c.lower))
        elif c.upper is not None: rhs = float(value(c.upper))
        out.append(value(c.body) - rhs)
    return np.array(out)

@pytest.mark.parametrize("ds", ["gtap7_3x3", "gtap7_10x7"])
def test_F_matches_pyomo_elementwise(ds):
    if not _available(ds):
        pytest.skip(f"dataset {ds} not available")
    from pyomo.environ import value

    m = _build_model(ds)
    F, var_index, cons_order = build_F(m)
    z = _z_in_index_order(m, var_index)

    f_jax = np.asarray(F(z))
    f_pyo = _pyomo_residual(cons_order)
    assert f_jax.shape == f_pyo.shape, f"shape {f_jax.shape} vs {f_pyo.shape}"
    diff = np.max(np.abs(f_jax - f_pyo))
    assert diff < 1e-9, f"max elementwise diff {diff:.2e} on {ds}"


@pytest.mark.parametrize("ds", ["gtap7_3x3", "gtap7_10x7"])
def test_vmap_F_matches_pyomo(ds):
    """F built the vmap way (vectorize=True) must still match Pyomo elementwise <1e-9."""
    if not _available(ds):
        pytest.skip(f"dataset {ds} not available")
    from pyomo.environ import value

    m = _build_model(ds)
    F, var_index, cons_order = build_F(m, vectorize=True)
    z = _z_in_index_order(m, var_index)
    f_jax = np.asarray(F(z))
    f_pyo = _pyomo_residual(cons_order)
    assert f_jax.shape == f_pyo.shape
    # HYBRID: build_F leaves the macro-closure rows (pyomo_rows — huge trees, evaluated by
    # Pyomo downstream) as 0; exclude them from the JAX-only parity check.
    mask = np.ones(len(cons_order), dtype=bool)
    mask[np.asarray(getattr(F, "pyomo_rows", []), dtype=int)] = False
    diff = np.max(np.abs(f_jax[mask] - f_pyo[mask])) if mask.any() else 0.0
    assert diff < 1e-9, f"vmap F (JAX rows) max diff {diff:.2e} on {ds}"


@pytest.mark.parametrize("ds", ["gtap7_10x7"])
def test_vmap_F_second_eval_is_fast(ds):
    """After compile, a second eval is fast — proves it's jitted, not re-tracing 12k fns."""
    if not _available(ds):
        pytest.skip(f"dataset {ds} not available")
    import time
    m = _build_model(ds)
    F, var_index, cons_order = build_F(m, vectorize=True)
    z = _z_in_index_order(m, var_index)
    F(z).block_until_ready()  # compile
    t = time.perf_counter()
    F(z).block_until_ready()
    assert time.perf_counter() - t < 0.5, "second eval too slow — not properly jitted/vmapped"


@pytest.mark.parametrize("ds", ["gtap7_3x3"])
def test_F_is_jittable_and_differentiable(ds):
    if not _available(ds):
        pytest.skip(f"dataset {ds} not available")
    m = _build_model(ds)
    F, var_index, cons_order = build_F(m)
    z = _z_in_index_order(m, var_index)
    # F already jitted; a jacobian-vector product must work (autodiff over the whole system)
    _, jvp = jax.jvp(F, (z,), (jnp.ones_like(z),))
    assert jvp.shape == (len(cons_order),)
    assert jnp.any(jvp != 0)
