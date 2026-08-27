"""The Pyomo->JAX translator reproduces the REAL GTAP equations exactly.

Builds real GTAP models (3x3 and 10x7), translates EVERY constraint family's body to JAX,
and checks the JAX residual equals Pyomo's residual at the seed point. Validates the
tree-translation approach (Approach A) across all ~96 families. CPU-only (correctness is
hardware-independent). float64 so the comparison is meaningful.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts" / "gtap"))

from equilibria.templates.gtap_jax.expr_to_jax import translate


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


def _seed(m):
    from pyomo.core.base.var import Var
    from pyomo.environ import value
    vars_list = [v for v in m.component_data_objects(Var, active=True)]
    var_index = {id(v): i for i, v in enumerate(vars_list)}
    z = jnp.asarray([value(v) if v.value is not None else 1.0 for v in vars_list], dtype=float)
    return var_index, z


@pytest.mark.parametrize("ds", ["gtap7_3x3", "gtap7_10x7"])
def test_all_families_translate_match(ds):
    """Every equation family translates to JAX and matches Pyomo's residual <1e-7."""
    if not _available(ds):
        pytest.skip(f"dataset {ds} not available")
    from pyomo.core.base.constraint import Constraint
    from pyomo.environ import value

    m = _build_model(ds)
    var_index, z = _seed(m)

    seen = set()
    failures = []
    for c in m.component_data_objects(Constraint, active=True):
        fam = str(c.name).split("[")[0]
        if fam in seen:
            continue
        seen.add(fam)
        try:
            f = translate(c.body, var_index)
            jr = float(f(z))
            pr = value(c.body)
            if not (abs(pr - jr) < 1e-7 or abs(pr - jr) < 1e-7 * max(1.0, abs(pr))):
                failures.append(f"{fam}: pyomo={pr:.4e} jax={jr:.4e}")
        except Exception as e:
            failures.append(f"{fam}: {type(e).__name__}: {str(e)[:80]}")
    assert not failures, f"{len(failures)} families failed on {ds}:\n" + "\n".join(failures)
    assert len(seen) >= 90, f"only {len(seen)} families found on {ds}"


@pytest.mark.parametrize("ds", ["gtap7_3x3", "gtap7_10x7"])
def test_translated_is_differentiable(ds):
    """The translated residual is jax-differentiable (grad works, non-zero) — this is what
    makes the Jacobian free via autodiff/sparsejac."""
    if not _available(ds):
        pytest.skip(f"dataset {ds} not available")
    from pyomo.core.base.constraint import Constraint

    m = _build_model(ds)
    var_index, z = _seed(m)
    for c in m.component_data_objects(Constraint, active=True):
        if str(c.name).startswith("eq_nd"):
            f = translate(c.body, var_index)
            g = jax.grad(f)(z)
            assert g.shape == z.shape
            assert jnp.any(g != 0), "gradient all-zero — translation dropped the vars"
            return
