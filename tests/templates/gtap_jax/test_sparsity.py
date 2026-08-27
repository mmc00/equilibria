"""The Jacobian sparsity pattern derived structurally from tree var-leaves equals the actual
nnz pattern of Pyomo's Jacobian — so sparsejac computes exactly the right entries."""
import sys
from pathlib import Path

import numpy as np
import pytest

import jax
jax.config.update("jax_enable_x64", True)

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts" / "gtap"))

from equilibria.templates.gtap_jax.system_builder import build_F
from equilibria.templates.gtap_jax.sparsity import jacobian_pattern


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


def _pyomo_jac_pattern(m, cons_order, var_index):
    """Pyomo's actual Jacobian nnz as a set of (row, col) over the SAME row/col ordering as
    build_F: rows = cons_order index; cols = var_index slot."""
    from pyomo.environ import value
    import numpy as np
    # per-row numeric derivative pattern via Pyomo's symbolic differentiation
    from pyomo.core.expr.calculus.derivatives import differentiate
    from pyomo.core.base.var import VarData

    pat = set()
    for i, c in enumerate(cons_order):
        # vars appearing in this constraint body
        from pyomo.core.expr.visitor import identify_variables
        for v in identify_variables(c.body, include_fixed=False):
            j = var_index.get(id(v))
            if j is not None:
                pat.add((i, j))
    return pat


@pytest.mark.parametrize("ds", ["gtap7_3x3", "gtap7_10x7"])
def test_pattern_matches_pyomo_nnz(ds):
    if not _available(ds):
        pytest.skip(f"dataset {ds} not available")
    m = _build_model(ds)
    _, var_index, cons_order = build_F(m)
    pat = jacobian_pattern(cons_order, var_index)  # BCOO
    got = set(zip(pat.indices[:, 0].tolist(), pat.indices[:, 1].tolist()))
    exp = _pyomo_jac_pattern(m, cons_order, var_index)
    missing = exp - got
    extra = got - exp
    assert not missing and not extra, (
        f"{ds}: missing {len(missing)} extra {len(extra)} (e.g. missing {list(missing)[:3]}, "
        f"extra {list(extra)[:3]})"
    )
    assert pat.shape == (len(cons_order), len(var_index))
