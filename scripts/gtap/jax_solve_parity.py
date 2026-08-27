"""End-to-end parity: solve the GTAP 3x3 with EQUILIBRIA_GTAP_EVAL=jax and confirm it reaches
the SAME solution as the default Pyomo path. Runs the real driver (newton_tr) on the squared
system. Needs GAMS env for path_capi. Not a pytest (external deps).

Run:
  DYLD_LIBRARY_PATH="/Library/Frameworks/GAMS.framework/Versions/53/Resources" \
  EQUILIBRIA_GTAP_EVAL=jax EQUILIBRIA_GTAP_SOLVE_NLP=1 \
  EQUILIBRIA_GTAP_SOLVER=scipy_newton_tr EQUILIBRIA_GTAP_TR_LINSOLVE=mumps \
  PYTHONPATH=src:scripts/gtap <venv>/python scripts/gtap/jax_solve_parity.py [gtap7_3x3]
"""
import os
import sys
import time
from pathlib import Path

import numpy as np


def main():
    ds = sys.argv[1] if len(sys.argv) > 1 else "gtap7_3x3"
    from equilibria.templates.gtap import GTAPParameters
    from equilibria.templates.gtap.gtap_contract import GTAPClosureConfig
    from equilibria.templates.gtap_sparse.multiperiod import build_sparse_model_mp
    from equilibria.templates.gtap.gtap_block_model import solve_block_model
    from pyomo.environ import value

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
    t = time.perf_counter()
    m, mp, _ = build_sparse_model_mp(p, p.sets, ac, rr, base_calibrated=True, ref_gdx=None)
    res = solve_block_model(m, p, ac, ref_gdx=None, mode="gtap")
    dt = time.perf_counter() - t
    eval_mode = os.environ.get("EQUILIBRIA_GTAP_EVAL", "pyomo")
    print(f"[jax-parity] ds={ds} eval={eval_mode} wall={dt:.1f}s")
    # dump a few solved var values so the two runs can be diffed
    from pyomo.core.base.var import Var
    vals = {}
    for v in m.component_data_objects(Var, active=True):
        if not v.fixed and v.value is not None:
            vals[str(v.name)] = float(v.value)
    out = Path(f"/tmp/jax_parity_{eval_mode}_{ds}.txt")
    with open(out, "w") as f:
        for k in sorted(vals):
            f.write(f"{k}\t{vals[k]:.10e}\n")
    print(f"[jax-parity] wrote {len(vals)} var values -> {out}")


if __name__ == "__main__":
    main()
