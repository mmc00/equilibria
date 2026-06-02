"""Inventory GTAP model: list every Var, Param (mutable), and Constraint
with its index dimensions. Used as input for the t-indexing refactor."""
import contextlib, json, os, sys
from pathlib import Path
os.environ['EQUILIBRIA_GTAP_RRES'] = 'USA'
sys.path.insert(0, str(Path('src').resolve()))

from pyomo.environ import Var, Param, Constraint
from equilibria.templates.gtap.gtap_parameters import GTAPParameters
from equilibria.templates.gtap.gtap_model_equations import GTAPModelEquations
from equilibria.templates.gtap.gtap_contract import default_gtap_contract

GDX = Path('datasets/gtap7_15x10/GDX/basedata.gdx').resolve()
contract = default_gtap_contract()


def _index_dims(component):
    """Return tuple of set names for an indexed component, or () if scalar."""
    idx = component.index_set()
    if idx.name == 'AbstractOrderedScalarSet' or component.dim() == 0:
        return ()
    if hasattr(idx, 'subsets'):
        try:
            return tuple(s.name for s in idx.subsets())
        except Exception:
            return (idx.name,)
    return (idx.name,)


# Pyomo W1002 warnings write directly to FD1 (stdout) at the C level, bypassing
# sys.stdout. Use OS-level dup2 to redirect FD1 → FD2 during the build + iteration,
# then restore it before the final print so the JSON goes cleanly to stdout.
_saved_fd1 = os.dup(1)   # save a copy of the real stdout FD
os.dup2(2, 1)            # redirect FD1 → FD2 (stderr)

try:
    p = GTAPParameters(); p.load_from_gdx(GDX)
    eq = GTAPModelEquations(p.sets, p, contract.closure)
    m = eq.build_model()

    vars_out = []
    for v in m.component_objects(Var, active=None):
        vars_out.append({'name': v.name, 'dims': list(_index_dims(v)),
                         'doc': v.doc or ''})

    params_out = []
    for prm in m.component_objects(Param, active=None):
        if not prm.mutable:
            continue
        params_out.append({'name': prm.name, 'dims': list(_index_dims(prm)),
                           'doc': prm.doc or ''})

    cons_out = []
    for c in m.component_objects(Constraint, active=None):
        cons_out.append({'name': c.name, 'dims': list(_index_dims(c)),
                         'doc': c.doc or ''})
finally:
    os.dup2(_saved_fd1, 1)  # restore real stdout FD
    os.close(_saved_fd1)

out = {'vars': vars_out, 'params_mutable': params_out, 'constraints': cons_out,
       'counts': {'vars': len(vars_out), 'params_mutable': len(params_out),
                  'constraints': len(cons_out)}}
print(json.dumps(out, indent=2))
