"""Tamano del modelo GTAP (Seccion 2 del paper): familias de ecuaciones y de variables
del modelo de un periodo y del multiperiodo, 3x3, capFix, ifSUB 0 y 1."""

from pathlib import Path

from pyomo.environ import Constraint, Var

from equilibria.templates.gtap import GTAPParameters
from equilibria.templates.gtap.gtap_block_model import (
    build_block_model,
    build_block_single_period,
)
from equilibria.templates.gtap.gtap_contract import GTAPClosureConfig

d = Path("datasets/gtap7_3x3")
p = GTAPParameters()
p.load_from_har(
    basedata_path=d / "basedata.har",
    sets_path=d / "sets.har",
    default_path=d / "default.prm",
    baserate_path=d / "baserate.har",
)
rr = list(p.sets.r)[-1]


def fam(m):
    return len(list(m.component_objects(Constraint, active=True))), len(
        list(m.component_objects(Var))
    )


for ifsub in (0, 1):
    ac = GTAPClosureConfig(
        name="base",
        closure_type="MCP",
        capital_mobility="sluggish",
        fix_endowments=False,
        fix_taxes=False,
        fix_technology=False,
        if_sub=bool(ifsub),
        savf_flag="capFix",
        numeraire="pnum",
    )
    sp = build_block_single_period(p, p.sets, ac, rr)
    m, _ = build_block_model(p, p.sets, ac, rr)
    (se, sv), (me, mv) = fam(sp), fam(m)
    print(
        f"ifSUB={ifsub}: un periodo {se} familias de ecuaciones / {sv} de variables; "
        f"multiperiodo {me} / {mv}",
        flush=True,
    )
