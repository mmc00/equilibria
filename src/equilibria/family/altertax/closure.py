"""ALTERTAX closure on the family route: the table, the exp parser and the closure.

``closure_map.yaml`` maps every exogenous name of GEMPACK's ``altax.exp`` (plus the
numeraire) to its equilibria counterpart; ``altertax_closure`` is the config that table
decides. No Pyomo here.
"""

from __future__ import annotations

import re
from pathlib import Path

import yaml

from equilibria.templates.gtap.gtap_contract import GTAPClosureConfig

# nus333 has two regions (USA, ROW). In altax.exp del_tbal("USA") is exogenous (USA's
# savf fixed under capFix) and cgdslack("ROW") is exogenous: ROW is the region whose
# balance is left free, so ROW is the residual region of the capital account.
# The closure carries no residual region: Task 2 passes it explicitly, as
# ``solve_gtap_family(params, altertax_closure(), ALTAX_RESIDUAL_REGION, ...)``.
ALTAX_RESIDUAL_REGION = "ROW"

MAP_PATH = Path(__file__).with_name("closure_map.yaml")


def load_closure_map() -> list[dict]:
    """The rows of ``closure_map.yaml``: ``gempack``, ``equilibria``, ``kind``, ``reason``."""
    with MAP_PATH.open() as f:
        return yaml.safe_load(f)


def parse_exp_exogenous(exp_path: Path) -> list[str]:
    """Names in the ``Exogenous`` block of a GEMPACK ``.exp`` (to the closing ``;``), with
    their quoted element when there is one, e.g. ``del_tbal("USA")``. ``!`` starts a
    comment."""
    text = "\n".join(
        line.split("!", 1)[0] for line in Path(exp_path).read_text().splitlines()
    )
    m = re.search(r"(?im)^\s*exogenous\b(.*?);", text, re.S)
    if m is None:
        raise ValueError(f"{exp_path}: no Exogenous block")
    names: list[str] = []
    for tok in m.group(1).split():
        if tok not in names:
            names.append(tok)
    return names


def altertax_closure() -> GTAPClosureConfig:
    """The route's closure (``_solve_shock``'s) under the name ``altertax_gempack``:
    ``prep_square`` refuses the name ``altertax`` (Malcolm's closure). Taxes, technology
    and endowments are ShockBlock instruments (fixed per period), not closure flags.
    The residual region is not a field here: pass ``ALTAX_RESIDUAL_REGION`` explicitly.
    ``unit_sigma="gempack"``: the elasticities ``altertax.prm`` sets to 1 are exact
    Cobb-Douglas nests, not the 1.01 stand-in."""
    return GTAPClosureConfig(
        name="altertax_gempack",
        closure_type="MCP",
        capital_mobility="sluggish",
        fix_endowments=False,
        fix_taxes=False,
        fix_technology=False,
        if_sub=True,
        savf_flag="capFix",
        numeraire="pnum",
        va_subsidy_basis="gempack",
        unit_sigma="gempack",
    )
