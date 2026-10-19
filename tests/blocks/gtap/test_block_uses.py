"""``Block.uses`` de cada bloque GTAP es exactamente lo que sus ecuaciones leen.

Un bloque lee variables de otros por nombre (``model.pabs[r]`` en FactorBlock,
que declara IncomeBlock). ``uses`` lo hace explicito y ``Model.check_uses`` falla
con un mensaje claro si falta uno. Aca se mide lo que cada ecuacion lee de verdad
(un proxy del modelo Pyomo anota cada atributo que es una Var) y se compara con
la lista declarada: ni falta una variable ni sobra una vieja.

Se mide sobre la union de los cierres que cambian que ecuaciones existen: ifSUB
(las macros M_* se sustituyen dentro de las filas) y la regla de la cuenta de
capital (``savf_flag``).
"""

from __future__ import annotations

import itertools

import pytest
from pyomo.environ import Var
from tests.templates.gtap._nus333 import closure, nus333_params

from equilibria import model as eq_model
from equilibria.backends import pyomo_backend
from equilibria.templates.gtap.gtap_block_model import build_block_single_period

CONFIGS = list(itertools.product((False, True), ("capFix", "capFlex", "capSFix")))


class _Reads:
    """El modelo Pyomo, anotando que Var lee cada ecuacion."""

    def __init__(self, pm):
        object.__setattr__(self, "_pm", pm)
        object.__setattr__(self, "names", set())

    def __getattr__(self, name):
        obj = getattr(self._pm, name)
        if isinstance(obj, Var):
            self.names.add(name)
        return obj


def _foreign_reads(params, if_sub, savf_flag):
    """{bloque: Vars que sus ecuaciones leen y que el no declara}."""
    declared: dict[str, set[str]] = {}
    eqs: dict[str, set[str]] = {}
    built: dict[str, eq_model.Model] = {}
    add_block, build = eq_model.Model.add_block, pyomo_backend.PyomoBackend.build

    def spy_add(self, block):
        before = set(self.variable_manager.list_vars())
        n_eq = set(self.equation_manager.list_equations())
        add_block(self, block)
        mine = {
            v
            for v in self.variable_manager.list_vars()
            if v not in before or self.owner_of("variable", v) == block.name
        }
        # Las compartidas identicas tambien son suyas: las declara.
        bp, bv = {}, {}
        block.setup(self.set_manager, bp, bv)
        declared[block.name] = mine | set(bv)
        eqs[block.name] = set(self.equation_manager.list_equations()) - n_eq

    def spy_build(self, model):
        built["model"] = model
        return build(self, model)

    mp = pytest.MonkeyPatch()
    mp.setattr(eq_model.Model, "add_block", spy_add)
    mp.setattr(pyomo_backend.PyomoBackend, "build", spy_build)
    try:
        gc = closure(savf_flag).model_copy(update={"if_sub": if_sub})
        pm = build_block_single_period(params, params.sets, gc, "ROW")
    finally:
        mp.undo()

    model = built["model"]
    out = {}
    for block in model.blocks:
        proxy = _Reads(pm)
        for eq_name in eqs[block.name]:
            eq = model.equation_manager.get(eq_name)
            for idx in eq.get_indices(model.set_manager):
                eq.build_expression(proxy, idx)
        out[type(block).__name__] = proxy.names - declared[block.name]
    return out, {type(b).__name__: set(b.uses) for b in model.blocks}


@pytest.fixture(scope="module")
def measured():
    params = nus333_params()
    reads: dict[str, set[str]] = {}
    uses = None
    for if_sub, savf in CONFIGS:
        got, uses = _foreign_reads(params, if_sub, savf)
        for k, v in got.items():
            reads.setdefault(k, set()).update(v)
    return reads, uses


def test_every_block_lists_what_it_reads_from_others(measured):
    reads, uses = measured
    missing = {b: sorted(reads[b] - uses[b]) for b in reads if reads[b] - uses[b]}
    assert not missing, f"used but not in Block.uses: {missing}"


def test_no_block_lists_a_variable_it_no_longer_reads(measured):
    reads, uses = measured
    stale = {b: sorted(uses[b] - reads[b]) for b in uses if uses[b] - reads[b]}
    assert not stale, f"in Block.uses but never read: {stale}"
