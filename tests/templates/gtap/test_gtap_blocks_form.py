"""Aggregate form + domain gate: GTAP block units vs the monolith oracle (F3 Task 4).

Builds the gtap7_3x3 monolith (``if_sub=False`` comp-stat form, so every defining
equation is active) as the parity ORACLE, composes the migrated block units onto a
single equilibria Model, translates that to Pyomo, and asserts for every migrated
unit:

  * ``form_diff`` clean  — each block-built constraint cell's expanded Pyomo
    expression string-matches the monolith's (structural, cross-model — see
    ``blocks_diag._exprs_equal``).
  * ``domain_bounds_diff`` clean on the unit's OWNED vars — domain label + bounds
    tuple identical to the oracle (per-cell price floors included).

Units land incrementally; ``_MIGRATED`` lists those whose block exists and is gated.
As units 3-7 land, extend ``_MIGRATED`` (and any upstream stub they still need).

No solve — form + domain only. gd_share/ge_share drift (Blocker-C carry) is inert
on gtap7_3x3 (all omegax=inf, so gd/ge never enter an active Leontief body).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]
for _p in (ROOT / "src", ROOT / "scripts" / "gtap"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

# (module_attr, upstream stub var names it references but does not own)
_MIGRATED: list[tuple[str, list[str]]] = [
    ("TradeCETBlock", ["xds", "xs", "pd", "ps"]),
    ("ProductionSupplyBlock", ["pd", "ps", "pa", "pfa"]),
]

# Upstream shared vars a leaf unit references but a later unit owns — stubbed so
# the block builds standalone. (domains, domain_label). Not gated for domain.
_STUBS: dict[str, tuple[tuple[str, ...], str]] = {
    "xds": (("r", "i"), "NonNegativeReals"),
    "xs": (("r", "i"), "NonNegativeReals"),
    "pd": (("r", "i"), "NonNegativeReals"),
    "ps": (("r", "i"), "NonNegativeReals"),
    "pa": (("r", "i", "aa"), "NonNegativeReals"),
    "pfa": (("r", "f", "a"), "NonNegativeReals"),
}


def _oracle():
    from _parity_datasets import build_models

    m_b, _m_s = build_models("gtap7_3x3", close=False)
    return m_b


def _params_sets():
    from equilibria.templates.gtap import GTAPParameters

    p = GTAPParameters()
    d = ROOT / "datasets" / "gtap7_3x3"
    p.load_from_har(
        basedata_path=d / "basedata.har",
        sets_path=d / "sets.har",
        default_path=d / "default.prm",
        baserate_path=d / "baserate.har",
    )
    return p, p.sets


def _set_elems(sets: Any) -> dict[str, list[str]]:
    agents = ["hhd", "gov", "inv", "tmg"]
    return {
        "r": list(sets.r),
        "i": list(sets.i),
        "a": list(sets.a),
        "f": list(sets.f),
        "mf": list(sets.mf),
        "sf": list(sets.sf),
        "m": list(sets.m),
        "marg": list(sets.marg),
        "aa": list(sets.a) + agents,
    }


def _build_block_model(block_classes, p, sets, stub_names):
    from equilibria.backends.pyomo_backend import PyomoBackend
    from equilibria.core.sets import Set as ESet
    from equilibria.core.variables import Variable
    from equilibria.model import Model

    setmap = _set_elems(sets)
    model = Model(name="gtap_blocks")
    for name, elems in setmap.items():
        model.add_set(ESet(name=name, elements=elems))
    for cls in block_classes:
        model.add_block(cls(sets=sets, params=p))
    for n in stub_names:
        if n in model.variable_manager:
            continue
        doms, dlabel = _STUBS[n]
        shape = tuple(len(setmap[d]) for d in doms)
        model.add_variable(
            Variable(
                name=n,
                value=np.ones(shape),
                domains=doms,
                domain=dlabel,
                lower=1e-3,
                upper=float("inf"),
            )
        )
    backend = PyomoBackend()
    backend.build(model)
    return backend.pyomo_model, model


@pytest.fixture(scope="module")
def _fixtures():
    import equilibria.blocks.gtap as gtap_blocks

    p, sets = _params_sets()
    oracle = _oracle()
    classes = [getattr(gtap_blocks, name) for name, _ in _MIGRATED]
    stubs = sorted({s for _, sub in _MIGRATED for s in sub})
    bm, model = _build_block_model(classes, p, sets, stubs)
    return p, sets, oracle, bm, model, gtap_blocks


@pytest.mark.parametrize("unit_name", [n for n, _ in _MIGRATED])
def test_gtap_block_form_matches_monolith(_fixtures, unit_name):
    from blocks_diag import form_diff  # ty: ignore[unresolved-import]
    from pyomo.environ import Constraint

    _p, _sets, oracle, bm, _model, gtap_blocks = _fixtures
    unit = getattr(gtap_blocks, unit_name)
    # discover the equations this unit contributes (its build_expression names)
    eqs = unit(sets=_sets, params=_p).setup(_ScratchSM(_sets), {}, {})
    eq_names = {e.name for e in eqs}

    bm_cons = {c.name: c for c in bm.component_objects(Constraint, active=True)}
    or_cons = {c.name: c for c in oracle.component_objects(Constraint, active=True)}

    checked = 0
    for eq in sorted(eq_names):
        con_name = f"{eq}_con"
        assert con_name in bm_cons, f"{unit_name}: {con_name} missing from block model"
        assert eq in or_cons, f"{unit_name}: {eq} missing from oracle"
        diffs = form_diff(bm_cons[con_name], or_cons[eq])
        assert diffs == [], (
            f"{unit_name} {eq}: {len(diffs)} form diff(s); first: {diffs[0]}"
        )
        checked += 1
    assert checked == len(eq_names)


@pytest.mark.parametrize("unit_name, sub", _MIGRATED)
def test_gtap_block_domain_matches_monolith(_fixtures, unit_name, sub):
    from blocks_diag import domain_bounds_diff  # ty: ignore[unresolved-import]

    _p, _sets, oracle, bm, _model, gtap_blocks = _fixtures
    unit = getattr(gtap_blocks, unit_name)
    owned: dict[str, Any] = {}
    unit(sets=_sets, params=_p).setup(_ScratchSM(_sets), {}, owned)
    owned_names = set(owned)

    diffs = [d for d in domain_bounds_diff(bm, oracle) if d[0] in owned_names]
    assert diffs == [], (
        f"{unit_name}: domain/bounds mismatch on owned vars: {diffs[:3]}"
    )


class _ScratchSM:
    """Minimal SetManager stand-in for introspecting a block's owned symbols."""

    def __init__(self, sets: Any) -> None:
        self._m = _set_elems(sets)

    def get(self, name: str):
        return self._m[name]
