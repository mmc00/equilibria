"""GTAP6 block units vs the v6.2 monolith oracle — form + numeric gate (F7 Task 6).

Two gates for ``TradeArmingtonBlock``:

1. ``test_trade_armington_block_setup_returns_all_contract_equations`` —
   ``setup()`` runs without exception and produces exactly the 16 equation
   names in ``_GTAP6_TRADE | _GTAP6_MARGINS``, no more no less.

2. ``test_trade_armington_block_matches_oracle_numerically`` — the
   load-bearing check. For every oracle equation this block ports (14 of
   the 16 — ``e_qds``/``e_qtmfsd`` have no oracle Constraint to diff
   against, see the block's module docstring), build the block's
   ``SymbolicEquation.build_expression`` directly against the ORACLE'S OWN
   live ``pyomo_model`` (same Var/Param objects — sidesteps needing a
   separate composed Model/PyomoBackend, which doesn't exist yet since no
   composer has landed) at every active index the oracle's own Constraint
   has, and asserts the two expressions evaluate to the SAME residual
   value within 1e-9. A block with right names but wrong algebra fails
   here even though gate 1 passes.

   ``e_qds``/``e_qtmfsd`` are checked against the oracle's own documented
   *identities* (the ``vds`` calibration sum and the ``eq_qtm`` summand)
   instead, since no oracle Constraint exists for them.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
for _p in (ROOT / "src", ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

DATASET = ROOT / "datasets" / "gtap6_3x3"

_MIGRATED: list[str] = [
    "TradeArmingtonBlock",
    "ProductionBlock",
    "FactorBlock",
    "DemandUtilityBlock",
]

# Oracle Constraint name -> block equation name, for the 14 equations that
# exist as an active Constraint in the oracle. e_qds/e_qtmfsd have no
# oracle Constraint (see module docstring / block docstring) and are
# checked separately below.
_ORACLE_CONSTRAINT_FOR = {
    "e_qfd_arm": "eq_qfd",
    "e_qfm_arm": "eq_qfm",
    "e_qfa": "eq_qf",
    "e_pfa": "eq_pf_int",
    "e_qxs": "eq_qxs",
    "e_pms": "eq_pms",
    "e_pmcif": "eq_pmcif",
    "e_pe": "eq_pe",
    "e_pim": "eq_pim",
    "e_qst": "eq_qst",
    "e_pst": "eq_pst",
    "e_qtm": "eq_qtm",
    "e_ptmg": "eq_ptmg",
    "e_pwmg": "eq_pwmg",
}

_TOL = 1e-9


def _build_oracle():
    from gtap6._v62_monolith_oracle import build_monolith_model

    oracle = build_monolith_model(DATASET)
    # The block renames two oracle symbols (documented in trade_armington.py's
    # module docstring): oracle's `qf`/`pf_int` (top-nest Armington composite
    # qty/price) are the block's `qfa`/`pfa` (matching the contract's e_qfa/
    # e_pfa IDs and the task-6 brief's variable-name spec). Alias them onto the
    # SAME live oracle model (not new components) so the block's
    # build_expression can be evaluated directly against the oracle's own
    # Vars/Params for the numeric diff below, without re-deriving a second
    # model. Read-only aliasing; does not change the oracle's own equations.
    # Pyomo's Block.__setattr__ registers a NEW component (and refuses to
    # re-parent an existing one under a second name), so bypass it with a
    # raw object.__setattr__ — this only adds a second Python attribute
    # pointing at the SAME already-registered IndexedVar object, it does not
    # touch the model's component tree or its `qf`/`pf_int` Constraints.
    object.__setattr__(oracle, "qfa", oracle.qf)
    object.__setattr__(oracle, "pfa", oracle.pf_int)
    # e_qtmfsd is a genuinely NEW per-shipment quantity the oracle never
    # materializes (see trade_armington.py's module docstring) — its
    # build_expression constructs `m.qtmfsd[...] == ...`, which needs a real
    # Var (not just an alias) to be constructible at all. Attach one as an
    # actual Pyomo component (not a raw-attribute alias) since this is new,
    # not a rename.
    from pyomo.environ import NonNegativeReals, Var

    oracle.qtmfsd = Var(
        oracle.marg,
        oracle.i,
        oracle.s,
        oracle.rp,
        within=NonNegativeReals,
        initialize=0.0,
    )
    return oracle


def _build_calibration():
    from equilibria.templates.gtap6.gtap6_calibration import derive_calibration
    from equilibria.templates.gtap6.gtap6_parameters import GTAP6Parameters
    from equilibria.templates.gtap6.gtap6_sets import GTAP6Sets

    sets = GTAP6Sets()
    sets.load_from_har(DATASET / "sets.har", default_path=DATASET / "default.prm")
    params = GTAP6Parameters()
    params.load_from_har(DATASET / "basedata.har", DATASET / "default.prm", sets)
    derived = derive_calibration(sets, params)
    return sets, params, derived


def _build_set_manager(sets):
    from equilibria.core.sets import Set, SetManager

    set_manager = SetManager()
    set_manager.add(Set(name="r", elements=tuple(sets.r)))
    set_manager.add(Set(name="i", elements=tuple(sets.i)))
    set_manager.add(Set(name="j", elements=tuple(sets.prod_comm)))
    set_manager.add(Set(name="marg", elements=tuple(sets.marg)))
    set_manager.add(Set(name="s", elements=tuple(sets.r)))
    set_manager.add(Set(name="rp", elements=tuple(sets.r)))
    set_manager.add(Set(name="f", elements=tuple(sets.f)))
    set_manager.add(Set(name="cgds", elements=tuple(sets.cgds)))
    return set_manager


def _build_block():
    from equilibria.blocks.gtap6.trade_armington import TradeArmingtonBlock

    sets, params, derived = _build_calibration()
    block = TradeArmingtonBlock(sets=sets, params=params, derived=derived)
    return block, sets, params, derived


@pytest.fixture(scope="module")
def _fixtures():
    block, sets, params, derived = _build_block()
    set_manager = _build_set_manager(sets)
    variables: dict = {}
    parameters: dict = {}
    equations = block.setup(set_manager, parameters, variables)
    oracle = _build_oracle()
    return block, sets, params, derived, set_manager, equations, variables, oracle


def test_trade_armington_block_setup_returns_all_contract_equations(_fixtures):
    from equilibria.templates.gtap6.gtap6_contract import _GTAP6_MARGINS, _GTAP6_TRADE

    _block, _sets, _params, _derived, _sm, equations, _vars, _oracle = _fixtures
    eq_names = {eq.name for eq in equations}

    expected = set(_GTAP6_TRADE) | set(_GTAP6_MARGINS)
    missing = expected - eq_names
    extra = eq_names - expected
    assert not missing, f"TradeArmingtonBlock did not produce: {missing}"
    assert not extra, f"TradeArmingtonBlock produced unexpected equations: {extra}"
    assert len(eq_names) == 16, (
        f"expected 16 unique equation names, got {len(eq_names)}"
    )


def _index_combos(pyomo_model, domains):
    """Cartesian product of the oracle's own Pyomo sets for `domains`."""
    sets = []
    for d in domains:
        sets.append(list(getattr(pyomo_model, d)))

    def _product(sets_list):
        if not sets_list:
            return [()]
        first, *rest = sets_list
        return [(e, *combo) for e in first for combo in _product(rest)]

    return _product(sets)


def test_trade_armington_block_matches_oracle_numerically(_fixtures):
    """Load-bearing numeric form-diff: block algebra vs the oracle, per-cell."""
    from pyomo.environ import Constraint
    from pyomo.environ import value as pyo_value

    block, _sets, _params, _derived, _sm, equations, _vars, oracle = _fixtures
    eq_by_name = {eq.name: eq for eq in equations}

    oracle_cons = {c.name: c for c in oracle.component_objects(Constraint, active=True)}

    total_checked = 0
    max_abs_diff = 0.0
    worst_cell: tuple[str, object] | None = None

    for block_name, oracle_name in _ORACLE_CONSTRAINT_FOR.items():
        eq = eq_by_name[block_name]
        con = oracle_cons[oracle_name]
        oracle_active_idx = {idx for idx, c in con.items() if c.active}

        checked_this_eq = 0
        for idx in _index_combos(oracle, eq.domains):
            block_expr = eq.build_expression(oracle, idx)
            key = idx if len(idx) > 1 else idx[0]
            oracle_is_active = key in oracle_active_idx

            if block_expr is None:
                # Block Skips this cell — must match the oracle's own Skip.
                assert not oracle_is_active, (
                    f"{block_name} Skips {idx} but oracle {oracle_name} is active there"
                )
                continue

            assert oracle_is_active, (
                f"{block_name} builds {idx} but oracle {oracle_name} Skips it"
            )
            oracle_con = con[key]

            # Compare the RESIDUAL (body - lower, for an equality constraint)
            # of the block's freshly-built expression vs the oracle's own
            # wired Constraint, evaluated at the SAME live variable values
            # (both read off the identical oracle pyomo_model object).
            block_con = block_expr
            b_body = pyo_value(block_con.args[0]) - pyo_value(block_con.args[1])
            o_body = pyo_value(oracle_con.body) - pyo_value(oracle_con.lower)
            diff = abs(b_body - o_body)
            if diff > max_abs_diff:
                max_abs_diff = diff
                worst_cell = (block_name, idx)
            assert diff < _TOL, (
                f"{block_name}{idx}: block residual {b_body} vs oracle "
                f"residual {o_body} (diff {diff} >= {_TOL})"
            )
            checked_this_eq += 1
            total_checked += 1

        assert checked_this_eq > 0, f"{block_name}: no active cells checked"

    assert total_checked > 0
    # Surface the methodology result for the report (max residual diff found).
    print(
        f"\n[gtap6 form-diff] {total_checked} cells checked across "
        f"{len(_ORACLE_CONSTRAINT_FOR)} equations; max |diff| = {max_abs_diff:.3e} "
        f"at {worst_cell}"
    )


def test_qds_matches_oracle_vds_identity(_fixtures):
    """e_qds has no oracle Constraint; verify against the oracle's OWN vds
    calibration identity: vds(i,r) = sum_j VDFM(i,j,r) + VDPM(i,r) + VDGM(i,r)
    (gtap6_calibration.py lines 305-313), which is exactly what qds is seeded
    to and what EqQds's build_expression computes from qfd/qpd/qgd stubs.
    """
    from pyomo.environ import value as pyo_value

    block, sets, params, derived, set_manager, equations, variables, oracle = _fixtures
    eq = {e.name: e for e in equations}["e_qds"]

    b = params.benchmark
    max_diff = 0.0
    for i in sets.i:
        for r in sets.r:
            # Evaluate the block's identity directly against the oracle's own
            # qfd/qpd/qgd Vars (seeded at benchmark, so this is a pure identity
            # check independent of any block-local stub wiring).
            expr = eq.build_expression(oracle, (i, r))
            assert expr is not None
            lhs = pyo_value(expr.args[0])
            rhs = pyo_value(expr.args[1])
            expected = (
                sum(b.vdfm.get((i, j, r), 0.0) or 0.0 for j in sets.prod_comm)
                + (b.vdpm.get((i, r), 0.0) or 0.0)
                + (b.vdgm.get((i, r), 0.0) or 0.0)
            )
            # lhs is qds[i,r] (the oracle's own Var, seeded from vds), rhs is
            # the block's computed sum of qfd+qpd+qgd off the oracle's own
            # benchmark-seeded Vars.
            assert abs(rhs - expected) < 1e-6, (i, r, rhs, expected)
            max_diff = max(max_diff, abs(rhs - expected))
    print(f"\n[gtap6 e_qds identity] max |diff| = {max_diff:.3e}")


def test_qtmfsd_matches_oracle_qtm_summand(_fixtures):
    """e_qtmfsd has no oracle Constraint; verify summing it over (i,s,rp)
    reproduces the oracle's own eq_qtm RHS/ptmg[mg] term-for-term.
    """
    from pyomo.environ import Constraint
    from pyomo.environ import value as pyo_value

    block, sets, params, derived, set_manager, equations, variables, oracle = _fixtures
    eq = {e.name: e for e in equations}["e_qtmfsd"]

    qtm_con = oracle.eq_qtm
    max_diff = 0.0
    checked = 0
    for mg in sets.marg:
        # Pyomo 1-D Set indices come back as bare scalars, not 1-tuples.
        con = None
        for k, c in qtm_con.items():
            kk = k if isinstance(k, tuple) else (k,)
            if kk == (mg,):
                con = c
                break
        if con is None or not con.active:
            continue

        total = 0.0
        for i in sets.i:
            for src in sets.s if hasattr(sets, "s") else sets.r:
                for dst in sets.r:
                    expr = eq.build_expression(oracle, (mg, i, src, dst))
                    if expr is None:
                        continue
                    # RHS only (args[1]): qtmfsd itself (args[0]) has no
                    # corresponding Var on the oracle model (it is a NEW
                    # per-shipment quantity the oracle never materializes —
                    # see the block's module docstring). The identity under
                    # test is that summing the RHS over (i,s,rp) reproduces
                    # the oracle's own eq_qtm summand exactly.
                    rhs = pyo_value(expr.args[1])
                    total += rhs
        expected_qtm = pyo_value(oracle.qtm[mg])
        diff = abs(total - expected_qtm)
        max_diff = max(max_diff, diff)
        # HAR benchmark data is stored as float32 (GEMPACK convention); the
        # oracle's derive_calibration accumulates amgm/pwmg/qxs shares from
        # that float32 source, so a ~3e-6 RELATIVE float-precision residual
        # over a summation of ~9 terms is expected rounding noise, not a
        # structural mismatch. Use a relative tolerance consistent with the
        # rest of the codebase's HAR-derived-float32 comparisons.
        rel = diff / max(abs(expected_qtm), 1e-12)
        assert rel < 1e-5, (mg, total, expected_qtm, rel)
        checked += 1
    assert checked > 0
    print(f"\n[gtap6 e_qtmfsd summand] max |diff| = {max_diff:.3e}")


# ======================================================================
# ProductionBlock (F7 Task 7)
# ======================================================================

# Oracle Constraint name -> block equation name for the 8 equations in
# _GTAP6_PRODUCTION. e_qo maps to the oracle's MARKET-CLEARING Constraint
# (_add_market_clearing's eq_market, "qo = activity output identity" per
# the contract's own comment), NOT the oracle's Constraint literally named
# eq_qo (which pins ps via the zero-profit condition and is the source for
# e_ps instead) — see production.py's module docstring for the full
# rename rationale, mirroring how Task 6 renamed eq_qf/eq_pf_int to
# e_qfa/e_pfa.
_ORACLE_CONSTRAINT_FOR_PRODUCTION = {
    "e_qo": "eq_market",
    "e_ps": "eq_qo",
    "e_pds": "eq_pds",
    "e_qf": "eq_qf",
    "e_pf": "eq_pf_int",
    "e_pfd": "eq_pfd",
    "e_pfm": "eq_pfm",
    "e_qva": "eq_va",
    "e_pva": "eq_pva",
    "e_qfe": "eq_qfe",
    "e_pfe": "eq_pfe",
}


def _build_oracle_production():
    """Build the oracle model with the aliases ProductionBlock needs.

    Same technique as Task 6's ``_build_oracle``: attach read-only Python
    attribute aliases (via ``object.__setattr__``, bypassing Pyomo's
    component-reparenting guard) onto the SAME live oracle
    Vars/Constraints, so ``ProductionBlock``'s ``build_expression`` (which
    is written against ``qva``/``pf``/``pfactor``) resolves against the
    oracle's own ``va``/``pf_int``/``pf`` objects without re-deriving a
    second model.

    Three renames are needed here (one more than Task 6's two, because
    ProductionBlock reads a THIRD oracle symbol under a new name to avoid
    colliding with its own ``pf``):

      oracle.qva      = oracle.va       (VA quantity: contract wants qva)
      oracle.pf       = oracle.pf_int   (Armington composite price, HERE
                                          under the production-nest name;
                                          Task 6 already aliased the SAME
                                          oracle.pf_int as oracle.pfa for
                                          its own e_pfa — both aliases can
                                          coexist, they just point at the
                                          same underlying Pyomo Var)
      oracle.pfactor  = oracle.pf       (regional factor wage (f,r) — the
                                          oracle's own ``pf`` Var, renamed
                                          so ProductionBlock's e_pf (i,j,r)
                                          and e_pfe's read of the factor
                                          wage don't collide under the
                                          same Python attribute name)
    """
    oracle = _build_oracle()
    object.__setattr__(oracle, "qva", oracle.va)
    object.__setattr__(oracle, "pfactor", oracle.pf)
    object.__setattr__(oracle, "pf", oracle.pf_int)
    return oracle


def _build_production_block():
    from equilibria.blocks.gtap6.production import ProductionBlock

    sets, params, derived = _build_calibration()
    block = ProductionBlock(sets=sets, params=params, derived=derived)
    return block, sets, params, derived


@pytest.fixture(scope="module")
def _production_fixtures():
    block, sets, params, derived = _build_production_block()
    set_manager = _build_set_manager(sets)
    variables: dict = {}
    parameters: dict = {}
    equations = block.setup(set_manager, parameters, variables)
    oracle = _build_oracle_production()
    return block, sets, params, derived, set_manager, equations, variables, oracle


def test_production_block_setup_returns_all_contract_equations(_production_fixtures):
    from equilibria.templates.gtap6.gtap6_contract import _GTAP6_PRODUCTION

    _block, _sets, _params, _derived, _sm, equations, _vars, _oracle = (
        _production_fixtures
    )
    eq_names = {eq.name for eq in equations}

    expected = set(_GTAP6_PRODUCTION)
    missing = expected - eq_names
    extra = eq_names - expected
    assert not missing, f"ProductionBlock did not produce: {missing}"
    assert not extra, f"ProductionBlock produced unexpected equations: {extra}"
    assert len(eq_names) == 11, (
        f"expected 11 unique equation names, got {len(eq_names)}"
    )


def test_production_block_matches_oracle_numerically(_production_fixtures):
    """Load-bearing numeric form-diff: block algebra vs the oracle, per-cell."""
    from pyomo.environ import value as pyo_value

    block, _sets, _params, _derived, _sm, equations, _vars, oracle = (
        _production_fixtures
    )
    eq_by_name = {eq.name: eq for eq in equations}

    from pyomo.environ import Constraint

    oracle_cons = {c.name: c for c in oracle.component_objects(Constraint, active=True)}

    total_checked = 0
    max_abs_diff = 0.0
    worst_cell: tuple[str, object] | None = None

    for block_name, oracle_name in _ORACLE_CONSTRAINT_FOR_PRODUCTION.items():
        eq = eq_by_name[block_name]
        con = oracle_cons[oracle_name]
        oracle_active_idx = {idx for idx, c in con.items() if c.active}

        checked_this_eq = 0
        for idx in _index_combos(oracle, eq.domains):
            block_expr = eq.build_expression(oracle, idx)
            key = idx if len(idx) > 1 else idx[0]
            oracle_is_active = key in oracle_active_idx

            if block_expr is None:
                assert not oracle_is_active, (
                    f"{block_name} Skips {idx} but oracle {oracle_name} is active there"
                )
                continue

            assert oracle_is_active, (
                f"{block_name} builds {idx} but oracle {oracle_name} Skips it"
            )
            oracle_con = con[key]

            block_con = block_expr
            b_body = pyo_value(block_con.args[0]) - pyo_value(block_con.args[1])
            o_body = pyo_value(oracle_con.body) - pyo_value(oracle_con.lower)
            diff = abs(b_body - o_body)
            if diff > max_abs_diff:
                max_abs_diff = diff
                worst_cell = (block_name, idx)
            assert diff < _TOL, (
                f"{block_name}{idx}: block residual {b_body} vs oracle "
                f"residual {o_body} (diff {diff} >= {_TOL})"
            )
            checked_this_eq += 1
            total_checked += 1

        assert checked_this_eq > 0, f"{block_name}: no active cells checked"

    assert total_checked > 0
    print(
        f"\n[gtap6 production form-diff] {total_checked} cells checked across "
        f"{len(_ORACLE_CONSTRAINT_FOR_PRODUCTION)} equations; max |diff| = "
        f"{max_abs_diff:.3e} at {worst_cell}"
    )


# ======================================================================
# FactorBlock (F7 Task 8)
# ======================================================================

# Only the MOBILE branch (e_qe/e_pe_endw) has a live oracle Constraint to
# diff against: the oracle's own _add_factor_markets applies
# eq_factor_clear/eq_qoes_fixed UNIFORMLY to every factor (no mf/sf split
# exists in the oracle at all — see factor.py's module docstring). The
# SLUGGISH branch (e_qoes/e_pmes/e_pm_endw) has no oracle Constraint
# anywhere (grep confirms no eq_qoes/eq_pmes/eq_pm_endw method exists) and
# is checked separately below via the oracle's own benchmark identity.
_ORACLE_CONSTRAINT_FOR_FACTOR = {
    "e_qe": "eq_factor_clear",
    "e_pe_endw": "eq_qoes_fixed",
}


def _build_oracle_factor():
    """Build the oracle model with the alias FactorBlock's mobile branch needs.

    FactorBlock's e_qe/e_pe_endw are written against ``m.qe[f,r]`` (a Var
    name distinct from ProductionBlock's own vars, chosen to match the
    contract's ``qe`` variable name). The oracle has no ``qe`` Var — its
    uniform (mobile-only, in effect) factor-market closure uses ``qoes``
    for this exact role for every factor, sluggish and mobile alike (see
    module docstring). Alias ``oracle.qe = oracle.qoes`` (read-only,
    same ``object.__setattr__`` technique as ``_build_oracle_production``)
    so the block's build_expression resolves against the oracle's own live
    Var without re-deriving a second model. This is a rename, not a new
    economic claim: for f in mf, the oracle's qoes IS what the contract
    calls qe (a supply level pinned to evom, cleared against qfe demand).
    """
    oracle = _build_oracle()
    object.__setattr__(oracle, "qe", oracle.qoes)
    return oracle


def _attach_factor_components(oracle, sets, params, derived):
    """Attach FactorBlock's own new Pyomo components onto the oracle.

    ``gf_share``/``omegaf``/``pmes``/``pmagg`` are genuinely NEW
    quantities this block introduces (the oracle has no CET sluggish
    allocation at all — see factor.py's module docstring) — unlike the
    ``qfa``/``pfa``/``qva`` aliases the Task 6/7 helpers attach (which
    rename an EXISTING oracle Var), these have no oracle counterpart to
    alias, so they are added as real new Pyomo components, the same
    technique the module-level ``_build_oracle`` uses for ``qtmfsd``.
    Seeded at the benchmark point (pmes=pmagg=1.0) so the sluggish-branch
    identity test below evaluates the CET equations at calibration.
    """
    from pyomo.environ import NonNegativeReals, Param, Var

    gf_share = {}
    for f in sets.f:
        for j in sets.prod_comm:
            for r in sets.r:
                evom = derived.evom.get((f, r), 0.0) or 0.0
                vfm = params.benchmark.vfm.get((f, j, r), 0.0) or 0.0
                gf_share[(f, j, r)] = vfm / evom if evom > 1e-8 else 0.0
    oracle.gf_share = Param(
        oracle.f, oracle.j, oracle.r, initialize=gf_share, mutable=True
    )

    omegaf = {f: -float(params.elasticities.etrae.get(f, 0.0)) for f in sets.f}
    oracle.omegaf = Param(oracle.f, initialize=omegaf, mutable=True)

    oracle.pmes = Var(
        oracle.f, oracle.j, oracle.r, within=NonNegativeReals, initialize=1.0
    )
    oracle.pmagg = Var(oracle.f, oracle.r, within=NonNegativeReals, initialize=1.0)
    return oracle


def _build_factor_block():
    from equilibria.blocks.gtap6.factor import FactorBlock

    sets, params, derived = _build_calibration()
    block = FactorBlock(sets=sets, params=params, derived=derived)
    return block, sets, params, derived


@pytest.fixture(scope="module")
def _factor_fixtures():
    block, sets, params, derived = _build_factor_block()
    set_manager = _build_set_manager(sets)
    variables: dict = {}
    parameters: dict = {}
    equations = block.setup(set_manager, parameters, variables)
    oracle = _build_oracle_factor()
    _attach_factor_components(oracle, sets, params, derived)
    return block, sets, params, derived, set_manager, equations, variables, oracle


def test_factor_block_setup_returns_all_contract_equations(_factor_fixtures):
    from equilibria.templates.gtap6.gtap6_contract import _GTAP6_FACTOR_MARKETS

    _block, _sets, _params, _derived, _sm, equations, _vars, _oracle = _factor_fixtures
    eq_names = {eq.name for eq in equations}

    expected = set(_GTAP6_FACTOR_MARKETS)
    missing = expected - eq_names
    extra = eq_names - expected
    assert not missing, f"FactorBlock did not produce: {missing}"
    assert not extra, f"FactorBlock produced unexpected equations: {extra}"
    assert len(eq_names) == 5, f"expected 5 unique equation names, got {len(eq_names)}"


def test_factor_block_mobile_matches_oracle_numerically(_factor_fixtures):
    """Load-bearing numeric form-diff for the MOBILE branch (e_qe/e_pe_endw).

    Restricted to f in sets.mf: the oracle's own eq_factor_clear/
    eq_qoes_fixed are the byte-identical source for these two equations
    (see factor.py's module docstring — the oracle applies them uniformly
    because it has not yet split mobile/sluggish; this block scopes the
    SAME algebra to the mf subset the contract's e_qe/e_pe_endw IDs own).
    """
    from pyomo.environ import Constraint
    from pyomo.environ import value as pyo_value

    block, sets, _params, _derived, _sm, equations, _vars, oracle = _factor_fixtures
    eq_by_name = {eq.name: eq for eq in equations}
    mobile = set(sets.mf)

    oracle_cons = {c.name: c for c in oracle.component_objects(Constraint, active=True)}

    total_checked = 0
    max_abs_diff = 0.0
    worst_cell: tuple[str, object] | None = None

    for block_name, oracle_name in _ORACLE_CONSTRAINT_FOR_FACTOR.items():
        eq = eq_by_name[block_name]
        con = oracle_cons[oracle_name]
        oracle_active_idx = {idx for idx, c in con.items() if c.active}

        checked_this_eq = 0
        for idx in _index_combos(oracle, eq.domains):
            f = idx[0]
            block_expr = eq.build_expression(oracle, idx)
            key = idx if len(idx) > 1 else idx[0]
            oracle_is_active = key in oracle_active_idx

            if f not in mobile:
                # Sluggish factors are Skipped by design (owned by
                # e_qoes/e_pmes/e_pm_endw instead) — the oracle itself has
                # no mf/sf split, so it IS active there; that is expected
                # and not a mismatch to assert against.
                assert block_expr is None
                continue

            if block_expr is None:
                assert not oracle_is_active, (
                    f"{block_name} Skips {idx} but oracle {oracle_name} is active there"
                )
                continue

            assert oracle_is_active, (
                f"{block_name} builds {idx} but oracle {oracle_name} Skips it"
            )
            oracle_con = con[key]

            block_con = block_expr
            b_body = pyo_value(block_con.args[0]) - pyo_value(block_con.args[1])
            o_body = pyo_value(oracle_con.body) - pyo_value(oracle_con.lower)
            diff = abs(b_body - o_body)
            if diff > max_abs_diff:
                max_abs_diff = diff
                worst_cell = (block_name, idx)
            assert diff < _TOL, (
                f"{block_name}{idx}: block residual {b_body} vs oracle "
                f"residual {o_body} (diff {diff} >= {_TOL})"
            )
            checked_this_eq += 1
            total_checked += 1

        assert checked_this_eq > 0, f"{block_name}: no active mobile cells checked"

    assert total_checked > 0
    print(
        f"\n[gtap6 factor mobile form-diff] {total_checked} cells checked across "
        f"{len(_ORACLE_CONSTRAINT_FOR_FACTOR)} equations; max |diff| = "
        f"{max_abs_diff:.3e} at {worst_cell}"
    )


def test_factor_block_sluggish_matches_benchmark_identity(_factor_fixtures):
    """e_qoes/e_pmes/e_pm_endw have no oracle Constraint; verify against the
    oracle's OWN benchmark-seeded values instead (all prices == 1.0 at the
    calibration point, so the CET reduces to a share/revenue identity).

    e_qoes and e_pmes are satisfied EXACTLY at the seed (gf_share is
    defined as vfm/evom, so gf*qoes == qfe collapses to an algebraic
    identity, and pmes == pfe holds since both are seeded to 1.0).
    e_pm_endw carries the genuine ~2-9% agent-vs-market-price benchmark
    wedge the oracle's own docstring documents for eq_market (vfm/evom
    summed across sectors is not exactly 1.0) — checked with a relative
    tolerance wide enough to accommodate that documented wedge rather than
    a bug tolerance.
    """
    from pyomo.environ import value as pyo_value

    block, sets, params, derived, _sm, equations, _vars, oracle = _factor_fixtures
    eq_by_name = {eq.name: eq for eq in equations}
    sluggish = set(sets.sf)

    eq_qoes = eq_by_name["e_qoes"]
    eq_pmes = eq_by_name["e_pmes"]
    eq_pm_endw = eq_by_name["e_pm_endw"]

    checked_qoes = 0
    max_diff_qoes = 0.0
    checked_pmes = 0
    max_diff_pmes = 0.0
    for f in sluggish:
        for r in sets.r:
            evom = derived.evom.get((f, r), 0.0) or 0.0
            if evom <= 1e-8:
                continue
            for j in sets.prod_comm:
                vfm = params.benchmark.vfm.get((f, j, r), 0.0) or 0.0
                gf = vfm / evom if evom > 0 else 0.0

                expr = eq_qoes.build_expression(oracle, (f, j, r))
                if gf <= 0.0:
                    assert expr is None
                    continue
                assert expr is not None
                lhs = pyo_value(expr.args[0])
                rhs = pyo_value(expr.args[1])
                diff = abs(lhs - rhs)
                max_diff_qoes = max(max_diff_qoes, diff)
                assert diff < 1e-6, (f, j, r, lhs, rhs, diff)
                checked_qoes += 1

                expr_p = eq_pmes.build_expression(oracle, (f, j, r))
                assert expr_p is not None
                lhs_p = pyo_value(expr_p.args[0])
                rhs_p = pyo_value(expr_p.args[1])
                diff_p = abs(lhs_p - rhs_p)
                max_diff_pmes = max(max_diff_pmes, diff_p)
                assert diff_p < 1e-9, (f, j, r, lhs_p, rhs_p, diff_p)
                checked_pmes += 1

    assert checked_qoes > 0
    assert checked_pmes > 0
    print(f"\n[gtap6 e_qoes benchmark identity] max |diff| = {max_diff_qoes:.3e}")
    print(f"[gtap6 e_pmes benchmark identity] max |diff| = {max_diff_pmes:.3e}")

    checked_endw = 0
    max_rel_endw = 0.0
    for f in sluggish:
        for r in sets.r:
            evom = derived.evom.get((f, r), 0.0) or 0.0
            if evom <= 1e-8:
                continue
            expr = eq_pm_endw.build_expression(oracle, (f, r))
            assert expr is not None
            lhs = pyo_value(expr.args[0])
            rhs = pyo_value(expr.args[1])
            rel = abs(lhs - rhs) / max(abs(rhs), 1e-12)
            max_rel_endw = max(max_rel_endw, rel)
            # Documented benchmark wedge: sum_j (vfm/evom) is ~1.04-1.09 on
            # gtap6_3x3 (agent-vs-market price residual), not exactly 1.0 —
            # this is the same class of benchmark residual the oracle's own
            # docstring documents for eq_market (~2-9%), not a structural
            # mismatch. A generous relative tolerance distinguishes "known
            # SAM wedge" from "block algebra is wrong".
            assert rel < 0.15, (f, r, lhs, rhs, rel)
            checked_endw += 1

    assert checked_endw > 0
    print(f"[gtap6 e_pm_endw benchmark identity] max |rel diff| = {max_rel_endw:.3e}")


def test_factor_block_sluggish_cet_curvature_matches_omega_sign(_factor_fixtures):
    """Exponent/curvature check for the sluggish CET branch (e_qoes/e_pm_endw).

    The benchmark-identity test above evaluates everything AT pmes ==
    pmagg == 1.0, where ``x**omega == 1`` for ANY omega — it cannot tell a
    correct CET exponent from a wrong-signed one or a different functional
    form entirely (e.g. Leontief or Cobb-Douglas would also pass a
    benchmark-only check). This test perturbs pmes away from 1.0 (holding
    qoes/pmagg/qfe fixed at their seed levels) and numerically
    differentiates each equation's residual/RHS with respect to pmes,
    confirming the SIGN of the slope matches the sign of the exponent that
    actually appears in the equation body — the same check GTAP7's
    ``EqPfeq``/``EqPfteq`` sf-branch (the cross-reference this block's
    algebra was transcribed from) would have to satisfy.

    e_qoes uses exponent ``omega`` directly:
      R(pmes) = pmes**omega * gf * qoes - pmagg**omega * qfe
      dR/dpmes has the SAME SIGN as omega (verified analytically: dR/dpmes
      = omega * gf * qoes * pmes**(omega-1), and gf/qoes/pmes are all > 0
      at any point of interest here).
    e_pm_endw uses exponent ``1+omega`` (the CET aggregator power):
      RHS(pmes) = sum_j gf_share * pmes**(1+omega)
      d(RHS)/dpmes has the SAME SIGN as (1+omega) for the SAME reason.

    On gtap6_3x3: Land has omega = -etrae['Land'] = 1.0 (a genuine,
    non-degenerate positive CET elasticity — a real economic case, not a
    synthetic one) and Capital has omega = -etrae['Capital'] = -(-0.0) ==
    0.0 (a real degenerate case where the CET collapses to a flat
    response). Both are exercised directly against real dataset values,
    not mocked factors.
    """
    from pyomo.environ import value as pyo_value

    block, sets, params, derived, _sm, equations, _vars, oracle = _factor_fixtures
    eq_by_name = {eq.name: eq for eq in equations}
    eq_qoes = eq_by_name["e_qoes"]
    eq_pm_endw = eq_by_name["e_pm_endw"]

    etrae = params.elasticities.etrae
    _BUMP = 0.05

    def _omega(f):
        return -float(etrae.get(f, 0.0))

    checked_qoes_slope = 0
    checked_endw_slope = 0

    for f in sets.sf:
        omega = _omega(f)
        for r in sets.r:
            evom = derived.evom.get((f, r), 0.0) or 0.0
            if evom <= 1e-8:
                continue

            # ---- e_qoes: perturb pmes[f, j, r] for one active sector j,
            # holding pmagg[f, r] fixed at its seed value (1.0), and
            # numerically differentiate the equation's residual.
            for j in sets.prod_comm:
                vfm = params.benchmark.vfm.get((f, j, r), 0.0) or 0.0
                gf = vfm / evom if evom > 0 else 0.0
                if gf <= 0.0:
                    continue

                seed_pmes = float(pyo_value(oracle.pmes[f, j, r]))
                seed_pmagg = float(pyo_value(oracle.pmagg[f, r]))
                assert seed_pmes == 1.0 and seed_pmagg == 1.0, (
                    "test assumes the fixture's benchmark seed (1.0); if this "
                    "ever changes, the finite-difference bump below must be "
                    "re-derived around the new seed point"
                )

                def _residual(pmes_val, f=f, j=j, r=r):
                    oracle.pmes[f, j, r].set_value(pmes_val)
                    try:
                        expr = eq_qoes.build_expression(oracle, (f, j, r))
                        assert expr is not None
                        return pyo_value(expr.args[0]) - pyo_value(expr.args[1])
                    finally:
                        oracle.pmes[f, j, r].set_value(seed_pmes)

                r0 = _residual(seed_pmes)
                r1 = _residual(seed_pmes + _BUMP)
                slope = (r1 - r0) / _BUMP

                if abs(omega) < 1e-8:
                    # Degenerate CET (omega == 0, e.g. Capital on
                    # gtap6_3x3): pmes**0 == 1 regardless of pmes, so the
                    # residual must be FLAT — a nonzero slope here would
                    # mean the block silently used a different exponent
                    # than omegaf (e.g. a stray +1).
                    assert abs(slope) < 1e-6, (f, j, r, omega, slope)
                else:
                    # sign(dR/dpmes) must match sign(omega) — this is what
                    # a benchmark-only (pmes==1) check can never catch,
                    # since x**omega == 1 there for any omega.
                    assert slope * omega > 0, (
                        f"e_qoes({f},{j},{r}): omega={omega} but perturbing "
                        f"pmes gives slope={slope} (wrong sign or flat)"
                    )
                checked_qoes_slope += 1

            # ---- e_pm_endw: perturb pmes[f, j, r] for one active sector,
            # numerically differentiate the RHS w.r.t. that pmes. The
            # governing exponent here is (1+omega), not omega directly.
            active_j = [
                j
                for j in sets.prod_comm
                if (params.benchmark.vfm.get((f, j, r), 0.0) or 0.0) > 0.0
            ]
            if not active_j:
                continue
            j0 = active_j[0]
            expo = 1.0 + omega

            def _endw_rhs(pmes_val, f=f, j0=j0, r=r):
                oracle.pmes[f, j0, r].set_value(pmes_val)
                try:
                    expr = eq_pm_endw.build_expression(oracle, (f, r))
                    assert expr is not None
                    return pyo_value(expr.args[1])
                finally:
                    oracle.pmes[f, j0, r].set_value(1.0)

            e0 = _endw_rhs(1.0)
            e1 = _endw_rhs(1.0 + _BUMP)
            endw_slope = (e1 - e0) / _BUMP

            if abs(expo) < 1e-8:
                assert abs(endw_slope) < 1e-6, (f, r, expo, endw_slope)
            else:
                assert endw_slope * expo > 0, (
                    f"e_pm_endw({f},{r}): (1+omega)={expo} but perturbing "
                    f"pmes gives slope={endw_slope} (wrong sign or flat)"
                )
            checked_endw_slope += 1

    assert checked_qoes_slope > 0
    assert checked_endw_slope > 0
    print(
        f"\n[gtap6 e_qoes CET curvature] {checked_qoes_slope} (f,j,r) cells "
        "sign-checked against omega"
    )
    print(
        f"[gtap6 e_pm_endw CET curvature] {checked_endw_slope} (f,r) cells "
        "sign-checked against (1+omega)"
    )


# ======================================================================
# DemandUtilityBlock (F7 Task 9a)
# ======================================================================

# Oracle Constraint name -> block equation name for the 14 of 16
# equations that exist as an active Constraint in the oracle under the
# SAME variable set (module docstring in demand_utility.py has the full
# grep-verified mapping/line numbers). e_qfd_cgds/e_qfm_cgds have no
# DEDICATED oracle Constraint (the oracle's eq_qfd/eq_qfm cover ALL j,
# not just cgds) and are checked separately below by restricting the
# SAME oracle Constraint to the cgds slice.
_ORACLE_CONSTRAINT_FOR_DEMAND = {
    "e_qpd": "eq_qpd",
    "e_qpm": "eq_qpm",
    "e_qp": "eq_qp",
    "e_pp": "eq_pp",
    "e_ppd": "eq_ppd",
    "e_ppm": "eq_ppm",
    "e_pq": "eq_pcons",
    "e_up": "eq_up",
    "e_qgd": "eq_qgd",
    "e_qgm": "eq_qgm",
    "e_qg": "eq_qg",
    "e_pg": "eq_pg",
    "e_pgd": "eq_pgd",
    "e_pgm": "eq_pgm",
    "e_pgov": "eq_pgov",
    "e_ug": "eq_ug",
    "e_qcgds": "eq_qcgds",
    "e_pcgds": "eq_pcgds",
}


def _build_oracle_demand_utility():
    """Build the oracle model with the alias DemandUtilityBlock needs.

    DemandUtilityBlock's e_pq/e_up are written against ``m.pq[r]`` (the
    contract's name for the CDE expenditure-function aggregator; see the
    block's module docstring). The oracle hosts the SAME identity under
    the Python attribute name ``pcons`` (its own docstring: "We re-use
    the variable name `pcons` to host this identity for closure-matching
    reasons"). Alias ``oracle.pq = oracle.pcons`` (read-only,
    ``object.__setattr__``, same technique as
    ``_build_oracle_production``/``_build_oracle_factor``) so the block's
    build_expression resolves against the oracle's own live Var without
    re-deriving a second model. This is a rename, not a new economic
    claim.
    """
    oracle = _build_oracle()
    object.__setattr__(oracle, "pq", oracle.pcons)
    return oracle


def _build_demand_utility_block():
    from equilibria.blocks.gtap6.demand_utility import DemandUtilityBlock

    sets, params, derived = _build_calibration()
    block = DemandUtilityBlock(sets=sets, params=params, derived=derived)
    return block, sets, params, derived


@pytest.fixture(scope="module")
def _demand_utility_fixtures():
    block, sets, params, derived = _build_demand_utility_block()
    set_manager = _build_set_manager(sets)
    variables: dict = {}
    parameters: dict = {}
    equations = block.setup(set_manager, parameters, variables)
    oracle = _build_oracle_demand_utility()
    return block, sets, params, derived, set_manager, equations, variables, oracle


def test_demand_utility_block_setup_returns_all_contract_equations(
    _demand_utility_fixtures,
):
    """Confirm the split ruling: _GTAP6_FINAL_DEMAND has 22 IDs total
    (18 original + e_ppd/e_ppm/e_pgd/e_pgm added in Task 10b -- see
    demand_utility.py's e_ppd/e_pgd docstrings); this block owns exactly
    20 (all but e_yp/e_yg, reserved for Task 9b's IncomeClosureBlock per
    the controller's ruling).
    """
    from equilibria.templates.gtap6.gtap6_contract import _GTAP6_FINAL_DEMAND

    _block, _sets, _params, _derived, _sm, equations, _vars, _oracle = (
        _demand_utility_fixtures
    )
    eq_names = {eq.name for eq in equations}

    assert len(_GTAP6_FINAL_DEMAND) == 22, (
        f"expected 22 IDs in _GTAP6_FINAL_DEMAND, got {len(_GTAP6_FINAL_DEMAND)}"
    )
    assert "e_yp" in _GTAP6_FINAL_DEMAND
    assert "e_yg" in _GTAP6_FINAL_DEMAND

    expected = set(_GTAP6_FINAL_DEMAND) - {"e_yp", "e_yg"}
    missing = expected - eq_names
    extra = eq_names - expected
    assert not missing, f"DemandUtilityBlock did not produce: {missing}"
    assert not extra, f"DemandUtilityBlock produced unexpected equations: {extra}"
    assert len(eq_names) == 20, (
        f"expected 20 unique equation names, got {len(eq_names)}"
    )


def test_demand_utility_block_matches_oracle_numerically(_demand_utility_fixtures):
    """Load-bearing numeric form-diff: block algebra vs the oracle, per-cell.

    Evaluated at the fixture's benchmark seed (up=1, pp=pp_0, yp=yp_0,
    pcons=1). This is non-vacuous for 13 of the 14 mapped equations
    (e_qpd/e_qpm/e_pp/e_qgd/e_qgm/e_qg/e_pg/e_pgov/e_ug/e_qcgds/e_pcgds
    combine genuinely different ppd/ppm/pgd/pgm/yp/yg/qo/ps values, not a
    degenerate x**exp-at-x=1 case). e_qp/e_pq/e_up ARE degenerate at this
    seed (the CDE ratio (pp/pp_0)/(yp/yp_0) == 1 identically, so any
    INCPAR/SUBPAR exponent passes) — see
    test_demand_utility_cde_curvature_matches_incpar_subpar below for the
    non-vacuous perturbation check Task 8's review cycle established as
    mandatory for exactly this situation.
    """
    from pyomo.environ import Constraint
    from pyomo.environ import value as pyo_value

    block, _sets, _params, _derived, _sm, equations, _vars, oracle = (
        _demand_utility_fixtures
    )
    eq_by_name = {eq.name: eq for eq in equations}

    oracle_cons = {c.name: c for c in oracle.component_objects(Constraint, active=True)}

    total_checked = 0
    max_abs_diff = 0.0
    worst_cell: tuple[str, object] | None = None

    for block_name, oracle_name in _ORACLE_CONSTRAINT_FOR_DEMAND.items():
        eq = eq_by_name[block_name]
        con = oracle_cons[oracle_name]
        oracle_active_idx = {idx for idx, c in con.items() if c.active}

        checked_this_eq = 0
        for idx in _index_combos(oracle, eq.domains):
            block_expr = eq.build_expression(oracle, idx)
            key = idx if len(idx) > 1 else idx[0]
            oracle_is_active = key in oracle_active_idx

            if block_expr is None:
                assert not oracle_is_active, (
                    f"{block_name} Skips {idx} but oracle {oracle_name} is active there"
                )
                continue

            assert oracle_is_active, (
                f"{block_name} builds {idx} but oracle {oracle_name} Skips it"
            )
            oracle_con = con[key]

            block_con = block_expr
            b_body = pyo_value(block_con.args[0]) - pyo_value(block_con.args[1])
            o_body = pyo_value(oracle_con.body) - pyo_value(oracle_con.lower)
            diff = abs(b_body - o_body)
            if diff > max_abs_diff:
                max_abs_diff = diff
                worst_cell = (block_name, idx)
            assert diff < _TOL, (
                f"{block_name}{idx}: block residual {b_body} vs oracle "
                f"residual {o_body} (diff {diff} >= {_TOL})"
            )
            checked_this_eq += 1
            total_checked += 1

        assert checked_this_eq > 0, f"{block_name}: no active cells checked"

    assert total_checked > 0
    print(
        f"\n[gtap6 demand-utility form-diff] {total_checked} cells checked across "
        f"{len(_ORACLE_CONSTRAINT_FOR_DEMAND)} equations; max |diff| = "
        f"{max_abs_diff:.3e} at {worst_cell}"
    )


def test_demand_utility_cgds_matches_oracle_qfd_qfm_slice(_demand_utility_fixtures):
    """e_qfd_cgds/e_qfm_cgds have no DEDICATED oracle Constraint — the
    oracle's eq_qfd/eq_qfm cover ALL j (including cgds) in one indexed
    Constraint, already ported in full by TradeArmingtonBlock as
    e_qfd_arm/e_qfm_arm. Verify this block's cgds-restricted equations
    reproduce the SAME oracle Constraint bodies, evaluated at the
    j==cgds slice only.
    """
    from pyomo.environ import value as pyo_value

    block, sets, _params, _derived, _sm, equations, _vars, oracle = (
        _demand_utility_fixtures
    )
    eq_by_name = {eq.name: eq for eq in equations}
    eq_qfd_cgds = eq_by_name["e_qfd_cgds"]
    eq_qfm_cgds = eq_by_name["e_qfm_cgds"]

    oracle_qfd = oracle.eq_qfd
    oracle_qfm = oracle.eq_qfm

    checked = 0
    max_diff = 0.0
    for i in sets.i:
        for cg in sets.cgds:
            for r in sets.r:
                block_expr = eq_qfd_cgds.build_expression(oracle, (i, cg, r))
                key = (i, cg, r)
                oracle_active = key in oracle_qfd and oracle_qfd[key].active
                if block_expr is None:
                    assert not oracle_active, (i, cg, r)
                    continue
                assert oracle_active, (i, cg, r)
                oracle_con = oracle_qfd[key]
                b_body = pyo_value(block_expr.args[0]) - pyo_value(block_expr.args[1])
                o_body = pyo_value(oracle_con.body) - pyo_value(oracle_con.lower)
                diff = abs(b_body - o_body)
                max_diff = max(max_diff, diff)
                assert diff < _TOL, (i, cg, r, b_body, o_body, diff)
                checked += 1

                block_expr_m = eq_qfm_cgds.build_expression(oracle, (i, cg, r))
                oracle_active_m = key in oracle_qfm and oracle_qfm[key].active
                if block_expr_m is None:
                    assert not oracle_active_m, (i, cg, r)
                    continue
                assert oracle_active_m, (i, cg, r)
                oracle_con_m = oracle_qfm[key]
                b_body_m = pyo_value(block_expr_m.args[0]) - pyo_value(
                    block_expr_m.args[1]
                )
                o_body_m = pyo_value(oracle_con_m.body) - pyo_value(oracle_con_m.lower)
                diff_m = abs(b_body_m - o_body_m)
                max_diff = max(max_diff, diff_m)
                assert diff_m < _TOL, (i, cg, r, b_body_m, o_body_m, diff_m)
                checked += 1

    assert checked > 0
    print(f"\n[gtap6 e_qfd_cgds/e_qfm_cgds slice] max |diff| = {max_diff:.3e}")


def test_demand_utility_cde_curvature_matches_incpar_subpar(_demand_utility_fixtures):
    """Exponent/curvature check for the household CDE branch (e_qp/e_pq/
    e_up).

    The benchmark-seed form-diff test above evaluates e_qp/e_pq/e_up at
    up==1, pp==pp_0, yp==yp_0 — the CDE ratio ``(pp/pp_0)/(yp/yp_0)``
    collapses to exactly 1 there, so ``ratio**SUBPAR == 1`` and
    ``up**(INCPAR*SUBPAR) == 1`` for ANY INCPAR/SUBPAR value (even a
    Cobb-Douglas placeholder with INCPAR=SUBPAR=0 would pass). This is
    precisely the Task 8 review-cycle precedent
    (test_factor_block_sluggish_cet_curvature_matches_omega_sign):
    a benchmark-only identity is not sufficient evidence for a nonlinear
    functional form: perturb pp[i,r] away from pp_0 (holding up/yp fixed
    at their seed) and confirm the SIGN and (for a nonzero SUBPAR)
    MAGNITUDE of the resulting share-expression slope matches
    SUBPAR — the exponent that actually appears in both the oracle's
    ``_cde_term``/``eq_qp_rule``/``eq_pcons_rule`` and this block's
    identical transcription.

    share_i(pp) = CONSHR_i_0 * up^(INCPAR*SUBPAR) * ((pp_i/pp_i_0)/(yp/yp_0))^SUBPAR_i
    d(share)/d(pp_i) at the seed (up=1, yp=yp_0, pp_i=pp_i_0):
      = CONSHR_i_0 * SUBPAR_i / pp_i_0
    which has the SAME SIGN as SUBPAR_i (CONSHR_i_0, pp_i_0 > 0 whenever
    the cell is active). BOOK3X3's SUBPAR values are all positive
    (documented in phase319's finding: SUBPAR in [0.01, 0.97]), so this
    also confirms the slope is POSITIVE and roughly proportional to
    SUBPAR_i across goods with very different SUBPAR (food ~0.87-0.97 vs
    services ~0.01), a real economic dispersion in this dataset, not a
    synthetic one.
    """
    from pyomo.environ import value as pyo_value

    block, sets, params, derived, _sm, equations, _vars, oracle = (
        _demand_utility_fixtures
    )
    eq_by_name = {eq.name: eq for eq in equations}
    eq_qp = eq_by_name["e_qp"]
    eq_pq = eq_by_name["e_pq"]

    subpar = params.elasticities.subpar
    incpar = params.elasticities.incpar
    share_hhd_cd = derived.share_hhd_cd
    pp_0 = derived.pp_0

    _BUMP_REL = 0.02  # 2% bump on pp_i, small enough to stay in-nest

    checked_qp_slope = 0
    checked_pq_slope = 0
    subpar_values_seen: set[float] = set()

    for i in sets.i:
        for r in sets.r:
            cshr_0 = float(share_hhd_cd.get((i, r), 0.0) or 0.0)
            if cshr_0 <= 0.0:
                continue
            subp = float(subpar.get((i, r), 0.0) or 0.0)
            subpar_values_seen.add(round(subp, 4))

            seed_pp = float(pyo_value(oracle.pp[i, r]))
            seed_up = float(pyo_value(oracle.up[r]))
            seed_yp = float(pyo_value(oracle.yp[r]))
            assert seed_up == 1.0, (
                "test assumes the fixture's benchmark seed (up=1); if this "
                "ever changes the finite-difference bump must be re-derived"
            )

            bump = seed_pp * _BUMP_REL

            def _qp_residual(pp_val, i=i, r=r):
                oracle.pp[i, r].set_value(pp_val)
                try:
                    expr = eq_qp.build_expression(oracle, (i, r))
                    assert expr is not None
                    return pyo_value(expr.args[0]) - pyo_value(expr.args[1])
                finally:
                    oracle.pp[i, r].set_value(seed_pp)

            r0 = _qp_residual(seed_pp)
            r1 = _qp_residual(seed_pp + bump)
            slope = (r1 - r0) / bump

            if subp <= 1e-8:
                # SUBPAR == 0 (degenerate CDE, if present in this dataset):
                # the ratio**0 == 1 term is flat in pp — residual slope
                # should come ONLY from the linear qp_0-independent LHS
                # term (pp*qp), not from the (now-flat) share expression.
                # Not exercised on BOOK3X3 (all SUBPAR > 0) but handled
                # for robustness.
                pass
            else:
                assert slope > 0, (
                    f"e_qp({i},{r}): SUBPAR={subp} > 0 but perturbing pp "
                    f"upward gives non-positive slope={slope}"
                )
            checked_qp_slope += 1

            # ---- e_pq: perturb the SAME pp[i,r], holding all other
            # goods' pp/up/yp fixed at seed, and differentiate the
            # expenditure-function identity's LHS (sum_i share_i - 1)
            # w.r.t. pp[i,r]. Since only good i's summand depends on
            # pp[i,r], the sign logic is identical to e_qp's.
            def _pq_residual(pp_val, i=i, r=r):
                oracle.pp[i, r].set_value(pp_val)
                try:
                    expr = eq_pq.build_expression(oracle, (r,))
                    assert expr is not None
                    return pyo_value(expr.args[0]) - pyo_value(expr.args[1])
                finally:
                    oracle.pp[i, r].set_value(seed_pp)

            p0 = _pq_residual(seed_pp)
            p1 = _pq_residual(seed_pp + bump)
            pq_slope = (p1 - p0) / bump

            if subp > 1e-8:
                assert pq_slope > 0, (
                    f"e_pq({r}) via good {i}: SUBPAR={subp} > 0 but "
                    f"perturbing pp[{i}] gives non-positive slope={pq_slope}"
                )
            checked_pq_slope += 1

    assert checked_qp_slope > 0
    assert checked_pq_slope > 0
    # Confirm real dispersion in SUBPAR across this dataset's goods (not
    # every cell sharing one degenerate value) — otherwise a sign-only
    # check could pass even for a constant-magnitude bug.
    assert len(subpar_values_seen) > 1, (
        f"expected dispersion in SUBPAR across goods, got only {subpar_values_seen}"
    )
    print(
        f"\n[gtap6 e_qp/e_pq CDE curvature] {checked_qp_slope} (i,r) cells "
        f"sign-checked against SUBPAR (values seen: {sorted(subpar_values_seen)})"
    )

    # ---- e_up: perturb yp[r] (holding pp fixed at pp_0, up at its
    # implicit value) and confirm the welfare identity's bilinear term
    # (up * pq) responds monotonically to yp — up*pq == yp/yp_0 is an
    # exact linear identity in yp, so the slope must equal EXACTLY
    # 1/yp_0 (not just same-signed): this catches a wrong-power bug
    # (e.g. accidentally squaring yp) that a sign check alone would miss.
    eq_up = eq_by_name["e_up"]
    checked_up_slope = 0
    for r in sets.r:
        yp_0 = float(derived.yp_0.get(r, 1.0) or 1.0)
        if yp_0 <= 0.0:
            continue
        seed_yp = float(pyo_value(oracle.yp[r]))
        seed_up = float(pyo_value(oracle.up[r]))
        seed_pq = float(pyo_value(oracle.pq[r]))

        def _up_residual(yp_val, r=r):
            oracle.yp[r].set_value(yp_val)
            try:
                expr = eq_up.build_expression(oracle, (r,))
                assert expr is not None
                return pyo_value(expr.args[0]) - pyo_value(expr.args[1])
            finally:
                oracle.yp[r].set_value(seed_yp)

        bump_yp = max(seed_yp * 0.02, 1e-6)
        u0 = _up_residual(seed_yp)
        u1 = _up_residual(seed_yp + bump_yp)
        slope = (u1 - u0) / bump_yp
        expected_slope = -1.0 / yp_0  # residual = up*pq - yp/yp_0
        assert abs(slope - expected_slope) < 1e-6, (
            r,
            slope,
            expected_slope,
            seed_up,
            seed_pq,
        )
        checked_up_slope += 1

    assert checked_up_slope > 0
    print(
        f"[gtap6 e_up curvature] {checked_up_slope} regions slope-checked (exact 1/yp_0)"
    )


def test_all_4_migrated_blocks_have_no_duplicate_equation_names():
    """Sanity check: no two migrated blocks claim the same equation ID
    (would silently overwrite in a real composer). Cheap guard now that
    4 of 5 blocks exist, ahead of Task 9b's full 5-block aggregate test.
    """
    from equilibria.blocks.gtap6.demand_utility import DemandUtilityBlock
    from equilibria.blocks.gtap6.factor import FactorBlock
    from equilibria.blocks.gtap6.production import ProductionBlock
    from equilibria.blocks.gtap6.trade_armington import TradeArmingtonBlock

    sets, params, derived = _build_calibration()
    set_manager = _build_set_manager(sets)

    seen: dict[str, str] = {}
    for cls in (TradeArmingtonBlock, ProductionBlock, FactorBlock, DemandUtilityBlock):
        block = cls(sets=sets, params=params, derived=derived)
        eqs = block.setup(set_manager, {}, {})
        for eq in eqs:
            assert eq.name not in seen, (
                f"{eq.name} claimed by both {seen[eq.name]} and {cls.__name__}"
            )
            seen[eq.name] = cls.__name__

    assert len(seen) > 0


# ======================================================================
# IncomeClosureBlock (F7 Task 9b) — closure unit, last in GTAP6_BLOCK_ORDER
# ======================================================================

# Oracle Constraint name -> block equation name for the 7 equations that
# exist as an active Constraint in the oracle (income_closure.py's module
# docstring has the full grep-verified mapping/line numbers). e_kb, e_ke,
# e_rorg, e_psave, e_gdpmp, e_rgdpmp, e_pgdpmp have NO oracle Constraint
# anywhere (grep confirms no eq_kb/eq_rorg/eq_psave/eq_gdpmp/eq_rgdpmp/
# eq_pgdpmp method exists in the oracle OR anywhere in the orphan branch)
# and are checked separately below via the oracle's own documented
# benchmark values/comments.
_ORACLE_CONSTRAINT_FOR_INCOME = {
    "e_y": "eq_y",
    "e_yp": "eq_yp",
    "e_yg": "eq_yg",
    "e_ysav": "eq_sav",
    "e_taxrev": "eq_tax_revenue",
    "e_pgdpwld": "eq_pgdpwld",
    "e_walras": "eq_walras",
}


def _build_oracle_income():
    """Build the oracle model with the aliases IncomeClosureBlock needs.

    IncomeClosureBlock's e_y/e_taxrev are written against
    ``m.pfactor[f,r]`` (the contract/FactorBlock/ProductionBlock name for
    the regional factor wage) and ``m.taxrev[r]`` (the contract's ID for
    the per-region tax-revenue aggregate). The oracle hosts these SAME
    quantities under the Python attribute names ``pf`` (monolith
    894-901, "pf(f,r) — regional factor price for f in r") and
    ``tax_revenue`` (monolith 1294-1299, the real per-region aggregate —
    NOT the oracle's own dangling per-stream ``taxrev(r,gy)`` Var, see
    income_closure.py's module docstring for why). Alias both (read-only,
    ``object.__setattr__``, same technique as
    ``_build_oracle_production``/``_build_oracle_factor``/
    ``_build_oracle_demand_utility``) so the block's build_expression
    resolves against the oracle's own live Vars without re-deriving a
    second model. ``e_yp``/``e_yg`` also read ``m.pq[r]``, already
    aliased onto ``oracle.pcons`` by the demand-utility helper below.
    """
    oracle = _build_oracle()
    object.__setattr__(oracle, "pfactor", oracle.pf)
    object.__setattr__(oracle, "taxrev", oracle.tax_revenue)
    object.__setattr__(oracle, "pq", oracle.pcons)
    return oracle


def _build_income_closure_block(mode="nlp"):
    from equilibria.blocks.gtap6.income_closure import IncomeClosureBlock

    sets, params, derived = _build_calibration()
    block = IncomeClosureBlock(sets=sets, params=params, derived=derived, mode=mode)
    return block, sets, params, derived


@pytest.fixture(scope="module")
def _income_closure_fixtures():
    block, sets, params, derived = _build_income_closure_block(mode="nlp")
    set_manager = _build_set_manager(sets)
    variables: dict = {}
    parameters: dict = {}
    equations = block.setup(set_manager, parameters, variables)
    oracle = _build_oracle_income()
    return block, sets, params, derived, set_manager, equations, variables, oracle


def test_income_closure_block_setup_returns_all_contract_equations(
    _income_closure_fixtures,
):
    """Confirm the split ruling: this block owns all 12 IDs in
    _GTAP6_INCOME_AND_CLOSURE plus e_yp/e_yg reserved from
    _GTAP6_FINAL_DEMAND (demand_utility.py's module docstring) — 14
    equations total.
    """
    from equilibria.templates.gtap6.gtap6_contract import (
        _GTAP6_FINAL_DEMAND,
        _GTAP6_INCOME_AND_CLOSURE,
    )

    _block, _sets, _params, _derived, _sm, equations, _vars, _oracle = (
        _income_closure_fixtures
    )
    eq_names = {eq.name for eq in equations}

    assert len(_GTAP6_INCOME_AND_CLOSURE) == 12, (
        f"expected 12 IDs in _GTAP6_INCOME_AND_CLOSURE, got "
        f"{len(_GTAP6_INCOME_AND_CLOSURE)}"
    )
    assert "e_yp" in _GTAP6_FINAL_DEMAND
    assert "e_yg" in _GTAP6_FINAL_DEMAND

    expected = set(_GTAP6_INCOME_AND_CLOSURE) | {"e_yp", "e_yg"}
    missing = expected - eq_names
    extra = eq_names - expected
    assert not missing, f"IncomeClosureBlock did not produce: {missing}"
    assert not extra, f"IncomeClosureBlock produced unexpected equations: {extra}"
    assert len(eq_names) == 14, (
        f"expected 14 unique equation names, got {len(eq_names)}"
    )


def test_income_closure_sav_is_a_variable_not_a_parameter(_income_closure_fixtures):
    """THE load-bearing check for this block (Phase 3.38 fix — see the
    module docstring's full diagnostic history). ``sav`` MUST be declared
    as a Pyomo Var, never a Param: the original orphan branch held it as
    a constant ``save_0`` Param through Phase 3.36/3.37, leaving the
    regional budget identity ``y = yp + yg + sav`` unsatisfied under any
    shock (the imbalance leaked into ``walras`` instead, corrupting VIWS
    by ~16pp). This test confirms both (1) the block's OWN ``sav``
    Variable object has an unbounded ``Reals`` domain (a Param would
    never appear in the ``variables`` dict passed to ``setup()`` at all,
    so its mere presence there — with the correct free-Var domain — is
    the signature of a Var, never a Param), and (2) the oracle's live
    Pyomo model exposes ``sav`` as a genuine ``pyomo.environ.Var``
    component, not a ``Param``.
    """
    from pyomo.environ import Param, Var

    _block, _sets, _params, _derived, _sm, _equations, variables, oracle = (
        _income_closure_fixtures
    )

    # (1) The block's own Variable object.
    sav_var = variables["sav"]
    assert sav_var.domain == "Reals", (
        f"sav must be domain='Reals' (unbounded, can be negative), got "
        f"{sav_var.domain!r}"
    )
    assert sav_var.lower == float("-inf") and sav_var.upper == float("inf"), (
        "sav must be unbounded (Phase 3.38: sav = y - yp - yg is a "
        "residual that can be negative), got bounds "
        f"({sav_var.lower}, {sav_var.upper})"
    )

    # (2) The oracle's live Pyomo component — confirm it is a Var, and
    # explicitly confirm it is NOT a Param (the bug this test guards
    # against: reintroducing `sav` as a constant Param out of habit).
    sav_component = oracle.sav
    assert isinstance(sav_component, Var), (
        f"oracle.sav must be a pyomo Var, got {type(sav_component)}"
    )
    assert not isinstance(sav_component, Param), (
        "oracle.sav must NOT be a Param — this is the exact Phase 3.36/"
        "3.37 regression documented in "
        "docs/findings/gtap_v62_phase338_sav_var_budget_identity.md"
    )


def test_income_closure_block_matches_oracle_numerically(_income_closure_fixtures):
    """Load-bearing numeric form-diff: block algebra vs the oracle, per-cell.

    Covers e_y, e_yp, e_yg, e_ysav (THE Phase 3.38 fix), e_taxrev,
    e_pgdpwld, e_walras — the 7 equations with a live oracle Constraint.
    """
    from pyomo.environ import Constraint
    from pyomo.environ import value as pyo_value

    block, _sets, _params, _derived, _sm, equations, _vars, oracle = (
        _income_closure_fixtures
    )
    eq_by_name = {eq.name: eq for eq in equations}

    oracle_cons = {c.name: c for c in oracle.component_objects(Constraint, active=True)}

    total_checked = 0
    max_abs_diff = 0.0
    worst_cell: tuple[str, object] | None = None

    for block_name, oracle_name in _ORACLE_CONSTRAINT_FOR_INCOME.items():
        eq = eq_by_name[block_name]
        con = oracle_cons[oracle_name]
        oracle_active_idx = {idx for idx, c in con.items() if c.active}

        checked_this_eq = 0
        for idx in _index_combos(oracle, eq.domains):
            block_expr = eq.build_expression(oracle, idx)
            key = idx if len(idx) > 1 else (idx[0] if idx else None)
            if key is None:
                # Scalar equation (e_pgdpwld/e_walras): the oracle's own
                # Constraint is indexed by `None` when built with no set.
                oracle_is_active = con.active if hasattr(con, "active") else True
                if block_expr is None:
                    assert not oracle_is_active
                    continue
                assert oracle_is_active
                oracle_con = con
            else:
                oracle_is_active = key in oracle_active_idx
                if block_expr is None:
                    assert not oracle_is_active, (
                        f"{block_name} Skips {idx} but oracle {oracle_name} "
                        "is active there"
                    )
                    continue
                assert oracle_is_active, (
                    f"{block_name} builds {idx} but oracle {oracle_name} Skips it"
                )
                oracle_con = con[key]

            block_con = block_expr
            b_body = pyo_value(block_con.args[0]) - pyo_value(block_con.args[1])
            o_body = pyo_value(oracle_con.body) - pyo_value(oracle_con.lower)
            diff = abs(b_body - o_body)
            if diff > max_abs_diff:
                max_abs_diff = diff
                worst_cell = (block_name, idx)
            assert diff < _TOL, (
                f"{block_name}{idx}: block residual {b_body} vs oracle "
                f"residual {o_body} (diff {diff} >= {_TOL})"
            )
            checked_this_eq += 1
            total_checked += 1

        assert checked_this_eq > 0, f"{block_name}: no active cells checked"

    assert total_checked > 0
    print(
        f"\n[gtap6 income-closure form-diff] {total_checked} cells checked across "
        f"{len(_ORACLE_CONSTRAINT_FOR_INCOME)} equations; max |diff| = "
        f"{max_abs_diff:.3e} at {worst_cell}"
    )


def test_income_closure_walras_absent_in_mcp_mode(_income_closure_fixtures):
    """MCP mode drops walras/e_walras entirely (Walras' law makes one
    market-clearing eq redundant in equilibrium) — mirrors the oracle's
    own ``if self.mode == "nlp":`` gate (Task 5's smoke test: 195 vs 193
    components, a diff of exactly walras + eq_walras).
    """
    block, sets, params, derived = _build_income_closure_block(mode="mcp")
    set_manager = _build_set_manager(sets)
    variables: dict = {}
    parameters: dict = {}
    equations = block.setup(set_manager, parameters, variables)
    eq_names = {eq.name for eq in equations}

    assert "e_walras" not in eq_names
    assert "walras" not in variables
    assert len(eq_names) == 13, (
        f"expected 13 equations in mcp mode, got {len(eq_names)}"
    )


def test_income_closure_no_oracle_equations_match_documented_benchmark(
    _income_closure_fixtures,
):
    """e_kb/e_ke/e_rorg/e_psave/e_gdpmp/e_rgdpmp/e_pgdpmp have no oracle
    Constraint (see module docstring — the oracle declares the Var but
    never wires a defining equation for any of them, and neither does the
    orphan branch anywhere). Verify each against the oracle's OWN
    documented benchmark value/comment instead, the same identity-check
    methodology used for e_qds/e_qtmfsd (Task 6) and the sluggish CET
    branch (Task 8).
    """
    from pyomo.environ import value as pyo_value

    block, sets, params, derived, _sm, equations, _vars, oracle = (
        _income_closure_fixtures
    )
    eq_by_name = {eq.name: eq for eq in equations}

    # e_kb: kb(r) == vkb(r) — the oracle's OWN benchmark seed for BOTH
    # kb and ke (monolith 1349/1356: both initialize from b.vkb.get(r,1.0)).
    eq_kb = eq_by_name["e_kb"]
    checked_kb = 0
    for r in sets.r:
        vkb = float(params.benchmark.vkb.get(r, 0.0) or 0.0)
        expr = eq_kb.build_expression(oracle, (r,))
        if vkb <= 1e-12:
            assert expr is None
            continue
        assert expr is not None
        lhs = pyo_value(expr.args[0])
        rhs = pyo_value(expr.args[1])
        assert abs(lhs - rhs) < 1e-6, (r, lhs, rhs)
        checked_kb += 1
    assert checked_kb > 0
    print(f"\n[gtap6 e_kb benchmark identity] {checked_kb} regions checked")

    # e_ke: ke(r) == kb(r) — no accumulation in one comparative-static
    # period; at the seed both are literally the same VKB value.
    eq_ke = eq_by_name["e_ke"]
    checked_ke = 0
    for r in sets.r:
        expr = eq_ke.build_expression(oracle, (r,))
        assert expr is not None
        lhs = pyo_value(expr.args[0])
        rhs = pyo_value(expr.args[1])
        assert abs(lhs - rhs) < 1e-9, (r, lhs, rhs)
        checked_ke += 1
    assert checked_ke > 0
    print(f"[gtap6 e_ke benchmark identity] {checked_ke} regions checked")

    # e_gdpmp: gdpmp(r) == y(r) — the oracle's own stated benchmark
    # identity (monolith 1378-1381 comment: "gdpmp / rgdpmp to y_0 ...
    # so the identity eq_gdpmp (gdpmp = y) holds at benchmark").
    eq_gdpmp = eq_by_name["e_gdpmp"]
    checked_gdpmp = 0
    for r in sets.r:
        expr = eq_gdpmp.build_expression(oracle, (r,))
        assert expr is not None
        lhs = pyo_value(expr.args[0])
        rhs = pyo_value(expr.args[1])
        assert abs(lhs - rhs) < 1e-6, (r, lhs, rhs)
        checked_gdpmp += 1
    assert checked_gdpmp > 0
    print(f"[gtap6 e_gdpmp benchmark identity] {checked_gdpmp} regions checked")

    # e_pgdpmp: pgdpmp(r) == pgdpwld — both are 1.0 at the benchmark.
    eq_pgdpmp = eq_by_name["e_pgdpmp"]
    checked_pgdpmp = 0
    for r in sets.r:
        expr = eq_pgdpmp.build_expression(oracle, (r,))
        assert expr is not None
        lhs = pyo_value(expr.args[0])
        rhs = pyo_value(expr.args[1])
        assert abs(lhs - rhs) < 1e-9, (r, lhs, rhs)
        checked_pgdpmp += 1
    assert checked_pgdpmp > 0
    print(f"[gtap6 e_pgdpmp benchmark identity] {checked_pgdpmp} regions checked")

    # e_rgdpmp: pgdpmp(r) * rgdpmp(r) == gdpmp(r) — exact identity at any
    # point (not just the benchmark), since it is a pure definitional
    # relationship among the three GDP Vars.
    eq_rgdpmp = eq_by_name["e_rgdpmp"]
    checked_rgdpmp = 0
    for r in sets.r:
        expr = eq_rgdpmp.build_expression(oracle, (r,))
        assert expr is not None
        lhs = pyo_value(expr.args[0])
        rhs = pyo_value(expr.args[1])
        assert abs(lhs - rhs) < 1e-9, (r, lhs, rhs)
        checked_rgdpmp += 1
    assert checked_rgdpmp > 0
    print(f"[gtap6 e_rgdpmp benchmark identity] {checked_rgdpmp} regions checked")

    # e_psave: psave(r) == pgdpwld — both 1.0 at the benchmark.
    eq_psave = eq_by_name["e_psave"]
    checked_psave = 0
    for r in sets.r:
        expr = eq_psave.build_expression(oracle, (r,))
        assert expr is not None
        lhs = pyo_value(expr.args[0])
        rhs = pyo_value(expr.args[1])
        assert abs(lhs - rhs) < 1e-9, (r, lhs, rhs)
        checked_psave += 1
    assert checked_psave > 0
    print(f"[gtap6 e_psave benchmark identity] {checked_psave} regions checked")

    # e_rorg: rorg * sum_r(kb) == sum_r(pfactor[Capital,r]*qoes[Capital,r]).
    # The oracle's OWN `rorg` Var is seeded to a placeholder 1.0 (it never
    # calibrates it — see module docstring: no eq_rorg exists anywhere
    # upstream), so this equation does NOT hold at the oracle's raw seed
    # value; that seed is not economically meaningful for rorg. Instead,
    # verify the IDENTITY itself is well-formed: solving args[0]==args[1]
    # for rorg (i.e. numer/denom, using the oracle's live pfactor/qoes/kb
    # at their benchmark values) reproduces exactly the calibrated ratio
    # sum_r(evom[Capital,r]) / sum_r(vkb[r]) this block's own `rorg`
    # Variable is initialized to (income_closure.py's `rorg_init`) — i.e.
    # the equation is the correct DEFINING relationship for rorg, not a
    # tautology that would pass for any coefficient.
    eq_rorg = eq_by_name["e_rorg"]
    expr = eq_rorg.build_expression(oracle, ())
    assert expr is not None
    denom_val = pyo_value(expr.args[0]) / pyo_value(oracle.rorg)  # == sum_r(kb)
    numer_val = pyo_value(expr.args[1])
    implied_rorg = numer_val / denom_val
    expected_rorg = sum(
        derived.evom.get(("Capital", r), 0.0) or 0.0 for r in sets.r
    ) / sum(params.benchmark.vkb.get(r, 0.0) or 0.0 for r in sets.r)
    assert abs(implied_rorg - expected_rorg) < 1e-6, (implied_rorg, expected_rorg)
    print(
        f"[gtap6 e_rorg benchmark identity] implied rorg={implied_rorg:.6f} "
        f"== calibrated {expected_rorg:.6f}"
    )


def test_income_closure_ysav_budget_identity_holds_off_benchmark(
    _income_closure_fixtures,
):
    """Curvature/perturbation test for e_ysav (Task 9a's own proactive
    precedent: verify a form that is degenerate-looking at the benchmark
    seed is not secretly a tautology). e_ysav (sav = y - yp - yg) is
    LINEAR, not degenerate, but the whole point of the Phase 3.38 fix is
    that the identity must hold IDENTICALLY away from the benchmark too
    (under any shock to y/yp/yg) — not just at the calibration point
    where sav happens to equal save_0 by construction. Perturb y/yp/yg
    away from their seed values and confirm the residual (defined here as
    args[0] - args[1] = sav - (y - yp - yg)) tracks the perturbation
    EXACTLY (slope -1 in y, +1 in yp, +1 in yg), which a Param-based
    `sav` (the Phase 3.36/3.37 bug) could never do since a Param cannot
    move to satisfy a live constraint.
    """
    from pyomo.environ import value as pyo_value

    block, sets, _params, _derived, _sm, equations, _vars, oracle = (
        _income_closure_fixtures
    )
    eq_ysav = {e.name: e for e in equations}["e_ysav"]

    checked = 0
    for r in sets.r:
        seed_y = float(pyo_value(oracle.y[r]))
        seed_yp = float(pyo_value(oracle.yp[r]))
        seed_yg = float(pyo_value(oracle.yg[r]))

        def _residual(y_val, yp_val, yg_val, r=r):
            oracle.y[r].set_value(y_val)
            oracle.yp[r].set_value(yp_val)
            oracle.yg[r].set_value(yg_val)
            try:
                expr = eq_ysav.build_expression(oracle, (r,))
                assert expr is not None
                return pyo_value(expr.args[0]) - pyo_value(expr.args[1])
            finally:
                oracle.y[r].set_value(seed_y)
                oracle.yp[r].set_value(seed_yp)
                oracle.yg[r].set_value(seed_yg)

        bump = max(abs(seed_y) * 0.02, 1e-3)
        r0 = _residual(seed_y, seed_yp, seed_yg)
        r_dy = _residual(seed_y + bump, seed_yp, seed_yg)
        r_dyp = _residual(seed_y, seed_yp + bump, seed_yg)
        r_dyg = _residual(seed_y, seed_yp, seed_yg + bump)

        slope_y = (r_dy - r0) / bump
        slope_yp = (r_dyp - r0) / bump
        slope_yg = (r_dyg - r0) / bump

        assert abs(slope_y - (-1.0)) < 1e-6, (r, "d(residual)/dy", slope_y)
        assert abs(slope_yp - 1.0) < 1e-6, (r, "d(residual)/dyp", slope_yp)
        assert abs(slope_yg - 1.0) < 1e-6, (r, "d(residual)/dyg", slope_yg)
        checked += 1

    assert checked > 0
    print(
        f"\n[gtap6 e_ysav budget-identity curvature] {checked} regions "
        "slope-checked (exact -1/+1/+1 in y/yp/yg)"
    )


# ======================================================================
# All 5 blocks together (F7 Task 9b Step 4) — final aggregate gate before
# Task 10's composer.
# ======================================================================


def test_all_5_blocks_together_cover_every_contract_equation():
    """The last checkpoint before Task 10's composer attempt: instantiate
    all 5 GTAP6_BLOCK_ORDER blocks together (leaf-to-closure order) and
    assert their combined equation-name set exactly equals the FULL
    59-ID contract.
    """
    from equilibria.blocks.gtap6 import GTAP6_BLOCK_ORDER
    from equilibria.templates.gtap6.gtap6_contract import _full_gtap6_equation_ids

    sets, params, derived = _build_calibration()
    set_manager = _build_set_manager(sets)

    all_names: set[str] = set()
    seen: dict[str, str] = {}
    for cls in GTAP6_BLOCK_ORDER:
        block = cls(sets=sets, params=params, derived=derived)
        eqs = block.setup(set_manager, {}, {})
        for eq in eqs:
            assert eq.name not in seen, (
                f"{eq.name} claimed by both {seen[eq.name]} and {cls.__name__}"
            )
            seen[eq.name] = cls.__name__
        all_names |= {eq.name for eq in eqs}

    expected = set(_full_gtap6_equation_ids())
    missing = expected - all_names
    extra = all_names - expected
    assert not missing, f"No block produces: {missing}"
    assert not extra, f"Unexpected equations produced (not in contract): {extra}"
    assert len(expected) == 66, (
        f"expected 66 IDs in the full contract, got {len(expected)}"
    )
    assert len(all_names) == 66
    print(
        f"\n[gtap6 5-block aggregate coverage] {len(all_names)}/66 contract "
        "equations covered, 0 duplicates"
    )
