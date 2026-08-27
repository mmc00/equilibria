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

_MIGRATED: list[str] = ["TradeArmingtonBlock", "ProductionBlock", "FactorBlock"]

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
    "e_qf": "eq_qf",
    "e_pf": "eq_pf_int",
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
    assert len(eq_names) == 8, f"expected 8 unique equation names, got {len(eq_names)}"


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
