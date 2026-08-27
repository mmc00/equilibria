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

_MIGRATED: list[str] = ["TradeArmingtonBlock"]

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
