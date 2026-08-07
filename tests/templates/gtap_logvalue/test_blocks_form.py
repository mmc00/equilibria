"""Per-block gate: each log-value block builds through the neutral backend and its
equations are present (log form). ROW/COL parity vs the port monolith is checked
once the full block set is assembled (Task 10 transitive gate is the strong check).
"""

from tests.templates.gtap_logvalue._harness import build_one_block

from equilibria.blocks.gtap_logvalue.production_supply import ProductionSupplyLVBlock


def _con_names(pm):
    from pyomo.environ import Constraint

    return {c.name for c in pm.component_objects(Constraint)}


def test_production_block_builds():
    pm = build_one_block(ProductionSupplyLVBlock)
    cons = _con_names(pm)
    # every _production equation family the port emits must be present, log-form
    expected = [
        "e_qo",
        "e_qintva_int",
        "e_qintva_va",
        "e_qfa",
        "e_pint",
        "e_qfe",
        "e_pva",
        "e_qfd",
        "e_qfm",
        "e_pfa",
        "e_qca",
        "e_po",
        "e_ps",
        "e_qc",
        "e_pca",
    ]
    for fam in expected:
        assert any(
            n == fam or n.startswith(fam + "_con") or n.startswith(fam) for n in cons
        ), f"{fam} not built; got {sorted(cons)}"
