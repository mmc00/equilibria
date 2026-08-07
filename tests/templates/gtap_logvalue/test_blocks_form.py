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


def test_factor_block_builds():
    from equilibria.blocks.gtap_logvalue.factor import FactorLVBlock
    from equilibria.blocks.gtap_logvalue.production_supply import (
        ProductionSupplyLVBlock,
    )

    pm = build_one_block(FactorLVBlock, deps=(ProductionSupplyLVBlock,))
    cons = _con_names(pm)
    for fam in [
        "e_peb",
        "e_pfe",
        "e_pes",
        "e_pfactor",
        "e_pe1",
        "e_qes1",
        "e_qes2",
        "e_pe2",
        "e_qes3",
    ]:
        assert any(n.startswith(fam) for n in cons), (
            f"{fam} not built; got {sorted(cons)}"
        )


def test_all_seven_blocks_compose():
    """The 7 log-value blocks compose together: every referenced var resolves by
    name (dedup first-wins), and every equation family builds. This is the coherence
    gate; numeric identity vs the port is the Task-10 transitive solve gate."""
    from tests.templates.gtap_logvalue._harness import build_all_blocks

    pm = build_all_blocks()
    cons = _con_names(pm)
    # spot-check one family from each block is present
    for fam in ["e_qo", "e_peb", "e_qxs", "e_qds", "e_qpa", "e_y", "e_rorg"]:
        assert any(n.startswith(fam) for n in cons), (
            f"{fam} missing; got {len(cons)} cons"
        )
