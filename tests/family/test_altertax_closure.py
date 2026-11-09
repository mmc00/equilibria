"""ALTERTAX closure table (GEMPACK ``altax.exp`` -> equilibria) and closure config."""

from __future__ import annotations

from equilibria._local_refs import nus333_dir
from equilibria.family.altertax.closure import (
    ALTAX_RESIDUAL_REGION,
    altertax_closure,
    load_closure_map,
    parse_exp_exogenous,
)

KINDS = {"fixed_var", "fixed_param", "by_construction", "absent"}


def _exo() -> list[str]:
    return parse_exp_exogenous(nus333_dir() / "altax.exp")


def test_parse_exp_exogenous_reads_the_block():
    names = _exo()
    assert "pop" in names and "pfactwld" in names and "tim" in names
    assert 'del_tbal("USA")' in names and 'cgdslack("ROW")' in names
    assert len(names) == len(set(names))
    # the commented-out `cgdslack` (no element) and the comment words are not names
    assert "cgdslack" not in names and "Altertax" not in names
    assert "Rest" not in names


def test_closure_map_covers_exp():
    rows = load_closure_map()
    have = {row["gempack"] for row in rows}
    assert have >= set(_exo())
    assert "pfactwld" in have
    by = {row["gempack"]: row for row in rows}
    for name in ('del_tbal("USA")', 'cgdslack("ROW")'):
        assert by[name]["kind"] != "absent" or by[name]["reason"].strip()


def test_every_row_has_reason():
    for row in load_closure_map():
        assert set(row) == {"gempack", "equilibria", "kind", "reason"}
        assert row["kind"] in KINDS, row
        assert row["reason"].strip(), row
        if row["kind"] in ("fixed_var", "fixed_param"):
            assert row["equilibria"], row
        if row["kind"] == "absent":
            assert row["equilibria"] is None, row


def test_altertax_closure_fields():
    c = altertax_closure()
    assert c.name == "altertax_gempack" and c.name != "altertax"
    assert c.closure_type == "MCP"
    assert c.if_sub is True
    assert c.savf_flag == "capFix"
    assert c.va_subsidy_basis == "gempack"
    assert c.numeraire == "pnum"
    assert (c.fix_endowments, c.fix_taxes, c.fix_technology) == (False, False, False)
    assert ALTAX_RESIDUAL_REGION == "ROW"


def test_fixed_var_rows_name_a_variable_of_the_arrays():
    """The equilibria side of every ``fixed_var`` row is a Var of the family arrays."""
    from equilibria.family.mp_arrays import MPArrays
    from equilibria.family.prep_capture import pairs
    from tests.family._prep import capture_dir

    mp = MPArrays.from_photo(pairs(capture_dir("gtap7_3x3"))[0][0])
    for row in load_closure_map():
        if row["kind"] == "fixed_var":
            assert row["equilibria"].split("[")[0] in mp.vars, row
