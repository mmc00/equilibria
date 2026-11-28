"""MANAGE's name dictionary is complete against the GAMS side (ADR-0009).

Oracle: the CONVERT dictionary of the 2015 solve, the 167 equation and 136
variable families that are active in it. Until a block declares a family, its
row is GAMS-only with ``why: "pending: Task N ..."`` (the plan's task)."""

from __future__ import annotations

from pathlib import Path

from equilibria.compare.names import load_names
from equilibria.templates.manage.convert import read_dict


def test_every_gams_family_has_its_row(dict_path: Path) -> None:
    names = read_dict(dict_path)
    d = load_names("manage")
    for kind, fams in (
        ("eq", {f for f, _ in names.rows}),
        ("var", {f for f, _ in names.cols}),
    ):
        missing = sorted(f for f in fams if d.lookup("gams", kind, f) is None)
        assert not missing, (kind, missing)


def test_no_row_for_a_family_that_does_not_exist(dict_path: Path) -> None:
    names = read_dict(dict_path)
    known = {f for f, _ in names.rows} | {f for f, _ in names.cols}
    stale = sorted(e.gams for e in load_names("manage").entries if e.gams not in known)
    assert not stale


def test_pending_rows_name_their_task() -> None:
    for e in load_names("manage").entries:
        if e.equilibria is None:
            assert e.why.startswith("pending: Task "), e
