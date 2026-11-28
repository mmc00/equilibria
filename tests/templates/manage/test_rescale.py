"""The table of what GAMS rescaled before writing GHA_BaU.gdx (inc/cal/rescale.inc).

Oracle independent of the table: rescale.inc itself, parsed here. Every
assignment inside its ``if( rescale ne 1, ... );`` block must be in the table
with the same factor, and nothing else.
"""

from __future__ import annotations

import re
from pathlib import Path

from tests.templates.manage.conftest import gams_run

from equilibria.sources import DATA_ROOT, load_source
from equilibria.templates.manage.rescale import RESCALED

# The model the recorded run was made from (its source.yaml, ADR-0010).
MODEL = DATA_ROOT / load_source(gams_run()).raw["generated_from"]["model"]["path"]
RESCALE_INC = MODEL / "inc" / "cal" / "rescale.inc"
_LHS = re.compile(r"^\s*(?:\$\$ifi? defined \w+\s+)?(\w+)(?:\.l)?\s*(?:\(|=|\$)", re.I)


def _rescale_inc() -> dict[str, tuple[int, dict[str, int]]]:
    """{symbol (lower case): (line, exponents)} of every assignment in the block."""
    out: dict[str, tuple[int, dict[str, int]]] = {}
    lines = RESCALE_INC.read_text().splitlines()
    start = next(i for i, ln in enumerate(lines) if "rescale ne 1" in ln)
    for n, line in enumerate(lines[start + 1 :], start + 2):
        if line.strip().startswith(");"):
            break
        if line.lstrip().startswith("*") or "=" not in line:
            continue
        m = _LHS.match(line)
        assert m, line
        rhs = line.split("=", 1)[1].replace(" ", "").lower()
        exps = {
            "rescale": rhs.count("*rescale") - rhs.count("/rescale"),
            "reescale": rhs.count("*reescale") - rhs.count("/reescale"),
            "usd": rhs.count("*usd"),
        }
        out[m.group(1).lower()] = (n, exps)
    return out


def test_table_matches_rescale_inc() -> None:
    inc = _rescale_inc()
    assert len(inc) == 109
    table = {k: v for k, v in RESCALED.items() if v.source.startswith("rescale.inc")}
    assert set(table) == set(inc)
    for name, (line, exps) in inc.items():
        rule = table[name]
        assert rule.source == f"rescale.inc:{line}", name
        got = {"rescale": rule.rescale, "reescale": rule.reescale, "usd": rule.usd}
        assert got == exps, name


def test_lcovarea_rule_from_run_gms() -> None:
    rule = RESCALED["lcovarea"]
    assert rule.source == "run.gms:402"
    assert (rule.inscale, rule.usd) == (-1, -1)
