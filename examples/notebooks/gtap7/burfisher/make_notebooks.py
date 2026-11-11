"""Writes one notebook per Burfisher exercise: replicate it from scratch, compare with GEMPACK.

Each notebook loads the nus333 database and the exercise's ``.prm``, sets the GTAP
closure, applies the exercise's shock, solves base/check/shock, and compares every
level against GEMPACK's solution of the same ``.EXP``. It passes when every cell is
within 0.5% (the criterion of the paper's Table 3a).

The standard exercises come from ``scripts/gtap/run_burfisher.py`` (``EXERCISES``,
``SHOCK_STEPS``), so the notebooks and the paper's tables run the same shocks. The
exercises whose closure differs from GTAP's (TBL65B, ME3C, TBL94, ME9A) add their
``@overwrite`` hooks below.

The six exercises the book solves with Johansen (``Method = Johansen`` in the
``.EXP``) also solve with ``shock_method="johansen"`` and compare, in linear form,
with the book's GEMPACK Johansen run. ME4A and ME4B have only that comparison:
solved exactly, neither GEMPACK nor equilibria reach their shock.

Usage, from the repository root::

    uv run python examples/notebooks/gtap7/burfisher/make_notebooks.py [--out DIR]

``tests/examples/test_burfisher_notebooks.py`` checks that the committed notebooks
are exactly what this script writes, and runs them.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT / "scripts" / "gtap"))

import run_burfisher as rb  # noqa: E402

from equilibria.sources import find_reference  # noqa: E402

# Solved exactly, neither GEMPACK ("Accuracy too low") nor equilibria reach the shock:
# these get only the Johansen comparison.
JOHANSEN_ONLY = ("ME4A", "ME4B")
# The book solves these with Johansen (its .EXP says Method = Johansen).
JOHANSEN = ("TBL45A", "TBL45B", "TBL45C", "TBL64", *JOHANSEN_ONLY)
# The book's own GEMPACK runs, with each .EXP's method: compare_burfisher_gempack's
# "book" source (data/gtap7/nus333/gempack_burfisher/<exp>/).
GEMPACK_BOOK = '"book"'
# GEMPACK's Gragg solutions of the book's exercises: compare_burfisher_gempack's
# "gragg" source (data/gtap7/nus333/gemsim_burfisher_gragg/<exp>/). ME4A and ME4B
# have none (D1 of plan-phase-3b: Gragg fails, "Accuracy too low").
GEMPACK_GRAGG = '"gragg"'

UNEMPLOYMENT = """\
@overwrite(ShockBlock, period="shock")
def unemployment(b):
    b.endogenous("aft", ("USA", "LABOR"))


@overwrite(ClosureBlock, period="shock")
def real_wage(b):
    # When the row is built, value(...) is the base level (a constant).
    b.equation(
        "eq_wreal",
        ("USA", "LABOR"),
        lambda m, r, f: m.pft[r, f] == value(m.pft[r, f]) * ppriv_tornqvist(m, r),
    )
"""

REGULATION = """\
CELL = ("USA", "MFG", "MFG")


@overwrite(ShockBlock, period="shock")
def regulation(b):
    b.endogenous("prdtx_rai", CELL)
    b.target(
        "qca_target",
        CELL,
        quantity=lambda m, r, a, i: m.x[r, a, i],
        domains=("r", "a", "i"),
    )
"""

BASELINE = """\
REGS = ("USA", "ROW")
ACTS = ("AGR", "MFG", "SER")


@overwrite(ShockBlock, period="shock")
def baseline(b):
    for r in REGS:
        for a in ACTS:
            b.endogenous("axp", (r, a))
        b.target("gdp_target", (r,), quantity=lambda m, r: m.rgdpmp[r], domains=("r",))


@overwrite(ClosureBlock, period="shock")
def uniform_productivity(b):
    for r in REGS:
        for a in ("MFG", "SER"):
            b.equation(
                "eq_aoreg",
                (r, a),
                lambda m, r, a: m.axp[r, a] / value(m.axp[r, a])
                == m.axp[r, "AGR"] / value(m.axp[r, "AGR"]),
                domains=("r", "a"),
            )
"""

_UNEMPLOYMENT_MD = (
    'GEMPACK\'s `ME3C.EXP` swaps `qe("LABOR","USA")` for '
    '`pebfactreal("LABOR","USA")`: the real wage of USA is fixed and employment '
    "adjusts. Here, with two `@overwrite` hooks that act only in the shock period:\n\n"
    "- `ShockBlock`: labour supply `aft[USA,LABOR]` becomes endogenous;\n"
    "- `ClosureBlock`: a new row `eq_wreal`, `pft = pft0 × T`, with `T` the Törnqvist "
    "index of household consumption prices against the base.\n\n"
    "One unknown freed, one row added: the system stays square."
)

# exp: (prm, shock rows, hook code, markdown that explains the hook, checked)
HOOKED: dict[str, tuple[str, list, str, str, bool]] = {
    "TBL65B": (
        "default.prm",
        [("prdtx_rai", ("USA", "MFG", "MFG"), -10.9414)],
        UNEMPLOYMENT,
        _UNEMPLOYMENT_MD,
        True,
    ),
    "ME3C": (
        "default.prm",
        [("prdtx_rai", ("USA", "MFG", "MFG"), -10.9414)],
        UNEMPLOYMENT,
        _UNEMPLOYMENT_MD + "\n\nME3C runs the same `.EXP` as TBL65B.",
        True,
    ),
    "TBL94": (
        "default.prm",
        [("qca_target", ("USA", "MFG", "MFG"), -1.0)],
        REGULATION,
        "A regulation cuts the output of MFG by the MFG activity of USA by 1%. The "
        "model finds the production tax that has the same effect: the shadow tax "
        "of the regulation. GEMPACK's `TBL94.EXP` swaps "
        '`qca("MFG","MFG","USA")` for `to("MFG","MFG","USA")` and '
        "shocks `qca = -1`. Here:\n\n"
        "- the production tax `prdtx_rai[USA,MFG,MFG]` becomes endogenous;\n"
        "- `b.target` declares the target `qca_target` on output `x`: a new "
        "variable, fixed at 1, and the row `x[shock] = qca_target × x[check]`.\n\n"
        "One unknown freed, one row added: the system stays square.",
        True,
    ),
    "ME9A": (
        "climatechange.prm",
        [("gdp_target", ("USA",), 109.6), ("gdp_target", ("ROW",), 284.5)]
        + [s for s in rb.me9(land=(-0.93, 4.4), ao_agr=0.0, afe=None) if s[0] != "axp"],
        BASELINE,
        "A 2010-2050 baseline: real GDP grows by a given amount and productivity "
        "adjusts. GEMPACK's `ME9A.EXP` swaps `qgdp(REG)` for `aoreg(REG)`. Here:\n\n"
        "- productivity `axp` becomes endogenous in every activity, and "
        "`b.target` fixes real GDP `rgdpmp[shock] = gdp_target × rgdpmp[check]`;\n"
        "- `eq_aoreg` keeps the productivity change the same in the three "
        "activities of a region, as `aoreg` does.\n\n"
        "**Not yet within 0.5% of GEMPACK:** up to 0.77% (`pe` ROW LAND), while it "
        "matches GAMS. GEMPACK's own log rates many of its ME9A results to 2 "
        "figures; a rerun with more steps will decide which side moves.",
        False,
    ),
}

CLOSURE = """\
gc = GTAPClosureConfig(
    name="base",
    closure_type="MCP",
    capital_mobility="sluggish",
    fix_endowments=False,
    fix_taxes=False,
    fix_technology=False,
    if_sub=False,
    savf_flag="capFlex",
    numeraire="pnum",
)
# Like GAMS compStat, the base is not solved: its levels are the reference of the
# Fisher indices of check and shock.
m, _ = build_block_model(p, p.sets, gc, "ROW", base_calibrated=False, ref_gdx=None)
"""

SOLVE = f"""\
res = solve_multiperiod(
    m,
    p,
    gc,
    ref_gdx=None,
    skip_base_solve=True,
    mute_welfare=True,
    seed_from_prior=False,
    mode="gtap",
    solve_check=True,
    # the shock in {len(rb.SHOCK_STEPS)} steps, each solve starting from the previous one
    shock_steps={rb.SHOCK_STEPS!r},
)
codes = {{k: int(res[k]["code"]) for k in res}}
print("codes:", codes)
assert set(codes.values()) == {{1}}, codes
"""


def _solve_johansen(model: str) -> str:
    return f"""\
res = solve_multiperiod(
    {model},
    p,
    gc,
    ref_gdx=None,
    skip_base_solve=True,
    mute_welfare=True,
    seed_from_prior=False,
    mode="gtap",
    solve_check=True,
    shock_method="johansen",  # one linear step from check, as GEMPACK's Johansen
)
codes = {{k: int(res[k]["code"]) for k in res}}
print("codes:", codes)
assert set(codes.values()) == {{1}}, codes
"""


def _compare_johansen(exp: str, model: str) -> str:
    return (
        "import compare_burfisher_gempack as cbg\n"
        "import compare_gempack_levels as cg\n\n"
        f'EXP = "{exp}"\n'
        f"BOOK = {GEMPACK_BOOK}\n"
        'GOVERNMENT = {"qgd", "qgm"}  # government demand: the open gap of #201\n'
        f"report = cbg.run(EXP, cg.model_levels({model}), dict(xscale_data(p, p.sets)), "
        "BOOK, linear=True)\n"
        "cells = [c for c in report.cells if c.variable not in GOVERNMENT]\n"
        "gov = max(c.d for c in report.cells if c.variable in GOVERNMENT)\n"
        "print(f\"{len(cells)} cells (pre and post) against the book's GEMPACK run, "
        'method {cbg.method(EXP, BOOK)}")\n'
        'print(f"largest difference: {100 * max(c.d for c in cells):.3f}%")\n'
        'print(f"government demand qgd/qgm (#201, not checked): up to {100 * gov:.2f}%")\n'
        'print("\\nlargest differences:")\n'
        "for c in sorted(cells, key=lambda c: -c.d)[:5]:\n"
        '    cell = ",".join(c.index)\n'
        '    print(f"{c.variable:10s}{c.period:>7s}  {cell:24s}{c.a:14.6g}{c.b:14.6g}'
        '{100 * c.d:8.3f}%")\n'
        "assert max(c.d for c in cells) <= 5e-3, (\n"
        '    f"{EXP}: not every cell within 0.5% of the book\'s Johansen result"\n'
        ")\n"
        'print("every cell within 0.5% of the book\'s Johansen result (qgd/qgm aside, #201)")\n'
    )


_JOHANSEN_MD = (
    "The book solves {exp} with GEMPACK's Johansen method (`Method = Johansen; "
    "Steps = 1` in the `.EXP`): the model linearized at the starting point, solved "
    'once. `shock_method="johansen"` gives the same solution: the slope of the '
    "solution path at check (finite differences of the system PATH solves), times "
    "the whole shock. A Johansen result is compared in linear form: a product moves "
    "by the sum of the % changes of its factors."
)

_WALL_MD = (
    "\n\n**Why only Johansen here.** A land productivity loss of 81% in ROW "
    "agriculture is too large to solve exactly: on equilibria's nonlinear path ROW "
    "investment turns negative at 55% of the shock and the solve fails at 60%; "
    'GEMPACK\'s Gragg run reports negative flows and "Accuracy too low". The '
    "book's numbers are a first-order answer, and that is what this notebook "
    "reproduces."
)


def _setup(hooked: bool) -> str:
    hooks = (
        "from equilibria.blocks.gtap import ClosureBlock, ShockBlock, overwrite, "
        "ppriv_tornqvist\n"
        if hooked
        else ""
    )
    return (
        "import sys\n"
        "from pathlib import Path\n\n"
        "# the repository root, wherever the notebook is opened from\n"
        "ROOT = next(d for d in (Path.cwd(), *Path.cwd().parents) "
        'if (d / "pyproject.toml").exists())\n'
        'sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts" / "gtap")]\n\n'
        + ("from pyomo.environ import value\n\n" if hooked else "")
        + "from equilibria._local_refs import nus333_dir, path_capi_src\n"
        + hooks
        + "from equilibria.blocks.gtap._derived_params import xscale_data\n"
        "from equilibria.templates.gtap import GTAPParameters\n"
        "from equilibria.templates.gtap.gtap_block_model import build_block_model\n"
        "from equilibria.templates.gtap.gtap_contract import GTAPClosureConfig\n"
        "from equilibria.templates.gtap.gtap_multiperiod_driver import solve_multiperiod\n"
        "from equilibria.templates.gtap.shocks import apply_shock\n\n"
        "sys.path.insert(0, str(path_capi_src()))  # the PATH solver's Python bindings\n"
    )


def _data(prm: str) -> str:
    return (
        "har = nus333_dir()\n"
        "p = GTAPParameters()\n"
        "p.load_from_har(\n"
        '    basedata_path=har / "basedata.har",\n'
        '    sets_path=har / "sets.har",\n'
        f'    default_path=har / "{prm}",\n'
        '    baserate_path=har / "baserate.har",\n'
        ")\n"
    )


def _shock(rows: list) -> str:
    lines = ["SHOCK = {"]
    for name, cells in rb.as_shock(rows).items():
        if len(cells) == 1:
            ((idx, pct),) = cells.items()
            lines.append(f"    {name!r}: {{{idx!r}: {pct!r}}},".replace("'", '"'))
            continue
        lines.append(f'    "{name}": {{')
        lines += [
            f"        {idx!r}: {pct!r},".replace("'", '"') for idx, pct in cells.items()
        ]
        lines.append("    },")
    lines.append("}")
    return "\n".join(lines) + "\napply_shock(m, SHOCK)\n"


def _compare(exp: str, checked: bool) -> str:
    check = (
        '\nassert o["max"] <= 5e-3, f"{EXP}: not every cell within 0.5% of GEMPACK"\n'
        'print("every cell within 0.5% of GEMPACK")\n'
        if checked
        else ""
    )
    return (
        "import compare_burfisher_gempack as cbg\n"
        "import compare_gempack_levels as cg\n\n"
        f'EXP = "{exp}"\n'
        f"GRAGG = {GEMPACK_GRAGG}\n"
        "report = cbg.run(EXP, cg.model_levels(m), dict(xscale_data(p, p.sets)), "
        "GRAGG)\n"
        "o = report.overall\n"
        'w = o["pct_within"]\n'
        "print(f\"{o['cells']} cells (pre and post) against GEMPACK {cbg.method(EXP, GRAGG)}\")\n"
        'print(f"within 0.1%: {w[1e-3]:.2f}%   within 0.5%: {w[5e-3]:.2f}%   '
        'within 1%: {w[1e-2]:.2f}%")\n'
        'print("\\nlargest differences:")\n'
        "print(f\"{'variable':10s}{'period':>7s}  {'cell':24s}{'equilibria':>14s}"
        "{'GEMPACK':>14s}{'diff':>9s}\")\n"
        "for c in sorted(report.cells, key=lambda c: -c.d)[:5]:\n"
        '    cell = ",".join(c.index)\n'
        '    print(f"{c.variable:10s}{c.period:>7s}  {cell:24s}{c.a:14.6g}{c.b:14.6g}'
        '{100 * c.d:8.3f}%")\n' + check
    )


def _md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(True)}


def _py(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.rstrip("\n").splitlines(True),
    }


def notebook(exp: str) -> dict:
    """The notebook of ``exp``, as nbformat 4 JSON."""
    if exp in HOOKED:
        prm, rows, hook, hook_md, checked = HOOKED[exp]
    else:
        (prm, rows), hook, hook_md, checked = rb.EXERCISES[exp], "", "", True
    claim = (
        "every level is checked against GEMPACK's solution of the same `.EXP`: "
        "the notebook fails unless every cell is within 0.5%"
        if checked
        else "every level is compared with GEMPACK's solution of the same `.EXP`"
    )
    cells = [
        _md(
            f"# Burfisher 3e: {exp}\n\n"
            f"Replicates exercise {exp} of Burfisher, *Introduction to Computable "
            "General Equilibrium Models* (3rd ed.), from scratch, on the book's "
            f"nus333 database (2 regions × 3 sectors), and {claim}.\n\n"
            "Run it from anywhere inside the repository."
        ),
        _py(_setup(bool(hook))),
        _md(f"## 1. Data: nus333 and the exercise's parameters (`{prm}`)"),
        _py(_data(prm)),
    ]
    if hook:
        cells += [
            _md("## 2. The exercise's closure, with `@overwrite`\n\n" + hook_md),
            _py(hook),
        ]
    n = 3 if hook else 2
    cells += [
        _md(f"## {n}. GTAP closure and model (base / check / shock)"),
        _py(CLOSURE),
        _md(
            f"## {n + 1}. The shock\n\n"
            "As in the `.EXP`, in % change; `apply_shock` converts each instrument to "
            "levels and writes it in the shock period only."
        ),
        _py(_shock(rows)),
    ]
    if exp in JOHANSEN_ONLY:
        cells += [
            _md(
                f"## {n + 2}. Solve check, and the shock with Johansen\n\n"
                + _JOHANSEN_MD.format(exp=exp)
                + _WALL_MD
            ),
            _py(_solve_johansen("m")),
            _md(f"## {n + 3}. Every level against the book's Johansen result"),
            _py(_compare_johansen(exp, "m")),
        ]
    else:
        cells += [
            _md(f"## {n + 2}. Solve check and shock"),
            _py(SOLVE),
            _md(f"## {n + 3}. Every level against GEMPACK"),
            _py(_compare(exp, checked)),
        ]
    if exp in JOHANSEN and exp not in JOHANSEN_ONLY:
        cells += [
            _md(
                f"## {n + 4}. The book's own result: Johansen\n\n"
                + _JOHANSEN_MD.format(exp=exp)
                + " The same exercise, on a fresh model:"
            ),
            _py(
                'mj, _ = build_block_model(p, p.sets, gc, "ROW", base_calibrated=False, '
                "ref_gdx=None)\napply_shock(mj, SHOCK)\n" + _solve_johansen("mj")
            ),
            _py(_compare_johansen(exp, "mj")),
        ]
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 4,
    }


def exercises() -> list[str]:
    """The exercises that get a notebook: those with a GEMPACK solution to compare."""
    names = [e for e in (*rb.EXERCISES, *HOOKED) if e not in JOHANSEN_ONLY]
    names = [
        e
        for e in names
        if find_reference("nus333", "gemsim", experiment=e.lower(), method="gragg")
    ]
    return names + [
        e
        for e in JOHANSEN_ONLY
        if find_reference("nus333", "gempack", experiment=e.lower(), method="book")
    ]


def main() -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--out", type=Path, default=HERE, help="where to write them")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    for exp in exercises():
        text = json.dumps(notebook(exp), indent=1, ensure_ascii=False) + "\n"
        (a.out / f"{exp.lower()}.ipynb").write_text(
            text, encoding="utf-8", newline="\n"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
