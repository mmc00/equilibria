"""One textbook exercise whose closure is built in a notebook, against GEMPACK or GAMS.

TBL65B, TBL94, ME3C and ME9A change the closure with ``@overwrite`` in their
notebooks, so compare_burfisher_gempack.py and run_burfisher.py (Tables A2 and
A3) leave them out. This script measures one of them per process: the
``@overwrite`` of one exercise leaks into the next one solved in the same
process (TBL94 gave 64% within 0.1% after TBL65B, 100% alone).

    gempack EXP --json OUT   Table A2 row: runs the notebook
                             examples/notebooks/gtap7/burfisher/<exp>.ipynb as is,
                             except its final ``assert ... within 0.5%`` (ME9A is
                             above it), and reports its comparison against the
                             GEMPACK Gragg solution.
    gams EXP --gams-dir DIR --json OUT
                             Table A3 row: solves the exercise with the fixture of
                             its test and compares against DIR/<EXP>_capFlex.gdx
                             with run_burfisher.compare.

Usage (from the repository root):
    PYTHONPATH=src:. python scripts/paper/notebook_exercises.py gempack TBL94 --json out.json
    PYTHONPATH=src:. python scripts/paper/notebook_exercises.py gams TBL94 \\
        --gams-dir paper_out/gams_burfisher --json out.json
"""

from __future__ import annotations

import argparse
import importlib
import json
import re
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
NOTEBOOKS = ROOT / "examples" / "notebooks" / "gtap7" / "burfisher"
EXERCISES = ["TBL65B", "TBL94", "ME3C", "ME9A"]
# The test whose `solved` fixture builds each exercise's closure and solves it.
FIXTURES = {
    "TBL65B": "tests.templates.gtap.test_nus333_tbl65b_unemployment",
    "ME3C": "tests.templates.gtap.test_nus333_tbl65b_unemployment",
    "TBL94": "tests.templates.gtap.test_nus333_tbl94_regulacion",
    "ME9A": "tests.templates.gtap.test_nus333_me9a_linea_base",
}


def against_gempack(exp: str) -> dict:
    cells = json.loads((NOTEBOOKS / f"{exp.lower()}.ipynb").read_text())["cells"]
    src = "\n\n".join("".join(c["source"]) for c in cells if c["cell_type"] == "code")
    src = re.sub(r"^assert o\[\"max\"\].*$", "pass", src, flags=re.M)
    ns: dict = {"__name__": f"nb_{exp}"}
    exec(compile(src, f"{exp.lower()}.ipynb", "exec"), ns)
    r = ns.get("report")
    if r is None:
        raise SystemExit(f"{exp}: the notebook made no comparison against GEMPACK")
    o = r.overall
    w = o["pct_within"]
    worst = max(r.cells, key=lambda c: c.d)
    per = {
        p: max((c.d for c in r.cells if c.period == p), default=float("nan"))
        for p in ("pre", "post")
    }
    row = {
        "cells": o["cells"], "le0.1": w[1e-3], "le0.5": w[5e-3], "le1": w[1e-2],
        "max_pct": 100 * worst.d, "max_var": worst.variable, "max_period": worst.period,
        "pre_max_pct": 100 * per["pre"], "post_max_pct": 100 * per["post"],
        "unmatched": len(r.unmatched),
    }  # fmt: skip
    print(
        f"A2ROW {exp} cells={row['cells']} le0.1={row['le0.1']:.2f} "
        f"le0.5={row['le0.5']:.2f} le1={row['le1']:.2f} max={row['max_pct']:.3f}% "
        f"var={row['max_var']} period={row['max_period']} "
        f"pre_max={row['pre_max_pct']:.3f}% post_max={row['post_max_pct']:.3f}% "
        f"unmatched={row['unmatched']}",
        flush=True,
    )
    return row


def against_gams(exp: str, gams_dir: Path) -> dict:
    sys.path.insert(0, str(ROOT / "scripts" / "gtap"))
    import run_burfisher as rb

    mod = importlib.import_module(FIXTURES[exp])
    fx = getattr(mod.solved, "__wrapped__", None) or mod.solved._get_wrapped_function()
    takes_request = "request" in fx.__code__.co_varnames[: fx.__code__.co_argcount]
    gen = fx(types.SimpleNamespace(param=exp)) if takes_request else fx()
    y = next(gen)
    m, p, res = (y[1], y[2], y[3]) if isinstance(y[0], str) else (y[0], y[1], y[2])
    codes = {k: int(v["code"]) for k, v in res.items()}
    row = {"codes": codes, **rb.compare(m, p, gams_dir / f"{exp}_capFlex.gdx")}
    c, s = row["check"], row["shock"]
    print(
        f"A3ROW {exp} codes={codes} check {c['cells']} {c['match_pct']['0.1%']} "
        f"shock {s['cells']} {s['match_pct']['0.1%']} "
        f"instr {s['instr_cells']} {len(s['instr_bad'])} {s['worst'][:3]}",
        flush=True,
    )
    try:
        next(gen)
    except StopIteration:
        pass
    return row


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("against", choices=["gempack", "gams"])
    ap.add_argument("exercise", choices=EXERCISES)
    ap.add_argument("--gams-dir", type=Path, help="GAMS references (gams only)")
    ap.add_argument("--json", type=Path, required=True, help="where to write the row")
    a = ap.parse_args(argv)
    if a.against == "gams":
        if a.gams_dir is None:
            ap.error("gams needs --gams-dir")
        row = against_gams(a.exercise, a.gams_dir)
    else:
        row = against_gempack(a.exercise)
    a.json.write_text(json.dumps({a.exercise: row}, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
