"""Measure the ALTERTAX array route on nus333 against GEMPACK (``sl4levels_altax.har``).

Three equilibria runs of the same experiment (``altax.exp``: ``to("AGR","AGR","USA")``):

- ``s1.01``: ``altertax.prm`` as loaded (production nests at 1.01 by #192; ``esubd``,
  ``esubm`` and ``sigmas`` equal to 1 also at 1.01, see ``run.UNIT_TRADE_ELASTICITIES``);
- ``s1.02-prod``: ``sigmap``/``sigmand``/``sigmav`` forced to 1.02, the rest as above;
- ``s1.02-all``: all of them at 1.02.

Plus ``s1.01+esubc`` (esubc filled from ESBC) and ``diag-esubi0`` (investment as Leontief,
a diagnostic that is not the route's setting).

Each is compared to GEMPACK with ``compare_gempack_levels.run`` (levels, the check as the
pre and the shock as the post), and the runs are compared to each other (the effect of
sigma on the raw ``EQ_VARS`` levels). Also whether ``esubc`` (empty after the load while
``altertax.prm`` has ``ESBC``) changes any level once filled.

Run with the MUMPS Python (one process, ~1 min)::

    PYTHONPATH=<worktree>/src:<worktree> ~/proyectos/Pedro/.venv/bin/python \\
        scripts/family/altertax_measure.py [--out DIR]

Writes ``family_altertax_nus333_<sha>.json`` into DIR (default
``~/proyectos/dev-tools/equilibria-tools/plans/mediciones``).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

DEFAULT_OUT = Path.home() / "proyectos/dev-tools/equilibria-tools/plans/mediciones"
FIXTURES = ROOT / "tests/fixtures/gtap7_gempack/burfisher_gragg"
HAR = FIXTURES / "sl4levels_altax.har"
UPDATED = FIXTURES / "altax.upd"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _set_env() -> None:
    cg = _load("perf_budget_20x41", ROOT / "scripts/gtap/perf_budget_20x41.py")
    for k, v in cg.NEWTON_MUMPS.items():
        os.environ.setdefault(k, v)
    os.environ["EQUILIBRIA_FAMILY_CACHE"] = tempfile.mkdtemp(prefix="altax_cache_")


def worst_by_variable(report, top: int = 10) -> dict:
    """``{variable: {cells, max, worst: [10 cells]}}`` of a ``compare`` Report."""
    by: dict[str, list] = {}
    for c in report.cells:
        by.setdefault(c.variable, []).append(c)
    out = {}
    for var, cs in sorted(by.items()):
        cs = sorted(cs, key=lambda c: (c.d != c.d, c.d), reverse=True)
        out[var] = {
            "cells": len(cs),
            "max": cs[0].d,
            "worst": [
                {"period": c.period, "index": list(c.index), "eq": c.a, "gempack": c.b, "d": c.d}
                for c in cs[:top]
            ],
        }
    return out


def level_diff(la: dict, lb: dict, period: str, top: int = 10) -> dict:
    """Per variable, ``|a - b| / max(1, |b|)`` over the cells of ``period``."""
    out = {}
    for var, cells in la.items():
        rows = []
        for key, a in cells.items():
            if key[-1] != period:
                continue
            b = lb[var].get(key)
            if b is None or a != a or b != b:
                continue
            rows.append((abs(a - b) / max(1.0, abs(b)), key, a, b))
        rows.sort(key=lambda r: r[0], reverse=True)
        if rows:
            out[var] = {
                "cells": len(rows),
                "max": rows[0][0],
                "worst": [
                    {"index": list(k[:-1]), "a": a, "b": b, "d": d}
                    for d, k, a, b in rows[:top]
                ],
            }
    return out


RESOLVABLE = 1e-8  # a move below this fraction of the level is below float32 noise


def move_errors(rows, top: int = 10) -> dict:
    """``rows`` = (variable, index, pre_a, post_a, pre_b, post_b): per variable the error
    of the move (post - pre) of A against that of B, ``|dA - dB| / |dB|`` over the cells
    where B's move is resolvable (``|dB| > RESOLVABLE * max(1, |pre_b|)``), plus the same
    error as a fraction of the level."""
    by: dict[str, list] = {}
    for var, idx, pa, qa, pb, qb in rows:
        da, db = qa - pa, qb - pb
        if not abs(db) > RESOLVABLE * max(1.0, abs(pb)):
            continue
        by.setdefault(var, []).append(
            (abs(da - db) / abs(db), abs(da - db) / max(1.0, abs(pb)), idx, da, db)
        )
    out = {}
    for var, rs in sorted(by.items()):
        rs.sort(key=lambda r: r[0], reverse=True)
        out[var] = {
            "cells": len(rs),
            "max_vs_move": rs[0][0],
            "median_vs_move": sorted(r[0] for r in rs)[len(rs) // 2],
            "max_vs_level": max(r[1] for r in rs),
            "worst": [
                {"index": list(i), "d_eq": da, "d_ref": db, "err_vs_move": e}
                for e, _, i, da, db in rs[:top]
            ],
        }
    return out


def moves_vs_gempack(report) -> dict:
    cells = {(c.variable, c.index, c.period): c for c in report.cells}
    rows = []
    for (var, idx, per), c in cells.items():
        if per != "pre":
            continue
        post = cells.get((var, idx, "post"))
        if post is not None:
            rows.append((var, idx, c.a, post.a, c.b, post.b))
    return move_errors(rows)


def moves_between(la: dict, lb: dict) -> dict:
    rows = []
    for var, cells in la.items():
        for key, a_chk in cells.items():
            if key[-1] != "check":
                continue
            idx = tuple(key[:-1])
            a_sh = cells.get((*idx, "shock"))
            b_chk, b_sh = lb[var].get(key), lb[var].get((*idx, "shock"))
            if None in (a_sh, b_chk, b_sh) or a_chk != a_chk or a_sh != a_sh:
                continue
            rows.append((var, idx, a_chk, a_sh, b_chk, b_sh))
    return move_errors(rows)


def _max(table: dict, key: str = "max") -> float:
    return max((v[key] for v in table.values()), default=0.0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    a = ap.parse_args()
    _set_env()

    from equilibria._local_refs import nus333_dir
    from equilibria.babel.har.reader import read_har
    from equilibria.blocks.gtap._derived_params import xscale_data
    from equilibria.family.altertax.levels import family_levels
    from equilibria.family.altertax.run import (
        load_altertax_params,
        set_production_sigma,
        solve_altertax,
    )

    cg = _load("compare_gempack_levels", ROOT / "scripts/gtap/compare_gempack_levels.py")
    d = nus333_dir()
    basedata = d / "basedata.har"
    sha = subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "--short=8", "HEAD"],
        capture_output=True,
        text=True,
    ).stdout.strip()

    runs: dict[str, dict] = {}

    def go(tag: str, params) -> None:
        run = solve_altertax(d, params=params)
        levels = family_levels(run.result.mp, cg.EQ_VARS)
        xscale = dict(xscale_data(run.params, run.params.sets))
        rep = cg.run(levels, xscale, HAR, basedata, UPDATED)
        runs[tag] = {
            "codes": run.codes,
            "minutes": run.minutes,
            "residual": {k: v.get("residual") for k, v in run.result.results.items()},
            "nit": {k: v.get("nit") for k, v in run.result.results.items()},
            "levels": levels,
            "vs_gempack": worst_by_variable(rep),
            "moves_vs_gempack": moves_vs_gempack(rep),
            "unmatched": len(rep.unmatched),
            "excluded": [
                {"var": e.variable, "where": e.where, "why": e.why, "cells": e.cells}
                for e in rep.exclusions
            ],
            "overall": {
                "cells": rep.overall["cells"],
                "max": rep.overall["max"],
                "max_variable": rep.overall["max_variable"],
                "pct_within": {repr(k): v for k, v in rep.overall["pct_within"].items()},
            },
        }
        print(f"{tag}: codes={run.codes} min={run.minutes:.2f} max_vs_gempack={rep.overall['max']:.3g}")

    go("s1.01", load_altertax_params(d))
    p = load_altertax_params(d)
    set_production_sigma(p, 1.02)
    go("s1.02-prod", p)
    p = load_altertax_params(d, unit_sigma=1.02)
    set_production_sigma(p, 1.02)
    go("s1.02-all", p)

    # esubc: empty after the load, ESBC in altertax.prm; fill it and compare levels
    p = load_altertax_params(d)
    esbc = read_har(d / "altertax.prm")["ESBC"]
    empty = not p.elasticities.esubc
    acts = [str(x) for x in p.sets.a]
    regs = [str(x) for x in p.sets.r]
    arr = esbc.array  # (ACTS, REG)
    for ia, act in enumerate(acts):
        for ir, reg in enumerate(regs):
            p.elasticities.esubc[(reg, act)] = float(arr[ia, ir])
    go("s1.01+esubc", p)
    esubc_change = level_diff(runs["s1.01"]["levels"], runs["s1.01+esubc"]["levels"], "shock")
    esubc_max = max(_max(esubc_change), _max(level_diff(runs["s1.01"]["levels"], runs["s1.01+esubc"]["levels"], "check")))

    # diagnostic: investment demand across commodities as Leontief (esubi = 0). Not the
    # route's setting: the file has ESBI = 1. Measured to see whether the worst cells
    # (qid, qim of AGR in USA) come from it.
    p = load_altertax_params(d)
    for key in p.elasticities.esubi:
        p.elasticities.esubi[key] = 0.0
    p.shares.calibrate(p.benchmark, p.elasticities, p.sets, p.taxes)
    p._calibrate()
    go("diag-esubi0", p)

    pairs = {}
    for x, y in (("s1.01", "s1.02-prod"), ("s1.01", "s1.02-all")):
        pairs[f"{x}_vs_{y}"] = {
            period: level_diff(runs[x]["levels"], runs[y]["levels"], period)
            for period in ("check", "shock")
        }
        pairs[f"{x}_vs_{y}"]["move"] = moves_between(runs[x]["levels"], runs[y]["levels"])

    for r in runs.values():
        del r["levels"]  # the JSON keeps the diffs, not 4 copies of the levels

    out = {
        "sha": sha,
        "har": str(HAR.relative_to(ROOT)),
        "updated": str(UPDATED.relative_to(ROOT)),
        "runs": runs,
        "sigma_effect": {
            k: {
                p_: {
                    "max": _max(t, "max_vs_move" if p_ == "move" else "max"),
                    "by_variable": t,
                }
                for p_, t in v.items()
            }
            for k, v in pairs.items()
        },
        "esubc": {
            "empty_after_load": empty,
            "ESBC_values": sorted({float(x) for x in arr.flatten()}),
            "max_level_change_when_filled": esubc_max,
            "changes_any_level_above_1e-12": esubc_max > 1e-12,
        },
    }
    a.out.mkdir(parents=True, exist_ok=True)
    path = a.out / f"family_altertax_nus333_{sha}.json"
    path.write_text(json.dumps(out, indent=1, default=float))
    print(path)


if __name__ == "__main__":
    main()
