"""Parity of the family route against the saved Pyomo run, one dataset.

usage: parity_ladder.py DATASET [--out FILE] [--reference DIR]
                        [--ifsub {0,1}] [--savf {capFix,capFlex}] [--mode gtap]

Runs the requested closure, (1, capFix) by default, with
``_solve_shock(DATASET, ifsub, savf, solve_nlp=True, backend="family")`` and
compares the three periods with ``<reference>/final.pkl`` (default
``~/.cache/equilibria/family/reference/<ds>/<closure_tag>``, ``tol_today`` for ifsub=1
capFix). Refuses a reference whose S files carry another ``closure`` (none = ifsub=1
capFix gtap).

Gate (exit 1 when it fails): every Var cell within ``|a - b| <= 1e-9 * max(1, |b|)``
(NaN equals NaN), identical fixed flags and upper bounds, lower bounds within 1e-9
relative like the values (the relative floors are 1e-3 x the prior level, so they
inherit its ~1e-10 difference), same
components and keys, code 1 on every solve. Newton iteration counts are reported
(ours vs the reference S files), never a gate.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import resource
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RTOL = 1e-9
LB_RTOL = RTOL
WORST = 20
# (S file number, period): settle check, check, shock
S_CALLS = ((2, "check"), (4, "check"), (5, "shock"))


def _closure_tag(ifsub: int, savf: str, mode: str) -> str:
    path = ROOT / "scripts" / "family" / "capture_prep.py"
    spec = importlib.util.spec_from_file_location("capture_prep", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.closure_tag(ifsub, savf, mode)


def reference_dir(
    dataset: str, ifsub: int = 1, savf: str = "capFix", mode: str = "gtap"
) -> Path:
    return (
        Path.home()
        / ".cache/equilibria/family/reference"
        / dataset
        / _closure_tag(ifsub, savf, mode)
    )


def reference_closure(ref: Path) -> dict:
    """Closure of the S files of ``ref``; a file with no ``closure`` field is read as
    ifsub=1 capFix gtap (the references captured before the field existed)."""
    default = {"ifsub": 1, "savf": "capFix", "mode": "gtap", "linsolve": None}
    found = []
    for f in sorted(ref.glob("*_S_*.json")):
        s = json.loads(f.read_text())
        c = {**default, **s.get("closure", {})}
        if c["linsolve"] is None:
            # old references carry only the top-level field of capture_solve
            c["linsolve"] = s.get("linsolve")
        found.append(c)
    for c in found[1:]:
        if {k: c[k] for k in ("ifsub", "savf", "mode")} != {
            k: found[0][k] for k in ("ifsub", "savf", "mode")
        }:
            raise ValueError(f"{ref}: S files disagree on the closure")
    return found[0] if found else default


def _same(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Equal, or both NaN."""
    with np.errstate(invalid="ignore"):
        return (a == b) | (np.isnan(a) & np.isnan(b))


def compare(mp, final) -> dict:
    """Cell by cell: ``mp`` (``MPArrays``) against a photo (``final.pkl``)."""
    per_component: dict[str, float] = {}
    worst: list[tuple] = []
    worst_lb: list[tuple] = []
    over = fixed_diff = ub_diff = lb_diff = cells = 0
    lb_max_rel = 0.0
    structure: list[str] = []
    for name in sorted(set(final.vars) - set(mp.vars)):
        structure.append(f"component {name} missing")
    for name in sorted(set(mp.vars) - set(final.vars)):
        structure.append(f"extra component {name}")
    for name, c in final.vars.items():
        v = mp.vars.get(name)
        if v is None:
            continue
        if set(v.keys) != set(c.keys) or len(v.keys) != len(c.keys):
            structure.append(f"{name}: keys differ")
            continue
        o = np.array([v.pos(k) for k in c.keys], dtype=int)
        a, b = v.value[o], np.asarray(c.value)
        cells += len(o)
        with np.errstate(invalid="ignore"):
            rel = np.abs(a - b) / np.maximum(1.0, np.abs(b))
        rel = np.where(_same(a, b), 0.0, np.where(np.isnan(rel), np.inf, rel))
        per_component[name] = float(rel.max()) if len(rel) else 0.0
        bad = rel > RTOL
        over += int(bad.sum())
        fixed_diff += int((v.fixed[o] != np.asarray(c.fixed)).sum())
        ub_diff += int((~_same(v.ub[o], np.asarray(c.ub))).sum())
        la, lbb = v.lb[o], np.asarray(c.lb)
        with np.errstate(invalid="ignore"):
            lrel = np.abs(la - lbb) / np.maximum(np.abs(lbb), 1e-300)
        lrel = np.where(_same(la, lbb), 0.0, np.where(np.isnan(lrel), np.inf, lrel))
        lb_diff += int((lrel > LB_RTOL).sum())
        if len(lrel):
            lb_max_rel = max(lb_max_rel, float(lrel.max()))
        for i in np.argsort(-lrel)[:WORST]:
            if lrel[i] > LB_RTOL:
                worst_lb.append(
                    (float(lrel[i]), name, c.keys[i], float(la[i]), float(lbb[i]))
                )
        # keep the WORST biggest per component, trimmed to WORST overall at the end
        for i in np.argsort(-rel)[:WORST]:
            if rel[i] > 0:
                worst.append((float(rel[i]), name, c.keys[i], float(a[i]), float(b[i])))
    worst = sorted(worst, key=lambda t: -t[0])[:WORST]
    worst_lb = sorted(worst_lb, key=lambda t: -t[0])[:WORST]
    return {
        "cells": cells,
        "over_1e9": over,
        "fixed_diff": fixed_diff,
        "ub_diff": ub_diff,
        "lb_diff": lb_diff,
        "lb_max_rel": lb_max_rel,
        "structure": structure,
        "max_rel": max(per_component.values(), default=0.0),
        "max_rel_by_component": dict(
            sorted(per_component.items(), key=lambda kv: -kv[1])
        ),
        "worst_cells": [
            {"rel": r, "var": n, "key": list(k), "ours": a, "reference": b}
            for r, n, k, a, b in worst
        ],
        "worst_lb_cells": [
            {"rel": r, "var": n, "key": list(k), "ours": a, "reference": b}
            for r, n, k, a, b in worst_lb
        ],
    }


def reference_nit(ref: Path) -> dict:
    out = {}
    for n, period in S_CALLS:
        f = ref / f"{n:03d}_S_{period}.json"
        if f.exists():
            s = json.loads(f.read_text())
            out[f"{n:03d}_{period}"] = {"nit": s.get("nit"), "code": s.get("code")}
    return out


def report(res, code, final, ref: Path, dataset: str) -> dict:
    out = compare(res.mp, final)
    codes = {k: v["code"] for k, v in res.results.items()}
    codes["settle_check"] = res.settle["check"]["code"]
    nit_ours = {k: v.get("nit") for k, v in res.results.items()}
    nit_ours["settle_check"] = res.settle["check"]["nit"]
    ref_nit = reference_nit(ref)
    gate = (
        out["over_1e9"] == 0
        and out["fixed_diff"] == 0
        and out["ub_diff"] == 0
        and out["lb_diff"] == 0
        and not out["structure"]
        and code == 1
        and all(c == 1 for k, c in codes.items() if k != "base")
        and all(r["code"] == 1 for r in ref_nit.values())
    )
    return {
        "dataset": dataset,
        "gate_passed": bool(gate),
        "code": code,
        "codes": codes,
        "nit_ours": nit_ours,
        "nit_reference": ref_nit,
        **out,
        "timing": getattr(res, "timing", {}),
    }


def _git_sha() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short=8", "HEAD"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except Exception:
        return "unknown"


def _newton_mumps() -> dict:
    path = ROOT / "scripts" / "gtap" / "perf_budget_20x41.py"
    spec = importlib.util.spec_from_file_location("perf_budget_20x41", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.NEWTON_MUMPS


def peak_rss_gb() -> float:
    ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return ru / (1e9 if sys.platform == "darwin" else 1e6)


def check_linsolve(ref: Path, have: dict, ours: str | None) -> None:
    """The Newton stopping point depends on the linear solver: refuse a reference
    captured with another one (an old reference, linsolve unknown, is allowed)."""
    theirs = have.get("linsolve")
    for f in sorted(ref.glob("*_S_*.json")):
        # the top-level field is what the solver actually used
        top = json.loads(f.read_text()).get("linsolve")
        if top is not None:
            theirs = top
            break
    if theirs is not None and ours is not None and theirs != ours:
        raise ValueError(
            f"{ref}: reference linsolve={theirs} differs from this run's linsolve={ours}"
        )


def run(
    dataset: str,
    reference: Path | None = None,
    *,
    ifsub: int = 1,
    savf: str = "capFix",
    mode: str = "gtap",
) -> dict:
    """Run the family route on ``dataset`` and compare with the saved run."""
    if mode != "gtap":
        raise NotImplementedError(f"mode={mode!r}: only 'gtap' until Phase 3")
    ref = reference or reference_dir(dataset, ifsub, savf, mode)
    have = reference_closure(ref)
    if (have["ifsub"], have["savf"], have["mode"]) != (ifsub, savf, mode):
        raise ValueError(
            f"{ref}: reference closure ifsub={have['ifsub']} savf={have['savf']} "
            f"mode={have['mode']} differs from the requested ifsub={ifsub} "
            f"savf={savf} mode={mode}"
        )
    from equilibria.family.prep_capture import load

    for k, v in _newton_mumps().items():
        os.environ.setdefault(k, v)
    check_linsolve(ref, have, os.environ.get("EQUILIBRIA_GTAP_TR_LINSOLVE"))
    sys.path[:0] = [str(ROOT), str(ROOT / "src")]
    from tests.templates.gtap.test_gtap7_gempack_parity import _solve_shock

    t = time.perf_counter()
    res, code = _solve_shock(dataset, ifsub, savf, solve_nlp=True, backend="family")
    wall = time.perf_counter() - t
    out = report(res, code, load(ref / "final.pkl"), ref, dataset)
    out.update(
        wall_s=wall,
        wall_min=wall / 60,
        peak_rss_gb=peak_rss_gb(),
        sha=_git_sha(),
        linsolve=os.environ.get("EQUILIBRIA_GTAP_TR_LINSOLVE"),
    )
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("dataset")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--reference", type=Path)
    ap.add_argument("--ifsub", type=int, choices=(0, 1), default=1)
    ap.add_argument("--savf", choices=("capFix", "capFlex"), default="capFix")
    ap.add_argument("--mode", choices=("gtap", "altertax"), default="gtap")
    args = ap.parse_args(argv)
    out = run(
        args.dataset, args.reference, ifsub=args.ifsub, savf=args.savf, mode=args.mode
    )
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(out, indent=1, default=str))
    top = list(out["max_rel_by_component"].items())[:5]
    print(
        f"{out['dataset']} gate={'PASS' if out['gate_passed'] else 'FAIL'} "
        f"cells={out['cells']} over_1e-9={out['over_1e9']} max_rel={out['max_rel']:.3g} "
        f"fixed={out['fixed_diff']} ub={out['ub_diff']} lb={out['lb_diff']} "
        f"codes={out['codes']} nit={out['nit_ours']} ref_nit={out['nit_reference']} "
        f"wall_min={out['wall_min']:.2f} peak_gb={out['peak_rss_gb']:.2f} top={top}"
    )
    for w in out["worst_lb_cells"][:5]:
        print(f"  worst lb {w['var']}{tuple(w['key'])} rel={w['rel']:.3g}")
    return 0 if out["gate_passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
