"""Parity of the family route against the saved Pyomo run, one dataset.

usage: parity_ladder.py DATASET [--out FILE] [--reference DIR]
                        [--ifsub {0,1}] [--savf {capFix,capFlex}] [--mode gtap]

Runs the requested closure, (1, capFix) by default, with
``_solve_shock(DATASET, ifsub, savf, solve_nlp=True, backend="family")`` and
compares the three periods with ``<reference>/final.pkl`` (default
``~/.cache/equilibria/family/reference/<ds>/<closure_tag>``, ``tol_today`` for ifsub=1
capFix). Refuses a reference whose S files carry another ``closure`` (none = ifsub=1
capFix gtap).

Gate (exit 1 when it fails; user answer 8, 2026-11-07, the polished-root gate). For each
solved call of the main loop (check, shock):

(i) both raw end points, ours and the reference's (``final.pkl``), pass the residual
    test of the run on the family system: ``||F/fscale||_inf < EQUILIBRIA_GTAP_TR_RELTOL``
    (when > 0) or ``||F||_inf < EQUILIBRIA_GTAP_TR_FTOL``, as ``newton_tr`` stops;
(ii) one exact, unregularized Newton step from each (the same linear solve for both:
    MUMPS with the values at analysis and no GMIN under ``linsolve=mumps``, else scipy
    ``splu``; the result clipped to the bounds) gives two "polished" points.

Then every Var cell of the polished photos is within ``|a - b| <= 1e-9 * max(1, |b|)``
(NaN equals NaN), with identical fixed flags and upper bounds, lower bounds within 1e-9
relative like the values (the relative floors are 1e-3 x the prior level, so they inherit
its ~1e-10 difference), same components and keys, and code 1 on every solve. Only the free
columns of a solved call are polished, and what the family post-solve derives from them
(the ifSUB report variables pfa / pfy / pm / ..., the welfare tail): the same post-solve
runs on two copies, from our polished point and from the reference's (whose other cells
are ours, identical to it within the gate). Fixed exogenous cells, the base period
(calibrated, not solved) and every other cell keep their raw values. The settle
(``calibrate_base``'s check, which seeds the base) is not polished: ``settled.pkl`` omits
the variables indexed by the period alone, so it has no full reference end point; the
base cells it produces are compared raw.

The raw comparison is still made and reported: ``max_rel``, ``worst_cells``,
``max_rel_by_component``, ``over_1e9`` keep their raw meaning (as ``raw_max_rel`` /
``raw_worst_cells``); ``polished_*`` are the gate's; ``periods`` holds per period the raw
and polished max rel and, per solved call, ``residual_ours`` / ``residual_reference``
(``||F||_inf`` at the raw end points). Newton iteration counts are reported (ours vs the
reference S files), never a gate.
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
PERIODS = ("base", "check", "shock")
# the driver's Newton calls in order: the settle's check, then the main loop's
CALL_TAGS = ("settle_check", "check", "shock")


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
    per_period: dict[str, float] = {}
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
        if len(rel):
            periods = np.array([k[-1] if k else "" for k in c.keys], dtype=object)
            for p in PERIODS:
                m = periods == p
                if m.any():
                    per_period[p] = max(per_period.get(p, 0.0), float(rel[m].max()))
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
        "max_rel_by_period": per_period,
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


def _exact_step(J, F, linsolve: str) -> np.ndarray:
    """``-J^-1 F`` without regularization: MUMPS (values at analysis, ICNTL 6=7, 8=8, no
    GMIN) under ``linsolve="mumps"``, else scipy ``splu``."""
    import scipy.sparse as sp

    A = sp.csr_matrix(J)
    if linsolve == "mumps":
        import mumps

        ctx = mumps.DMumpsContext(par=1, sym=0)
        try:
            ctx.set_silent()
            for k, v in ((14, 100), (6, 7), (8, 8)):
                ctx.set_icntl(k, v)
            ctx.set_centralized_sparse(A.tocoo())
            ctx.run(job=4)  # analysis with the values, then the factor
            d = -np.asarray(F, dtype=float).copy()
            ctx.set_rhs(d)
            ctx.run(job=3)
            return d
        finally:
            ctx.destroy()
    from scipy.sparse.linalg import splu

    return splu(A.tocsc()).solve(-np.asarray(F, dtype=float))


def _rel(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    with np.errstate(invalid="ignore"):
        r = np.abs(a - b) / np.maximum(1.0, np.abs(b))
    return np.where(_same(a, b), 0.0, np.where(np.isnan(r), np.inf, r))


def polish_call(
    F,
    J,
    fscale,
    x_ours: np.ndarray,
    x_ref: np.ndarray,
    lb: np.ndarray,
    ub: np.ndarray,
    *,
    reltol: float,
    ftol: float,
    linsolve: str,
) -> dict:
    """One solved call: the residual test at both raw end points and one exact Newton
    step from each. ``passed`` iff both pass the test and the polished points agree
    within ``RTOL``; ``x_ours_polished`` / ``x_ref_polished`` are the polished points."""
    out: dict = {}
    pts = {}
    for tag, x in (("ours", x_ours), ("reference", x_ref)):
        x = np.clip(x, lb, ub)
        f = np.asarray(F(x), dtype=float)
        rinf = float(np.abs(f).max()) if len(f) else 0.0
        rrel = (
            float(np.abs(f / fscale).max())
            if reltol > 0 and fscale is not None and len(f)
            else None
        )
        out[f"residual_{tag}"] = rinf
        out[f"residual_scaled_{tag}"] = rrel
        out[f"residual_ok_{tag}"] = bool(
            rinf < ftol or (rrel is not None and rrel < reltol)
        )
        pts[tag] = np.clip(x + _exact_step(J(x), f, linsolve), lb, ub)
    raw = _rel(np.clip(x_ours, lb, ub), np.clip(x_ref, lb, ub))
    pol = _rel(pts["ours"], pts["reference"])
    out["raw_max_rel"] = float(raw.max()) if len(raw) else 0.0
    out["polished_max_rel"] = float(pol.max()) if len(pol) else 0.0
    out["passed"] = bool(
        out["residual_ok_ours"]
        and out["residual_ok_reference"]
        and out["polished_max_rel"] <= RTOL
    )
    out["x_ours_polished"] = pts["ours"]
    out["x_ref_polished"] = pts["reference"]
    return out


def polished_gate(
    *,
    raw_over: int,
    polished_over: int,
    fixed_diff: int,
    ub_diff: int,
    lb_diff: int,
    structure: list,
    codes_ok: bool,
    calls: dict,
) -> bool:
    """User answer 8: the polished photos within RTOL, fixed/ub/lb and structure equal,
    code 1, and every solved call polished and passed. ``raw_over`` is reported only."""
    del raw_over  # reported, never a gate
    return bool(
        calls
        and all(c.get("passed") for c in calls.values())
        and polished_over == 0
        and fixed_diff == 0
        and ub_diff == 0
        and lb_diff == 0
        and not structure
        and codes_ok
    )


def _split(col: str) -> tuple[str, tuple]:
    name, _, rest = col.partition("[")
    return name, tuple(rest[:-1].split(",")) if rest else ()


def _ref_values(final, cols: list[str]) -> np.ndarray:
    """The reference's value of each squared-system column (names as ``name[a,b,c]``)."""
    pos: dict[str, dict] = {}
    out = np.empty(len(cols))
    for j, col in enumerate(cols):
        name, key = _split(col)
        c = final.vars[name]
        if name not in pos:
            pos[name] = {tuple(map(str, k)): i for i, k in enumerate(c.keys)}
        out[j] = c.value[pos[name][key]]
    return out


class PolishRecorder:
    """Wraps ``family.driver._newton``: per Newton call of the main loop (check, shock),
    ``polish_call`` at our end point and the reference's, on the same family system."""

    def __init__(self, final, linsolve: str) -> None:
        self.final = final
        self.linsolve = linsolve
        self.n = 0
        self.calls: dict[str, dict] = {}
        self.columns: dict[str, list[str]] = {}
        # per polished call: (var, keys, ours, reference) of every cell the post-solve
        # rewrites from the solution (the columns and the report / welfare cells)
        self.cells: dict[str, list[tuple]] = {}
        self._pending: str | None = None

    def wrap(self, newton):
        def _newton(ss, sq):
            res, resid, code = newton(ss, sq)
            tag = CALL_TAGS[self.n] if self.n < len(CALL_TAGS) else f"call{self.n}"
            self.n += 1
            if tag == "settle_check":
                return res, resid, code
            cols = list(sq.col_names)
            lb = np.array([sq.cells[c][2] for c in cols], dtype=float)
            ub = np.array([sq.cells[c][3] for c in cols], dtype=float)
            lb[~np.isfinite(lb)] = -1e20  # as driver._newton
            ub[~np.isfinite(ub)] = 1e20
            reltol = float(os.environ.get("EQUILIBRIA_GTAP_TR_RELTOL", "0") or 0.0)
            ftol = float(os.environ.get("EQUILIBRIA_GTAP_TR_FTOL", "1e-7"))

            def F(z):
                return ss.F(np.clip(z, lb, ub))

            def J(z):
                return ss.J(np.clip(z, lb, ub))

            self.calls[tag] = polish_call(
                F,
                J,
                ss.fscale() if reltol > 0 else None,
                np.asarray(res.x, dtype=float),
                _ref_values(self.final, cols),
                lb,
                ub,
                reltol=reltol,
                ftol=ftol,
                linsolve=self.linsolve,
            )
            self.columns[tag] = cols
            self._pending = tag
            return res, resid, code

        return _newton

    def wrap_post_solve(self, post_solve):
        """After the real post-solve of a polished call, the same post-solve on two
        copies, from our polished point and from the reference's: the report variables
        (pfa, pfy, pm, ...) and the welfare tail are derived from the solution."""

        def _post(mp, period, **kw):
            tag, self._pending = self._pending, None
            if tag is None or kw.get("x") is None:
                return post_solve(mp, period, **kw)
            pre = mp.copy()
            out = post_solve(mp, period, **kw)
            info = self.calls[tag]
            a = post_solve(pre.copy(), period, **{**kw, "x": info["x_ours_polished"]})
            b = post_solve(pre.copy(), period, **{**kw, "x": info["x_ref_polished"]})
            cells = []
            for name, v in out.vars.items():
                va, vb, v0 = (
                    a.vars[name].value,
                    b.vars[name].value,
                    pre.vars[name].value,
                )
                with np.errstate(invalid="ignore"):
                    m = (v.period == period) & (
                        ~_same(v.value, v0) | ~_same(va, v0) | ~_same(vb, v0)
                    )
                idx = np.flatnonzero(m)
                if len(idx):
                    keys = [v.keys[i] for i in idx.tolist()]
                    cells.append((name, keys, va[idx].copy(), vb[idx].copy()))
            self.cells[tag] = cells
            return out

        return _post


def _with_values(vars_: dict, cols: list[str], values: np.ndarray) -> dict:
    """``vars_`` with the cells named ``cols`` set to ``values`` (changed vars copied)."""
    import copy

    out = dict(vars_)
    pos: dict[str, dict] = {}
    for col, x in zip(cols, values, strict=True):
        name, key = _split(col)
        if name not in pos:
            v = copy.copy(vars_[name])
            v.value = np.array(v.value, dtype=float, copy=True)
            out[name] = v
            pos[name] = {tuple(map(str, k)): i for i, k in enumerate(v.keys)}
        out[name].value[pos[name][key]] = x
    return out


def _with_cells(vars_: dict, cells: list[tuple], side: int) -> dict:
    """``vars_`` with the recorded post-solve cells at their polished values (``side``
    2 = ours, 3 = the reference's)."""
    import copy

    out = dict(vars_)
    for row in cells:
        name, keys, vals = row[0], row[1], row[side]
        if name not in vars_:
            continue
        v = copy.copy(out[name])
        v.value = np.array(v.value, dtype=float, copy=True)
        pos = {tuple(map(str, k)): i for i, k in enumerate(v.keys)}
        for k, x in zip(keys, vals, strict=True):
            i = pos.get(tuple(map(str, k)))
            if i is not None:
                v.value[i] = x
        out[name] = v
    return out


def polished_photos(mp, final, rec: PolishRecorder):
    """``(mp, final)`` with each solved call's cells at their polished values: the
    columns and what the post-solve derives from them (``rec.cells``), else the
    columns alone."""
    import types

    ours, theirs = dict(mp.vars), dict(final.vars)
    for tag, info in rec.calls.items():
        if tag in rec.cells:
            ours = _with_cells(ours, rec.cells[tag], 2)
            theirs = _with_cells(theirs, rec.cells[tag], 3)
        else:
            cols = rec.columns[tag]
            ours = _with_values(ours, cols, info["x_ours_polished"])
            theirs = _with_values(theirs, cols, info["x_ref_polished"])
    return types.SimpleNamespace(vars=ours), types.SimpleNamespace(vars=theirs)


def reference_nit(ref: Path) -> dict:
    out = {}
    for n, period in S_CALLS:
        f = ref / f"{n:03d}_S_{period}.json"
        if f.exists():
            s = json.loads(f.read_text())
            out[f"{n:03d}_{period}"] = {"nit": s.get("nit"), "code": s.get("code")}
    return out


def report(
    res, code, final, ref: Path, dataset: str, rec: PolishRecorder | None = None
) -> dict:
    out = compare(res.mp, final)
    codes = {k: v["code"] for k, v in res.results.items()}
    codes["settle_check"] = res.settle["check"]["code"]
    nit_ours = {k: v.get("nit") for k, v in res.results.items()}
    nit_ours["settle_check"] = res.settle["check"]["nit"]
    ref_nit = reference_nit(ref)
    codes_ok = (
        code == 1
        and all(c == 1 for k, c in codes.items() if k != "base")
        and all(r["code"] == 1 for r in ref_nit.values())
    )
    calls = rec.calls if rec is not None else {}
    pol = compare(*polished_photos(res.mp, final, rec)) if rec is not None else out
    gate = polished_gate(
        raw_over=out["over_1e9"],
        polished_over=pol["over_1e9"],
        fixed_diff=pol["fixed_diff"],
        ub_diff=pol["ub_diff"],
        lb_diff=pol["lb_diff"],
        structure=pol["structure"],
        codes_ok=codes_ok,
        calls=calls,
    )
    periods: dict[str, dict] = {}
    for p in PERIODS:
        periods[p] = {
            "raw_max_rel": out["max_rel_by_period"].get(p, 0.0),
            "polished_max_rel": pol["max_rel_by_period"].get(p, 0.0),
        }
        info = calls.get(p)
        if info is not None:
            periods[p].update(
                {
                    k: info[k]
                    for k in (
                        "residual_ours",
                        "residual_reference",
                        "residual_scaled_ours",
                        "residual_scaled_reference",
                        "residual_ok_ours",
                        "residual_ok_reference",
                        "passed",
                    )
                }
            )
            periods[p]["call_raw_max_rel"] = info["raw_max_rel"]
            periods[p]["call_polished_max_rel"] = info["polished_max_rel"]
        else:
            periods[p]["polished"] = False  # not solved here: compared raw
    return {
        "dataset": dataset,
        "gate_passed": bool(gate),
        "gate": "polished-root (user answer 8)",
        "code": code,
        "codes": codes,
        "nit_ours": nit_ours,
        "nit_reference": ref_nit,
        **out,  # raw: max_rel, worst_cells, over_1e9, ... keep their raw meaning
        "raw_max_rel": out["max_rel"],
        "raw_worst_cells": out["worst_cells"],
        "polished_max_rel": pol["max_rel"],
        "polished_worst_cells": pol["worst_cells"],
        "polished_over_1e9": pol["over_1e9"],
        "periods": periods,
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

    import equilibria.family.post_solve as post_mod
    from equilibria.family import driver

    final = load(ref / "final.pkl")
    linsolve = (os.environ.get("EQUILIBRIA_GTAP_TR_LINSOLVE") or "direct").lower()
    rec = PolishRecorder(final, linsolve)
    newton, post = driver._newton, post_mod.post_solve
    driver._newton = rec.wrap(newton)
    post_mod.post_solve = rec.wrap_post_solve(post)
    try:
        t = time.perf_counter()
        res, code = _solve_shock(dataset, ifsub, savf, solve_nlp=True, backend="family")
        wall = time.perf_counter() - t
    finally:
        driver._newton = newton
        post_mod.post_solve = post
    out = report(res, code, final, ref, dataset, rec)
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
        f"cells={out['cells']} raw_over_1e-9={out['over_1e9']} "
        f"raw_max_rel={out['max_rel']:.3g} polished_max_rel={out['polished_max_rel']:.3g} "
        f"polished_over_1e-9={out['polished_over_1e9']} "
        f"fixed={out['fixed_diff']} ub={out['ub_diff']} lb={out['lb_diff']} "
        f"codes={out['codes']} nit={out['nit_ours']} ref_nit={out['nit_reference']} "
        f"wall_min={out['wall_min']:.2f} peak_gb={out['peak_rss_gb']:.2f} top={top}"
    )
    for p, v in out["periods"].items():
        print(f"  {p}: {v}")
    for w in out["worst_lb_cells"][:5]:
        print(f"  worst lb {w['var']}{tuple(w['key'])} rel={w['rel']:.3g}")
    return 0 if out["gate_passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
