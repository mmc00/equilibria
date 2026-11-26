"""Diff the MCP equation<->variable pairing: GAMS `model gtap /.../` vs Python.

THE BLIND SPOT this covers — the deepest one, invisible to ALL other cascade tools:
nl_compare (coefficients), diff_calibration (inputs) and diff_equation_form (forms)
all assume the same equation determines the same variable in both engines. But an
MCP is defined by its PAIRING — which equation is complementary to which variable,
and which equations are FREE ROWS (listed with no `.var` in GAMS's model statement,
so no variable is solved *for* them). If Python pairs an equation that GAMS leaves
free (or vice versa), PATH can converge (code=1, residual ~0) to a DIFFERENT root of
a multi-valued block — exactly the gtap7_3x3 factor-2 bug: GAMS `pfteq` is a free row,
Python solved it for pft and hit the spurious high root (pft≈3.6 vs GAMS 1.0). No
input/coefficient/form diff shows this; only the pairing does.

This tool:
  1. parses GAMS's `model <name> / eqA.varA, eqB.varB, eqC, ... /` block, classifying
     each equation as PAIRED(var) or FREE-ROW (no `.var`);
  2. runs the case the MCP gate builds on the core driver, seeded at the data/
     reference, and right after the solve of the period records, per equation family
     and over that period's rows, whether it is ACTIVE and whether its GAMS-paired
     variable is FIXED (the solver's free-row deactivations and fixings applied);
  3. flags families where GAMS treats the equation as a free row but Python keeps it
     active+paired (or vice versa) — the pairing mismatch class.

Usage:
  uv run python scripts/gtap/diff_mcp_pairing.py --dataset gtap7_3x3
  uv run python scripts/gtap/diff_mcp_pairing.py --dataset gtap7_3x3 --model-name gtap
"""

from __future__ import annotations
import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "gtap"))

from _parity_json import make_violation, run_tool  # noqa: E402 — shared JSON contract

from equilibria.sources import find_reference

# Explicit GAMS-equation -> Python-constraint pairing. The name heuristic below
# (eq_<gams_eq>, eq_<stem>, ...) is FRAGILE: it silently resolves to a same-named
# but WRONG (often legacy/deactivated) Python constraint. That produced 3 false
# positives on gtap7_3x3/check:
#   - pseq.ps -> heuristic finds eq_ps (legacy `ps==pd`, deactivated) instead of
#     the real homolog eq_xs (CES domestic supply, active);
#   - dintxeq/mintxeq -> the eq is deactivated but the VARIABLE is fixed (an
#     exogenous wedge / holdfix-equivalent), which is a valid pairing.
# This dict is the authoritative GAMS->Python equation map for the cases where the
# Python constraint name does NOT follow the eq_<gams_eq> convention. Anything not
# listed falls back to the name heuristic (so unmapped/new datasets still work).
GAMS_EQ_TO_PY = {
    "pseq": "eq_xs",  # CES domestic supply determines ps (NOT eq_ps legacy)
    "peq": "eq_peq",
    "pseq.ps": "eq_xs",  # tolerated alias form
}


# GAMS equation name (lowercased, no 'eq' suffix conventions vary) → Python family.
# Python uses eq_<name>; GAMS uses <name>eq. We map by stripping/adding around a stem.
def _gams_eq_to_py(gams_eq: str) -> str:
    """gamsname 'pfteq' -> python 'eq_pfteq'? No: python is eq_<stem> where GAMS is
    <stem>eq. So 'pfteq' -> stem 'pft' -> 'eq_pft'... but Python actually names it
    eq_pfteq. Python convention here is eq_<gamsname-without-trailing-eq's-stem>.
    Empirically Python = 'eq_' + gams_eq with trailing 'eq' kept for some, dropped
    for others. We try both and let the caller's model decide which exists."""
    return "eq_" + gams_eq


def _parse_gams_model(gms_path: Path, model_name: str) -> list[tuple[str, str | None]]:
    """Return [(gams_eq, paired_var_or_None)] from `model <name> / ... /`.
    A FREE ROW is an entry with no `.var` → paired_var is None."""
    text = gms_path.read_text()
    m = re.search(rf"(?ms)^\s*model\s+{re.escape(model_name)}\s*/(.*?)/\s*;", text)
    if not m:
        return []
    body = m.group(1)
    # strip GAMS line comments (* at col0) and inline ones
    lines = []
    for ln in body.splitlines():
        s = ln.strip()
        if not s or s.startswith("*"):
            continue
        lines.append(s)
    joined = " ".join(lines)
    pairs: list[tuple[str, str | None]] = []
    for tok in joined.split(","):
        tok = tok.strip()
        if not tok:
            continue
        if "." in tok:
            eq, var = tok.split(".", 1)
            pairs.append((eq.strip(), var.strip()))
        else:
            pairs.append((tok, None))  # free row
    return pairs


def _of_period(idx, period) -> bool:
    """True for every cell when ``period`` is None, else for the period's (last)."""
    if period is None:
        return True
    last = idx[-1] if isinstance(idx, tuple) and idx else idx
    return last == period


# GAMS variable -> Python Var name where they differ (as the multi-period seeder,
# GTAPMultiPeriodModel.seed_all_periods, maps them).
GAMS_VAR_TO_PY = {
    "xa": "xaa",
    "pp": "pp_rai",
    "p": "p_rai",
    "xd": "xda",
    "xm": "xma",
    "ytaxInd": "ytax_ind",
    "factY": "facty",
    "phiP": "phip",
    "regY": "regy",
    "xg": "xg_agg",
}


def _var_fixed_frac(model, var_name, period=None):
    """Return (fixed_count, total) for a Python Var, or None if absent; with
    ``period``, over that period's cells of a multi-period model.
    A GAMS-paired variable that Python FIXES (exogenous wedge / holdfix-equivalent)
    is a valid pairing even when its defining equation is deactivated."""
    from pyomo.environ import Var

    if not var_name:
        return None
    v = getattr(model, GAMS_VAR_TO_PY.get(var_name, var_name), None)
    if v is None or v.ctype is not Var:
        return None
    fixed = total = 0
    for idx in v:
        if not _of_period(idx, period):
            continue
        total += 1
        if v[idx].fixed:
            fixed += 1
    return (fixed, total)


def _py_family_state(model, gams_eq: str, paired_var: str | None = None, period=None):
    """Return (py_eq_name_or_None, active_count, total, paired_var_fixed_frac); with
    ``period``, over that period's rows and cells of a multi-period model.

    Resolution order: the explicit GAMS_EQ_TO_PY map FIRST (authoritative for
    non-conventional names like pseq->eq_xs), then the name heuristic as fallback.
    paired_var_fixed_frac is (fixed,total) for the GAMS-paired variable, or None."""
    from pyomo.environ import Constraint

    # 1) explicit map wins (handles pseq->eq_xs and any other non-eq_<name> homolog)
    cands = []
    if gams_eq in GAMS_EQ_TO_PY:
        cands.append(GAMS_EQ_TO_PY[gams_eq])
    # 2) name heuristic fallback
    cands.append(f"eq_{gams_eq}")
    if gams_eq.endswith("eq"):
        cands.append(f"eq_{gams_eq[:-2]}")  # pfteq -> eq_pft
    cands.append(f"eq_{gams_eq.replace('eq', '')}")
    comp = None
    pyname = None
    for c in cands:
        comp = getattr(model, c, None)
        if comp is not None and comp.ctype is Constraint:
            pyname = c
            break
    var_frac = _var_fixed_frac(model, paired_var, period)
    if comp is None:
        return (None, 0, 0, var_frac)
    active = total = 0
    for idx in comp:
        if not _of_period(idx, period):
            continue
        total += 1
        if comp[idx].active:
            active += 1
    return (pyname, active, total, var_frac)


# Free rows that are benign when Python keeps them active+paired:
#  - numeraire / Walras / welfare reports (pnum fixed=1, walras=slack, ev/cv=report);
#  - LINEAR market-clearing / balance identities (xseq with omegax=inf is xs=xds+xet;
#    capAccteq is Σsavf=0; savfeq under capFix is trivial) — these have ONE root, so
#    pairing them with the variable they define is correct, NOT the multi-root class.
#  - pfteq: it WAS the dangerous class, a power/quadratic free row
#    (pft^(1+omega)=Σgf·pfy^…, two roots: the gtap7_3x3 factor-2 bug). Since ef807eb
#    eq_pfteq is the single positive root pft=(Σgf·pfy^(1+omega))^(1/(1+omega)), so
#    pairing it with pft has one root (blocks/gtap/factor.py, EqPfteq).
# The DANGEROUS class is a power/quadratic free row, which is multi-valued; that is
# the only one PATH can converge to a wrong root for.
# NOTE: xseq/savfeq's benignness depends on omegax=inf / savfFlag=capFix — re-check
# for datasets with finite omegax or capFlex/capShrFix closures.
BENIGN = {"pnumeq", "walraseq", "eveq", "cveq", "xseq", "capAccteq", "savfeq", "pfteq"}
_NUMERAIRE_WALRAS = {"pnumeq", "walraseq", "eveq", "cveq"}


def _classify_free_row(
    eq: str, pyname, active: int, total: int
) -> tuple[str, float, str]:
    """Classify a GAMS FREE-ROW against Python state.

    Returns (kind, severity, note). Only kind=='pairing_mismatch' (severity 1.0)
    is a real violation: a multi-root ECONOMIC free-row that Python keeps
    active+paired (the pfteq factor-2 class). Everything else is non-violating
    context (severity 0.0).
    """
    if pyname is None:
        return ("free_row_gams_only_no_py_eq", 0.0, "no Python eq (GAMS-only)")
    if active == 0:
        return ("matches", 0.0, "deactivated → matches GAMS free-row")
    if eq in BENIGN:
        kindnote = (
            "numeraire/walras/welfare report"
            if eq in _NUMERAIRE_WALRAS
            else "single-root form, ef807eb"
            if eq == "pfteq"
            else "linear market-clearing/balance identity (single root)"
        )
        return ("benign_active", 0.0, f"active but BENIGN ({kindnote})")
    return (
        "pairing_mismatch",
        1.0,
        "ACTIVE+paired in Python but FREE-ROW in GAMS → may converge to a "
        "different root (pfteq factor-2 class)",
    )


# ---------------------------------------------------------------------------
# Human-readable formatter — KEPT for debug only; writes to STDERR so it can
# never contaminate the JSON on stdout. Not called in the normal path.
# ---------------------------------------------------------------------------
def _debug_print(model_name, gms_name, pairs, free_rows, rows):
    print(f"=== GAMS `model {model_name}` pairing ({gms_name}) ===", file=sys.stderr)
    print(f"  {len(pairs)} entries, {len(free_rows)} FREE ROWS:", file=sys.stderr)
    print(f"    {', '.join(free_rows)}", file=sys.stderr)
    print(
        f"\n  {'gams_eq':<12}{'python_eq':<16}{'py_active':>10}   note", file=sys.stderr
    )
    for r in rows:
        astr = (
            "-"
            if r["py_total"] == 0 and r["pyname"] is None
            else f"{r['py_active']}/{r['py_total']}"
        )
        print(
            f"  {r['gams_eq']:<12}{(r['pyname'] or '-'):<16}{astr:>10}   {r['note']}",
            file=sys.stderr,
        )


def _resolve_gms_path(dataset: str, ifsub: int) -> tuple[Path | None, bool]:
    """The GAMS model recorded with the cascade's altertax reference (data/,
    fbep=header; cascade_config.resolve_ref_gdx), so the pairing is read from the
    model that wrote it (the tool is altertax-only). The `model gtap /eq.var/` block is the SAME for
    every dataset (only sets/data differ), so a dataset without its own reference
    borrows a sibling's."""
    for i, ds in enumerate(
        (dataset, *(d for d in _GMS_FALLBACK_DATASETS if d != dataset))
    ):
        folder = find_reference(
            ds, "gams", form="mcp", mode="altertax", ifsub=ifsub, fbep="header"
        )
        if folder is not None and (folder / "model.gms").exists():
            return (folder / "model.gms", i > 0)
    return (None, False)


_GMS_FALLBACK_DATASETS = ("gtap7_3x3", "gtap7_10x7", "gtap7_15x10")


class _Cell:
    __slots__ = ("active", "fixed")

    def __init__(self, active: bool = False, fixed: bool = False):
        self.active, self.fixed = active, fixed


class _Frozen(dict):
    """One component's ``{index: _Cell}`` with its Pyomo ``ctype``."""

    def __init__(self, ctype):
        super().__init__()
        self.ctype = ctype


# In check and shock the driver swaps eq_nd/eq_va for the same demand row with its
# share recalibrated from the prior period (pyomo_ops, one scalar Constraint per
# cell named _eq_<nd|va>_recal_<r>_<a>_<period>); the cell's row is that one.
_RECAL_ROWS = frozenset({"eq_nd", "eq_va"})


def _recal_name(family: str, idx) -> str:
    return "_" + family + "_recal_" + "_".join(str(x) for x in idx)


def _freeze_state(model, period: str):
    """A copy of ``period``'s row activity and cell fixings, readable as the model
    is (``getattr(m, name)``, iteration, ``[idx].active``/``.fixed``, ``ctype``).
    The Pyomo model is shared: after the solve the driver fixes the period's cells
    (freeze_period) and the next period switches its rows off (freeze_inactive),
    so the model read later no longer shows what the solver was handed."""
    from types import SimpleNamespace

    from pyomo.environ import Constraint, Var

    state = SimpleNamespace()
    for c in model.component_objects(Constraint):
        comp = _Frozen(Constraint)
        for idx in c:
            if _of_period(idx, period):
                active = c[idx].active
                if not active and c.local_name in _RECAL_ROWS:
                    recal = getattr(model, _recal_name(c.local_name, idx), None)
                    active = recal is not None and recal.active
                comp[idx] = _Cell(active=active)
        setattr(state, c.local_name, comp)
    for v in model.component_objects(Var):
        comp = _Frozen(Var)
        for idx in v:
            if _of_period(idx, period):
                comp[idx] = _Cell(fixed=v[idx].fixed)
        setattr(state, v.local_name, comp)
    return state


def gate_model_at_solve(dataset: str, period: str, mode: str, ifsub: int, gdx: Path):
    """The state of the case the MCP gate builds, on the core driver seeded at
    ``gdx``, as it stands right after the solve of ``period`` (the solver's own free
    rows and fixings applied, the other periods fixed), copied there
    (``_freeze_state``); None if no solve of it ran."""
    from _core_run import solve_on_core
    from seed_and_solve import _case

    from equilibria.templates.gtap.recipe import _by_backend

    seen: dict = {}

    def spy(model):
        status = _by_backend(model)
        staged = model.staged
        if staged is not None and staged.period == period and "m" not in seen:
            seen["m"] = _freeze_state(model.pyomo, period)
        return status

    p, gc, rr = _case(dataset, bool(ifsub), mode)
    solve_on_core(
        p,
        gc,
        rr,
        mode=mode,
        ref_gdx=gdx,
        seed_gdx=gdx,
        solve_base=period == "base",
        solver=spy,
    )
    return seen.get("m")


def _work(args) -> dict:
    gms_path, gms_is_fallback = _resolve_gms_path(args.dataset, args.ifsub)
    if gms_path is None:
        return dict(
            status="error",
            period=args.period,
            headline=f"GAMS source not found for {args.dataset} "
            f"(no own .gms, no fallback)",
            violations=[],
            meta={"error_kind": "gams_source_missing", "dataset": args.dataset},
        )

    pairs = _parse_gams_model(gms_path, args.model_name)
    if not pairs:
        return dict(
            status="error",
            period=args.period,
            headline=(
                f"model '{args.model_name}' block not parsed from {gms_path.name}"
            ),
            violations=[],
            meta={"error_kind": "model_block_unparsed", "gms_path": str(gms_path)},
        )

    free_rows = [eq for eq, var in pairs if var is None]
    paired = [(eq, var) for eq, var in pairs if var is not None]

    mode = getattr(args, "mode", "altertax")
    gdx = getattr(args, "gdx", None)
    if gdx is None:
        folder = find_reference(
            args.dataset, "gams", form="mcp", mode=mode, ifsub=args.ifsub, fbep="header"
        )
        gdx = None if folder is None else folder / "out.gdx"
    if gdx is None or not Path(gdx).exists():
        return dict(
            status="error",
            period=args.period,
            violations=[],
            headline=f"reference GDX not found for {args.dataset}",
            meta={"error_kind": "gdx_not_found"},
        )
    model = gate_model_at_solve(args.dataset, args.period, mode, args.ifsub, Path(gdx))
    if model is None:
        return dict(
            status="error",
            period=args.period,
            violations=[],
            headline=f"MCP pairing: no solve of {args.period} was observed",
            meta={"error_kind": "solve_not_observed"},
        )
    closure_applied, closure_warn = True, None  # the core solve applied it
    per = args.period

    violations = []
    rows = []  # full per-free-row detail for meta + debug print

    # Direction 1: GAMS FREE-ROW vs Python state. The dangerous mismatch is a
    # multi-root economic free-row that Python keeps active+paired.
    for eq in free_rows:
        pyname, active, total, _ = _py_family_state(model, eq, None, per)
        kind, sev, note = _classify_free_row(eq, pyname, active, total)
        rows.append(
            {
                "gams_eq": eq,
                "pyname": pyname,
                "py_active": active,
                "py_total": total,
                "kind": kind,
                "note": note,
                "direction": "gams_free_row",
            }
        )
        if kind == "pairing_mismatch":
            v = make_violation(eq, [eq], "mcp_pairing", sev)
            v["kind"] = "pairing_mismatch"
            v["python_eq"] = pyname
            v["python_active"] = f"{active}/{total}"
            v["gams_side"] = "free_row"
            v["python_side"] = "active_paired"
            v["note"] = note
            violations.append(v)

    # Direction 2 (the "or vice versa"): a GAMS-PAIRED equation that Python has
    # fully DEACTIVATED — Python freed a row GAMS solves for a variable, so the
    # variable Python should pin via that eq is left to drift. Candidate for the
    # ytax/rorc root-selection class.
    # EXCEPTION: if Python FIXES the paired variable, the deactivated equation is
    # the correct Python encoding of an exogenous wedge / holdfix-equivalent (GAMS
    # pins it via holdfixed=1 from the prior period). A fixed variable IS paired —
    # not a free DOF — so this is NOT a violation (e.g. dintx/mintx tax wedges).
    for eq, var in paired:
        pyname, active, total, var_frac = _py_family_state(model, eq, var, per)
        if pyname is not None and total > 0 and active == 0:
            var_is_fixed = (
                var_frac is not None and var_frac[1] > 0 and var_frac[0] == var_frac[1]
            )
            if var_is_fixed:
                rows.append(
                    {
                        "gams_eq": eq,
                        "pyname": pyname,
                        "py_active": active,
                        "py_total": total,
                        "kind": "benign_fixed_var",
                        "note": (
                            f"eq deactivated but var {var} is "
                            f"{var_frac[0]}/{var_frac[1]} FIXED "
                            f"(exogenous wedge / holdfix-equivalent) → "
                            f"valid pairing"
                        ),
                        "direction": "gams_paired",
                    }
                )
                continue
            v = make_violation(eq, [eq], "mcp_pairing", 1.0)
            v["kind"] = "free_row_python_only"
            v["python_eq"] = pyname
            v["python_active"] = f"{active}/{total}"
            v["gams_paired_var"] = var
            v["gams_side"] = f"paired(.{var})"
            v["python_side"] = "deactivated"
            v["note"] = (
                f"GAMS pairs {eq}.{var} but Python deactivates {pyname} "
                f"→ {var} is left unpaired (root-selection candidate)"
            )
            violations.append(v)
            rows.append(
                {
                    "gams_eq": eq,
                    "pyname": pyname,
                    "py_active": active,
                    "py_total": total,
                    "kind": "free_row_python_only",
                    "note": v["note"],
                    "direction": "gams_paired",
                }
            )

    _debug_print(args.model_name, gms_path.name, pairs, free_rows, rows)

    n_mismatch = len(violations)
    n_benign = sum(
        1 for r in rows if r["kind"] in ("benign_active", "benign_fixed_var")
    )
    status = "dirty" if violations else "clean"
    if violations:
        kinds = {}
        for v in violations:
            kinds[v["kind"]] = kinds.get(v["kind"], 0) + 1
        kind_str = ", ".join(f"{n}×{k}" for k, n in kinds.items())
        headline = (
            f"MCP pairing ({args.period}): {n_mismatch} divergent row(s) "
            f"[{kind_str}] — a GAMS free-row Python pairs (or a GAMS-paired "
            f"eq Python frees); candidate missing anchor / wrong root"
        )
    else:
        headline = (
            f"MCP pairing ({args.period}): every economic GAMS free-row is "
            f"deactivated/absent in Python and every GAMS-paired eq is "
            f"active — pairing matches; this layer does not explain the gap"
        )

    return dict(
        status=status,
        period=args.period,
        headline=headline,
        violations=violations,
        meta={
            "model_name": args.model_name,
            "ifsub": args.ifsub,
            "gms_path": str(gms_path),
            "gms_is_fallback": gms_is_fallback,
            "n_model_entries": len(pairs),
            "n_free_rows": len(free_rows),
            "free_rows": free_rows,
            "n_benign_active": n_benign,
            "case": "mcp_gate",
            "apply_closure": args.apply_closure,
            "closure_applied": closure_applied,
            "closure_warn": closure_warn,
            "free_row_states": rows,
        },
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="gtap7_3x3")
    ap.add_argument(
        "--model-name", default="gtap", help="GAMS model name (default gtap)"
    )
    ap.add_argument("--ifsub", type=int, default=0, choices=[0, 1])
    ap.add_argument("--period", default="shock", choices=["base", "check", "shock"])
    ap.add_argument(
        "--apply-closure",
        action="store_true",
        help="kept for the orchestrator: the Python state is always read "
        "right after the core solve, with the solver's closure applied",
    )
    ap.add_argument(
        "--gdx",
        type=Path,
        default=None,
        help="reference GDX the case is seeded from (default: the data/ "
        "GAMS MCP reference)",
    )
    ap.add_argument(
        "--mode",
        default="altertax",
        choices=["altertax", "gtap"],
        help="altertax CD (default) or pure-gtap real-CES",
    )
    args = ap.parse_args()

    # mode=gtap UNSUPPORTED: this tool targets the altertax CD model; the
    # pure-gtap shock is built in place when the multi-period exercise runs (the
    # core recipe gtap_static). Accept the flag (the orchestrator passes it) but emit an
    # honest mode_unsupported rather than diff the wrong model. Use
    # seed_and_solve --mode gtap for the pure-gtap diagnostic.
    if args.mode == "gtap":

        def _unsupported() -> dict:
            return dict(
                status="error",
                period=getattr(args, "period", None),
                headline=(
                    "diff_mcp_pairing does not support --mode gtap (altertax-only tool); "
                    "use seed_and_solve --mode gtap for the pure-gtap diagnostic."
                ),
                violations=[],
                meta={
                    "error_kind": "mode_unsupported",
                    "mode": "gtap",
                    "ifsub": args.ifsub,
                },
            )

        return run_tool(
            "diff_mcp_pairing",
            args.dataset,
            _unsupported,
            period_hint=getattr(args, "period", None),
        )
    return run_tool(
        "diff_mcp_pairing", args.dataset, lambda: _work(args), period_hint=args.period
    )


if __name__ == "__main__":
    raise SystemExit(main())
