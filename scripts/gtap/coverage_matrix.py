"""GTAP7 parity coverage matrix — the single source of truth.

One declarative table (`ROWS`) describes parity coverage across
dataset x kind (gtap|altertax|gtap_solve) x ifSUB x phase (base/check/shock). Drives:
  * test_gtap7_nl_parity.py        (which gtap rows/phases run in CI)
  * test_altertax_multiperiod_parity.py  (per-row gap_min contract)
  * test_gtap_multiperiod_parity.py (the pure-gtap SOLVE gate, gtap_solve rows)
  * gen_coverage_doc.py            (the generated docs/site/guide/gtap7_coverage_matrix.md)

Three axes / kinds:
  * "gtap"       — the .nl COEFFICIENT gate (no solver, CI). Phases are base+shock,
                   plus `check` where a gams_check.nl fixture exists (3x3/5x5/10x7 →
                   the check is the CD multi-period step). Contract: 0 coeff diffs.
  * "altertax"   — the altertax-CD multi-period SOLVE gate (PATH, local), per ifSUB.
  * "gtap_solve" — the PURE-gtap (real-CES, non-altertax) multi-period SOLVE gate
                   (PATH, local), per ifSUB. CHECK + SHOCK match% vs the GAMS LOCAL
                   out_gtap_shock_ifsub{0,1}.gdx (gtap7_3x3 local, gtap7_5x5 via NEOS).

`gap_min` is a CONSERVATIVE floor the tests assert (margin below the measured
value); `gap_note` is the measured snapshot for humans. `gap_min is None` only
for the .nl-only gtap7_* rows (their contract is "0 coefficient diffs", not a
percentage). `ci_status` records honestly what runs where:
  * "ci"      runs on ubuntu CI without a solver (the .nl gate rows)
  * "local"   runs only locally (needs PATH+GAMS) — the SOLVE gate rows
  * "blocked" cannot be verified (reference unsound — 20x41 ref violates 37 of
              its own eqs, see validate_reference)
"""
from __future__ import annotations

from dataclasses import dataclass

CI_STATUSES = {"ci", "local", "blocked"}
KINDS = {"gtap", "altertax", "gtap_solve"}


@dataclass(frozen=True)
class Row:
    dataset: str
    kind: str               # "gtap" | "altertax" | "gtap_solve"
    ifsub: int | None       # None for gtap; 0 or 1 for altertax/gtap_solve
    phases: tuple[str, ...]  # phases with coverage
    gap_min: float | None   # contract floor; None for .nl-only gtap7_* rows
    gap_note: str           # measured snapshot, e.g. "100%", "~99%"
    ci_status: str          # "ci" | "local" | "blocked"
    ref: str                # provenance


ROWS: list[Row] = [
    # --- single-period .nl gate (CI, no solver) ---
    Row("nus333", "gtap", None, ("base", "shock"), 99.5, "100% (NEOS+GAMS)", "ci", "nus333 NEOS"),
    Row("9x10", "gtap", None, ("base", "shock"), 99.5, "100% (NEOS)", "ci", "job 18737509"),
    Row("gtap7_3x3", "gtap", None, ("base", "shock"), None, "0 diffs .nl", "ci", "gams_base/shock.nl"),
    Row("gtap7_5x5", "gtap", None, ("base", "shock"), None, "0 diffs .nl", "ci", "gams_base/shock.nl"),
    Row("gtap7_10x7", "gtap", None, ("base", "shock"), None, "0 diffs .nl", "ci", "gams_base/shock.nl"),
    Row("gtap7_15x10", "gtap", None, ("base", "shock"), None, "0 diffs .nl", "ci", "gams_base/shock.nl"),
    Row("gtap7_3x4", "gtap", None, ("base", "shock"), None, "0 diffs .nl", "ci", "gams_base/shock.nl"),
    # --- altertax multi-period (solver gate, local-only), both ifSUB modes ---
    # gap_note = freshly MEASURED shock match% @ tol1% (2026-06-30, all code=1/1/1).
    # The tol0.5% strict band is lower where noted (eq_paa micro-cells; see sweep doc).
    Row("gtap7_3x3", "altertax", 0, ("base", "check", "shock"), 98.0, "99.93% (98.88% @0.5%)", "local", "out_altertax_ifsub0.gdx"),
    Row("gtap7_3x3", "altertax", 1, ("base", "check", "shock"), 98.0, "99.78% (98.51% @0.5%)", "local", "out_altertax_ifsub1.gdx"),
    Row("gtap7_5x5", "altertax", 0, ("base", "check", "shock"), 99.5, "99.88% (98.53% @0.5%)", "local", "out_altertax_ifsub0.gdx"),
    Row("gtap7_5x5", "altertax", 1, ("base", "check", "shock"), 99.5, "99.81% (98.38% @0.5%)", "local", "out_altertax_ifsub1.gdx"),
    Row("gtap7_10x7", "altertax", 0, ("base", "check", "shock"), 98.0, "99.33% (96.83% @0.5%)", "local", "out_altertax_ifsub0.gdx"),
    Row("gtap7_10x7", "altertax", 1, ("base", "check", "shock"), 98.0, "99.31% (96.81% @0.5%)", "local", "out_altertax_ifsub1.gdx"),
    Row("gtap7_15x10", "altertax", 0, ("base", "check", "shock"), 99.0, "99.57% (98.19% @0.5%)", "local", "out_altertax_ifsub0.gdx"),
    Row("gtap7_15x10", "altertax", 1, ("base", "check", "shock"), 99.0, "99.40% (97.92% @0.5%)", "local", "out_altertax_ifsub1.gdx"),
    Row("gtap7_3x4", "altertax", 0, ("base", "check", "shock"), 99.0, "99.72% (96.79% @0.5%)", "local", "out_altertax_ifsub0.gdx"),
    Row("gtap7_3x4", "altertax", 1, ("base", "check", "shock"), 99.0, "99.72% (96.46% @0.5%)", "local", "out_altertax_ifsub1.gdx"),
    Row("gtap7_20x41", "altertax", 0, ("base",), None, "blocked: ref violates 37 own eqs", "blocked", "out_altertax_ifsub0.gdx (corrupt)"),
    # --- PURE-gtap (real-CES, non-altertax) multi-period SOLVE gate, local-only ---
    # gtap7_3x3: local GAMS (community license fits). gtap7_5x5: GAMS refs generated
    # via NEOS (job 19755905/6, PATH/MCP — exceeds the 2500-row community limit).
    # CHECK exact (100%) all rows; SHOCK measured @ tol1%. ALL CLOSED (commit f570e32):
    # the real-sluggish factor price pft was wrongly FIXED by fix_sluggish_pft (it used
    # a nonexistent xftflag Param → except:pass → froze pft against the tariff). Freeing
    # it (eq_xfteq.active guard) + deleting the 3x3-hardcoded eq_pfyeq trim (matcher
    # squares automatically) closed 5x5 ifSUB=0 64.87→100% and improved every case.
    Row("gtap7_3x3", "gtap_solve", 0, ("base", "check", "shock"), 99.0, "100% (CHECK 100%)", "local", "out_gtap_shock_ifsub0.gdx"),
    Row("gtap7_3x3", "gtap_solve", 1, ("base", "check", "shock"), 98.0, "99.4% (CHECK 100%)", "local", "out_gtap_shock_ifsub1.gdx"),
    Row("gtap7_5x5", "gtap_solve", 0, ("base", "check", "shock"), 99.0, "100% (CHECK 100%)", "local", "out_gtap_shock_ifsub0.gdx (NEOS)"),
    Row("gtap7_5x5", "gtap_solve", 1, ("base", "check", "shock"), 99.0, "99.59% (CHECK 100%)", "local", "out_gtap_shock_ifsub1.gdx (NEOS)"),
    # gtap7_10x7 ifSUB=0: NEOS ref (job 19760909, PATH, CLEAN solve). The same pft-free
    # fix generalizes — 100% with zero worst-cells (>1% count 0). Confirms the fix is not
    # dataset-specific.
    Row("gtap7_10x7", "gtap_solve", 0, ("base", "check", "shock"), 99.0, "100% (CHECK 100%)", "local", "out_gtap_shock_ifsub0.gdx (NEOS)"),
    # gtap7_10x7 ifSUB=1: NEOS ref (job 19762403, PATH). The ifSUB=1 shock is NOT the soft
    # basin the ifSUB=0 shock is: with default PATH the shock DIVERGES from the check
    # warm-start (regYeq 3.2→factYeq 13.8→xfeq 43.7, stalls, →inf on eveq) and exits at
    # the iteration limit with residual ~3.2 — the saved GDX then violated its OWN regYeq
    # (regY≠factY+yTaxInd by 1.2) → Python read ~67%. Fix: a PATH path.opt (proximal
    # perturbation + higher major/minor limits + crash pnewton) stabilises the restart →
    # '** EXIT - solution found.' → 99.5% (0.1/0.5/1% = 94.58/99.18/99.5). Faithful (solver
    # tuning, no eq change); gated ifSUB=1 in build_gtap7_pure_local_bundle.py.
    Row("gtap7_10x7", "gtap_solve", 1, ("base", "check", "shock"), 99.0, "99.5% (CHECK 100%)", "local", "out_gtap_shock_ifsub1.gdx (NEOS)"),
    # gtap7_15x10 ifSUB=0: NEOS ref (job 19761766, PATH, CLEAN). The RAW GtapView GDX was
    # missing incpar/subpar (CDE params → div-by-zero), a_/c_ prefixes (acts==comm collision),
    # and carried RAW rT*/VMTS ($140). build_v7_processed_from_raw.py rebuilds the PROCESSED
    # schema (calibration from default.prm) → all fixed at once → 100% (0.1/0.5/1% = 99.99/
    # 100/100). Largest dataset (41987 rows); ref needs NEOS (teaching license has the
    # 2500-row nonlinear cap).
    Row("gtap7_15x10", "gtap_solve", 0, ("base", "check", "shock"), 99.0, "100% (CHECK 100%)", "local", "out_gtap_shock_ifsub0.gdx (NEOS)"),
    # gtap7_15x10 ifSUB=1: NEOS ref (job 19762404, PATH + path.opt). Same non-converged
    # ref as 10x7 ifSUB=1 (default PATH diverged, iteration limit, ref violated its own
    # regYeq by 1.5); the path.opt stabilisation (proximal perturbation) converged it →
    # 66.25% → 95.68% (0.1/0.5/1% = 81.42/92.85/95.68), CHECK 100%, codes 1/1/1. The ~4pp
    # residual @ tol1% is the known eq_paa Armington micro-cell family (worst cells are
    # USA a_VegFruit / a_ForestFish inputs, py ~1e-2..1e-5 — see
    # project_gtap7_armington_shares_bug), scale-amplified on the 42k-row dataset, NOT a
    # new equation.
    Row("gtap7_15x10", "gtap_solve", 1, ("base", "check", "shock"), 95.0, "95.68% (CHECK 100%)", "local", "out_gtap_shock_ifsub1.gdx (NEOS)"),
]


def nl_rows() -> list[Row]:
    return [r for r in ROWS if r.kind == "gtap"]


def altertax_rows() -> list[Row]:
    return [r for r in ROWS if r.kind == "altertax"]


def gtap_solve_rows() -> list[Row]:
    return [r for r in ROWS if r.kind == "gtap_solve"]


def _validate() -> None:
    """Import-time schema invariants — fail fast on a malformed matrix."""
    for r in ROWS:
        # ifsub is None ONLY for the .nl "gtap" kind; the SOLVE kinds carry 0/1.
        assert (r.ifsub is None) == (r.kind == "gtap"), f"ifsub/kind mismatch: {r}"
        assert r.kind in KINDS, f"bad kind: {r}"
        assert r.ci_status in CI_STATUSES, f"bad ci_status: {r}"
        assert r.phases, f"empty phases: {r}"
        # gap_min invariants do NOT apply to blocked rows (never asserted).
        if r.ci_status != "blocked":
            nl_only = r.kind == "gtap" and r.dataset.startswith("gtap7_")
            assert (r.gap_min is None) == nl_only, f"gap_min/nl-only mismatch: {r}"
            if r.gap_min is not None:
                assert 0.0 < r.gap_min < 100.0, f"gap_min must be a floor <100: {r}"


_validate()
