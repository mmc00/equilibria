"""GTAP7 parity coverage matrix — the single source of truth.

One declarative table (`ROWS`) describes parity coverage across
dataset x kind (gtap|altertax) x ifSUB x phase (base/check/shock). It drives:
  * test_gtap7_nl_parity.py        (which gtap rows/phases run in CI)
  * test_altertax_multiperiod_parity.py  (per-row gap_min contract)
  * gen_coverage_doc.py            (the generated docs/site/guide/gtap7_coverage_matrix.md)

`gap_min` is a CONSERVATIVE floor the tests assert (margin below the measured
value); `gap_note` is the measured snapshot for humans. `gap_min is None` only
for the .nl-only gtap7_* single-period rows (their contract is "0 coefficient
diffs", not a percentage). `ci_status` records honestly what runs where:
  * "ci"      runs on ubuntu CI without a solver (the .nl gate rows)
  * "local"   runs only locally (needs PATH+GAMS) — the altertax solver rows
  * "blocked" cannot be verified (reference unsound, e.g. 20x41 NEOS Infeasible)
"""
from __future__ import annotations

from dataclasses import dataclass

CI_STATUSES = {"ci", "local", "blocked"}


@dataclass(frozen=True)
class Row:
    dataset: str
    kind: str               # "gtap" (.nl) | "altertax" | "gtap_solve"
    ifsub: int | None       # None for gtap (.nl); 0 or 1 for altertax/gtap_solve
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
    Row("gtap7_3x3", "altertax", 0, ("base", "check", "shock"), 98.0, "~99%", "local", "out_altertax_ifsub0.gdx"),
    Row("gtap7_3x3", "altertax", 1, ("base", "check", "shock"), 98.0, "~99%", "local", "out_altertax_ifsub1.gdx"),
    Row("gtap7_5x5", "altertax", 0, ("base", "check", "shock"), 99.5, "100%", "local", "out_altertax_ifsub0.gdx"),
    Row("gtap7_5x5", "altertax", 1, ("base", "check", "shock"), 99.5, "100%", "local", "out_altertax_ifsub1.gdx"),
    Row("gtap7_10x7", "altertax", 0, ("base", "check", "shock"), 98.0, "~99%", "local", "out_altertax_ifsub0.gdx"),
    Row("gtap7_10x7", "altertax", 1, ("base", "check", "shock"), 98.0, "~99%", "local", "out_altertax_ifsub1.gdx"),
    Row("gtap7_15x10", "altertax", 0, ("base", "check", "shock"), 99.0, "99.30%", "local", "out_altertax_ifsub0.gdx"),
    Row("gtap7_15x10", "altertax", 1, ("base", "check", "shock"), 99.0, "99.30%", "local", "out_altertax_ifsub1.gdx"),
    Row("gtap7_3x4", "altertax", 0, ("base", "check", "shock"), 99.0, "99.61%", "local", "out_altertax_ifsub0.gdx"),
    Row("gtap7_3x4", "altertax", 1, ("base", "check", "shock"), 99.0, "99.56%", "local", "out_altertax_ifsub1.gdx"),
    Row("gtap7_20x41", "altertax", 0, ("base",), None, "blocked", "blocked", "NEOS ref Infeasible"),
    # --- pure-gtap multi-period SOLVE gate (local-only), both ifSUB modes ---
    Row("gtap7_3x3", "gtap_solve", 0, ("base", "check", "shock"), 99.0, "99.70%", "local", "out_gtap_shock_ifsub0.gdx"),
    Row("gtap7_3x3", "gtap_solve", 1, ("base", "check", "shock"), 98.0, "98.95%", "local", "out_gtap_shock_ifsub1.gdx"),
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
        assert (r.ifsub is None) == (r.kind == "gtap"), f"ifsub/kind mismatch: {r}"
        assert r.kind in ("gtap", "altertax", "gtap_solve"), f"bad kind: {r}"
        assert r.ci_status in CI_STATUSES, f"bad ci_status: {r}"
        assert r.phases, f"empty phases: {r}"
        # gap_min invariants do NOT apply to blocked rows (never asserted).
        if r.ci_status != "blocked":
            nl_only = r.kind == "gtap" and r.dataset.startswith("gtap7_")
            assert (r.gap_min is None) == nl_only, f"gap_min/nl-only mismatch: {r}"
            if r.gap_min is not None:
                assert 0.0 < r.gap_min < 100.0, f"gap_min must be a floor <100: {r}"


_validate()
