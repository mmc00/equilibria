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

import re
from dataclasses import dataclass

CI_STATUSES = {"ci", "local", "blocked"}


@dataclass(frozen=True)
class Row:
    dataset: str
    variant: str            # "core" | "altertax"  (antes: kind "gtap"|"altertax")
    period: str             # "single" | "multi"
    ifsub: int | None       # None for core; 0 or 1 for altertax
    phases: tuple[str, ...]  # phases with coverage
    gap_min: float | None   # contract floor vs GAMS; None for .nl-only core rows
    gap_note: str           # measured snapshot vs GAMS, e.g. "100%", "~99%"
    ci_status: str          # "ci" | "local" | "blocked"
    ref: str                # GAMS provenance (MCP→ifMCP=1, NLP→ifMCP=0)
    model: str = "gtap7"    # "gtap7" | "gtap6"
    solver: str = "mcp"     # "mcp" | "nlp"
    gap_gempack: float | None = None   # floor vs GEMPACK/RunGTAP
    note_gempack: str = ""             # snapshot vs GEMPACK/RunGTAP
    ref_gempack: str | None = None     # RunGTAP provenance

    @property
    def kind(self) -> str:
        """Back-compat: legacy 'gtap'|'altertax' derived from variant."""
        return "altertax" if self.variant == "altertax" else "gtap"


ROWS: list[Row] = [
    # --- single-period .nl gate (CI, no solver) ---
    Row("nus333", "core", "single", None, ("base", "shock"), 99.5, "100% (NEOS+GAMS)", "ci", "nus333 NEOS"),
    Row("9x10", "core", "single", None, ("base", "shock"), 99.5, "100% (NEOS)", "ci", "job 18737509"),
    Row("gtap7_3x3", "core", "single", None, ("base", "shock"), None, "0 diffs .nl", "ci", "gams_base/shock.nl"),
    Row("gtap7_5x5", "core", "single", None, ("base", "shock"), None, "0 diffs .nl", "ci", "gams_base/shock.nl"),
    Row("gtap7_10x7", "core", "single", None, ("base", "shock"), None, "0 diffs .nl", "ci", "gams_base/shock.nl"),
    Row("gtap7_15x10", "core", "single", None, ("base", "shock"), None, "0 diffs .nl", "ci", "gams_base/shock.nl"),
    Row("gtap7_3x4", "core", "single", None, ("base", "shock"), None, "0 diffs .nl", "ci", "gams_base/shock.nl"),
    # --- altertax multi-period (solver gate, local-only), both ifSUB modes ---
    Row("gtap7_3x3", "altertax", "multi", 0, ("base", "check", "shock"), 98.0, "~99%", "local", "out_altertax_ifsub0.gdx"),
    Row("gtap7_3x3", "altertax", "multi", 1, ("base", "check", "shock"), 98.0, "~99%", "local", "out_altertax_ifsub1.gdx"),
    Row("gtap7_5x5", "altertax", "multi", 0, ("base", "check", "shock"), 99.5, "100%", "local", "out_altertax_ifsub0.gdx"),
    Row("gtap7_5x5", "altertax", "multi", 1, ("base", "check", "shock"), 99.5, "100%", "local", "out_altertax_ifsub1.gdx"),
    Row("gtap7_10x7", "altertax", "multi", 0, ("base", "check", "shock"), 98.0, "~99%", "local", "out_altertax_ifsub0.gdx"),
    Row("gtap7_10x7", "altertax", "multi", 1, ("base", "check", "shock"), 98.0, "~99%", "local", "out_altertax_ifsub1.gdx"),
    Row("gtap7_15x10", "altertax", "multi", 0, ("base", "check", "shock"), 99.0, "99.30%", "local", "out_altertax_ifsub0.gdx"),
    Row("gtap7_15x10", "altertax", "multi", 1, ("base", "check", "shock"), 99.0, "99.30%", "local", "out_altertax_ifsub1.gdx"),
    Row("gtap7_3x4", "altertax", "multi", 0, ("base", "check", "shock"), 99.0, "99.61%", "local", "out_altertax_ifsub0.gdx"),
    Row("gtap7_3x4", "altertax", "multi", 1, ("base", "check", "shock"), 99.0, "99.56%", "local", "out_altertax_ifsub1.gdx"),
    Row("gtap7_20x41", "altertax", "multi", 0, ("base",), None, "blocked", "blocked", "NEOS ref Infeasible"),
]


def nl_rows() -> list[Row]:
    return [r for r in ROWS if r.kind == "gtap"]


def altertax_rows() -> list[Row]:
    return [r for r in ROWS if r.kind == "altertax"]


def gtap7_rows() -> list[Row]:
    return [r for r in ROWS if r.model == "gtap7"]


def _is_done(r: Row) -> bool:
    note = r.gap_note.strip().lower()
    if "0 diffs" in note or "100%" in note:
        return True
    # parse a leading percentage like "99.30%" or "~99%"
    m = re.search(r"(\d+(?:\.\d+)?)\s*%", note)
    return bool(m) and float(m.group(1)) >= 99.0


def progress() -> dict[str, int]:
    buckets = {"done": 0, "partial": 0, "blocked": 0, "total": len(ROWS)}
    for r in ROWS:
        if r.ci_status == "blocked":
            buckets["blocked"] += 1
        elif _is_done(r):
            buckets["done"] += 1
        else:
            buckets["partial"] += 1
    return buckets


VARIANTS = {"core", "altertax"}
PERIODS = {"single", "multi"}
SOLVERS = {"mcp", "nlp"}
MODELS = {"gtap6", "gtap7"}


def _validate() -> None:
    """Import-time schema invariants — fail fast on a malformed matrix."""
    for r in ROWS:
        assert r.model in MODELS, f"bad model: {r}"
        assert r.variant in VARIANTS, f"bad variant: {r}"
        assert r.period in PERIODS, f"bad period: {r}"
        assert r.solver in SOLVERS, f"bad solver: {r}"
        assert r.ci_status in CI_STATUSES, f"bad ci_status: {r}"
        assert r.phases, f"empty phases: {r}"
        # core ⇒ single ⇒ ifsub None ; altertax ⇒ multi
        if r.variant == "core":
            assert r.period == "single", f"core must be single: {r}"
            assert r.ifsub is None, f"core has no ifsub: {r}"
        else:
            assert r.period == "multi", f"altertax must be multi: {r}"
        # gap_min invariants do NOT apply to blocked rows (never asserted).
        if r.ci_status != "blocked":
            nl_only = r.variant == "core" and r.dataset.startswith("gtap7_")
            assert (r.gap_min is None) == nl_only, f"gap_min/nl-only mismatch: {r}"
            if r.gap_min is not None:
                assert 0.0 < r.gap_min < 100.0, f"gap_min must be a floor <100: {r}"
        # gempack floor, when present, is a sane sub-100 percentage
        if r.gap_gempack is not None:
            assert 0.0 < r.gap_gempack < 100.0, f"gap_gempack must be <100: {r}"


_validate()
