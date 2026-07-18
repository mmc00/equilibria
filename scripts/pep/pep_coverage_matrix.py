"""PEP-1-1 parity coverage matrix — the single source of truth.

Mirrors the GTAP coverage matrix (scripts/gtap/coverage_matrix.py) at PEP's much
smaller scale: ONE dataset (pep2), no ifSUB, no CD multi-period `check` stage. The
PEP port (pep_pyomo) is validated on four axes, all against a GAMS reference solved
by the SAME engine so the solver tolerance cancels (the NLP-vs-NLP / MCP-vs-MCP idea):

  * "nlp"    — Pyomo NLP (IPOPT, raw model) vs the GAMS CNS reference (Results.gdx,
               val* layout). Scenario `base`.
  * "mcp"    — Pyomo MCP (PATH) vs the GAMS-NATIVE MCP reference (Results_mcp.gdx /
               Results_mcp_sim1.gdx, raw-symbol layout). Scenarios `base` and `sim1`
               (the −25% export-tax shock, `ttix.fx=ttixO*0.75`).
  * "mirror" — Pyomo NLP vs Pyomo MCP (both solved from the same benchmark). Proves
               the two forms land on the identical point. Scenario `base`.
  * "variant"— the objdef variant (dummy `OBJDEF: OBJ==0` objective) vs the base
               variant. Scenario `base`; both nlp and mcp forms.

`match_note` is the MEASURED snapshot for humans (a live gate re-measures it — it is
not asserted from here). `gate` names the pytest that enforces the row. `ci_status`:
`local` needs PATH+GAMS (run by hand / the local gate); PEP has no solver-free CI gate
(unlike GTAP's .nl coefficient gate) because the reference is a solved GDX.

`scripts/pep/gen_pep_coverage_doc.py` renders docs/site/guide/pep_coverage_matrix.md;
test_pep_coverage_matrix.py keeps the committed doc in sync.
"""
from __future__ import annotations

from dataclasses import dataclass

CI_STATUSES = {"local"}
KINDS = {"nlp", "mcp", "mirror", "variant"}
FORMS = {"nlp", "mcp", "both"}


@dataclass(frozen=True)
class Row:
    kind: str          # "nlp" | "mcp" | "mirror" | "variant"
    scenario: str      # "base" | "sim1"
    form: str          # "nlp" | "mcp" | "both" (which Pyomo form(s) the row covers)
    cells: int         # comparable cells at parity (measured)
    match_note: str    # measured snapshot, e.g. "100% (317 cells)"
    gate: str          # the pytest that enforces this row
    ci_status: str     # "local"
    ref: str           # reference provenance


DATASET = "pep2"

ROWS: list[Row] = [
    # --- NLP form vs the GAMS CNS reference ---
    Row("nlp", "base", "nlp", 317, "100%", "phase1_nlp.py --form nlp / test_faithful_at_benchmark",
        "local", "Results.gdx (GAMS CNS, val*)"),
    # --- MCP form vs the GAMS-native MCP reference (base + the SIM1 shock) ---
    Row("mcp", "base", "mcp", 367, "100%", "test_mcp_matches_gams_native_mcp",
        "local", "Results_mcp.gdx (GAMS /ALL/ MCP)"),
    Row("mcp", "sim1", "mcp", 314, "100% (GDP_BP 46707→46748.2)", "test_mcp_sim1_shock_matches_gams",
        "local", "Results_mcp_sim1.gdx (GAMS MCP, ttix·0.75)"),
    # --- NLP↔MCP mirror (both Pyomo forms land on the same point) ---
    Row("mirror", "base", "both", 466, "100%", "test_nlp_mcp_mirror",
        "local", "self (NLP vs MCP, LEON excl)"),
    # --- objdef variant parity ---
    Row("variant", "base", "nlp", 467, "100% (== base-NLP)", "test_objdef_variant_equals_base_nlp",
        "local", "self (objdef vs base)"),
    Row("variant", "base", "mcp", 358, "square, code=1", "test_objdef_mcp_is_square_and_solves",
        "local", "self (objdef+MCP squareness)"),
]


def nlp_rows() -> list[Row]:
    return [r for r in ROWS if r.kind == "nlp"]


def mcp_rows() -> list[Row]:
    return [r for r in ROWS if r.kind == "mcp"]


def mirror_rows() -> list[Row]:
    return [r for r in ROWS if r.kind == "mirror"]


def variant_rows() -> list[Row]:
    return [r for r in ROWS if r.kind == "variant"]


def _validate() -> None:
    """Import-time schema invariants — fail fast on a malformed matrix."""
    assert ROWS, "matrix must not be empty"
    for r in ROWS:
        assert r.kind in KINDS, f"bad kind: {r}"
        assert r.form in FORMS, f"bad form: {r}"
        assert r.scenario in {"base", "sim1"}, f"bad scenario: {r}"
        assert r.ci_status in CI_STATUSES, f"bad ci_status: {r}"
        assert r.cells > 0, f"cells must be positive: {r}"
        assert r.match_note and r.gate and r.ref, f"missing metadata: {r}"
    # the helper partition must cover ROWS exactly (no row lost / double-counted)
    covered = set(nlp_rows()) | set(mcp_rows()) | set(mirror_rows()) | set(variant_rows())
    assert covered == set(ROWS), "helper partition does not cover ROWS"


_validate()
