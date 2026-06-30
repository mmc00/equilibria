"""Pure config resolution for the cascade orchestrator: period->scenario and the
reference-GDX choice with a NON-SILENT fallback. No subprocess, no model build."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[2]
DURABLE_REFS = Path("/Users/marmol/proyectos2/equilibria_refs")
# Pure-gtap (real-CES, non-altertax) solved refs live committed under the .nl
# fixture tree, one per ifSUB mode (distinct from the altertax out_altertax_*).
GTAP_PURE_FIXTURES = ROOT / "tests" / "fixtures" / "gtap7"

SCENARIO_BY_PERIOD: dict[str, dict[str, str]] = {
    "gtap7": {"check": "altertax_check", "shock": "altertax_shock"},
    "bundle": {"base": "baseline", "shock": "shock_tm10"},
}
# In gtap-pure mode the closure is the real-CES base closure (NOT the altertax CD
# compStat); the scenario names reflect that so each tool builds the right model.
SCENARIO_BY_PERIOD_GTAP_PURE: dict[str, str] = {
    "check": "gtap_check", "shock": "gtap_shock",
}


def family(dataset: str) -> str:
    return "gtap7" if dataset.startswith("gtap7_") else "bundle"


def scenario_for(dataset: str, period: str, mode: str = "altertax") -> str:
    if mode == "gtap":
        return SCENARIO_BY_PERIOD_GTAP_PURE[period]
    return SCENARIO_BY_PERIOD[family(dataset)][period]


def resolve_periods(dataset: str, requested: list[str]) -> tuple[list[str], list[str]]:
    have = list(SCENARIO_BY_PERIOD[family(dataset)].keys())
    available = [p for p in requested if p in have]
    dropped = [p for p in requested if p not in have]
    return available, dropped


@dataclass
class GdxResolution:
    path: Optional[Path]
    source: str   # "durable" | "adapter_output" | "missing"
    note: str
    usable: bool


def _ok(p: Path) -> bool:
    return p.exists() and p.stat().st_size > 0


def resolve_ref_gdx(dataset: str, *, mode: str = "altertax", ifsub: int = 0,
                    explicit: Optional[Path] = None) -> GdxResolution:
    """Resolve the reference GDX for the cascade.

    explicit  : an exact GDX path the caller passes (--ref); wins if it exists.
    mode=gtap : the pure-gtap (real-CES) solved ref under the committed fixture
                tree, out_gtap_shock_ifsub{0,1}.gdx (one per ifSUB).
    mode=altertax (default) : the durable altertax CD ref (ifsub0 only, the
                cascade's original target) with the gitignored adapter fallback.
    """
    if explicit is not None:
        if _ok(explicit):
            return GdxResolution(explicit, "explicit", f"explicit ref {explicit}", True)
        return GdxResolution(None, "missing",
                             f"explicit ref does not exist/non-empty: {explicit}", False)

    if mode == "gtap":
        suffix = "ifsub1" if ifsub else "ifsub0"
        ref = GTAP_PURE_FIXTURES / dataset / f"out_gtap_shock_{suffix}.gdx"
        if _ok(ref):
            return GdxResolution(ref, "gtap_pure_fixture",
                                 f"pure-gtap solved ref {ref}", True)
        return GdxResolution(None, "missing",
                             f"NO usable pure-gtap ref GDX at {ref} (only gtap7_3x3 "
                             f"has these fixtures today)", False)

    durable = DURABLE_REFS / f"{dataset}_altertax_cd" / f"out_altertax_ifsub{ifsub}.gdx"
    if _ok(durable):
        return GdxResolution(durable, "durable", f"durable ref {durable}", True)
    adapter_out = ROOT / "output" / f"{dataset}_altertax_neos_bundle" / "out_local.gdx"
    if _ok(adapter_out):
        return GdxResolution(
            adapter_out, "adapter_output",
            f"FALLBACK: durable ref absent at {durable}; using gitignored "
            f"{adapter_out} (may be stale — verify before trusting dirty/clean)",
            True)
    return GdxResolution(
        None, "missing",
        f"NO usable ref GDX: neither {durable} nor {adapter_out} exists/non-empty",
        False)
