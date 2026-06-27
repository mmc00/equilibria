"""Pure config resolution for the cascade orchestrator: period->scenario and the
reference-GDX choice with a NON-SILENT fallback. No subprocess, no model build."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[2]
DURABLE_REFS = Path("/Users/marmol/proyectos2/equilibria_refs")

SCENARIO_BY_PERIOD: dict[str, dict[str, str]] = {
    "gtap7": {"check": "altertax_check", "shock": "altertax_shock"},
    "bundle": {"base": "baseline", "shock": "shock_tm10"},
}


def family(dataset: str) -> str:
    return "gtap7" if dataset.startswith("gtap7_") else "bundle"


def scenario_for(dataset: str, period: str) -> str:
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


def resolve_ref_gdx(dataset: str) -> GdxResolution:
    durable = DURABLE_REFS / f"{dataset}_altertax_cd" / "out_altertax_ifsub0.gdx"
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
