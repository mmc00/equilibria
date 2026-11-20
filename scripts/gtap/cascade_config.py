"""Pure config resolution for the cascade orchestrator: period->scenario and the
reference-GDX choice with a NON-SILENT fallback. No subprocess, no model build."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from equilibria.sources import find_reference

ROOT = Path(__file__).resolve().parents[2]
# Pure-gtap (real-CES, non-altertax) solved refs live committed under the .nl
# fixture tree, one per ifSUB mode (distinct from the altertax out_altertax_*).

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
    source: str   # "explicit" | "gams_reference" | "missing"
    note: str
    usable: bool


def _ok(p: Path) -> bool:
    return p.exists() and p.stat().st_size > 0


def resolve_ref_gdx(dataset: str, *, mode: str = "altertax", ifsub: int = 0,
                    explicit: Optional[Path] = None) -> GdxResolution:
    """Resolve the reference GDX for the cascade.

    explicit  : an exact GDX path the caller passes (--ref); wins if it exists.
    mode=gtap | altertax (default) : the GAMS MCP reference of that mode recorded
                under data/ (fbep=header, one per ifSUB), the one the MCP gate
                compares against. The old durable altertax CD ref (June 2026) no
                longer matches the model: its point is not a fixed point (MAR-216).
    """
    if explicit is not None:
        if _ok(explicit):
            return GdxResolution(explicit, "explicit", f"explicit ref {explicit}", True)
        return GdxResolution(None, "missing",
                             f"explicit ref does not exist/non-empty: {explicit}", False)

    folder = find_reference(dataset, "gams", form="mcp", mode=mode,
                            ifsub=ifsub, fbep="header")
    ref = None if folder is None else folder / "out.gdx"
    if ref is not None and _ok(ref):
        return GdxResolution(ref, "gams_reference",
                             f"{mode} GAMS MCP reference {ref}", True)
    return GdxResolution(None, "missing",
                         f"NO usable {mode} GAMS MCP reference for {dataset} "
                         f"ifSUB={ifsub}", False)
