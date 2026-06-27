"""KKT/marginals cascade layer: read equation duals (.m) from the local reference
GDX and check complementarity against Python's closure for each paired variable."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from equilibria.babel.gdx.reader import read_gdx, read_equation_values
from cascade_classify import LayerResult, CONTINUE

KKT_READER = "pure-python"
_TOL = 1e-6

# Equation -> paired variable (MCP). Reuse the documented pairs (drift_test._PAIR),
# keyed by equation name. Start with the families present in the ref GDX.
EQ_VAR_PAIRS: dict[str, str] = {
    "arenteq": "arent",
    "apeeq": "ape",
}


def read_marginals(gdx: Path, eq: str, period: str) -> dict[tuple, float]:
    data = read_gdx(str(gdx))
    raw = read_equation_values(data, eq)
    out: dict[tuple, float] = {}
    for key, attrs in raw.items():
        if period not in key:
            continue
        stripped = tuple(k for k in key if k != period)
        out[stripped] = float(attrs["marginal"])
    return out


def kkt_layer(dataset: str, period: str, gdx: Path) -> LayerResult:
    """Complementarity check: GAMS binding (|m|>tol) vs Python's paired-var state.

    MVP: GAMS-side only — report equations the reference reports binding, as the
    KKT signal no other layer surfaces. The Python-side closure comparison is
    deferred; the layer reports the binding set as info (clean) rather than a
    false dirty."""
    binding = []
    try:
        for eq in EQ_VAR_PAIRS:
            for idx, m in read_marginals(gdx, eq, period).items():
                if abs(m) > _TOL:
                    binding.append((eq, idx, m))
    except Exception as exc:  # reader failure must surface, not crash the sweep
        return LayerResult("kkt_marginals", "error", "exception",
                           f"KKT read failed: {exc}", CONTINUE, 2, {})
    headline = (f"{len(binding)} equation(s) binding in GAMS (|m|>{_TOL:g}) at "
                f"{period}; reader={KKT_READER}")
    # MVP verdict: surfacing the binding set is informational -> clean (no false
    # dirty until the Python-side complementarity comparison lands).
    viols = [{"entity": eq, "index": list(idx), "metric": "marginal", "value": m}
             for eq, idx, m in binding]
    return LayerResult("kkt_marginals", "clean", None, headline, CONTINUE, 0,
                       {"violations": viols, "reader": KKT_READER})
