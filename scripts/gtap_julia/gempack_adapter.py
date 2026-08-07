"""Adapt a Julia GTAPv7 base+shock solution (the CSVs dumped by run_from_csv.jl)
into an object the GEMPACK parity gate's ``_measure_pp`` can read unchanged.

``_measure_pp`` accesses ``getattr(m, pyname)[(*key, "base"/"shock")]`` where ``pyname``
is a block-model Var basename (``xda``, ``xw``, …, from ``Q_TO_VAR``) and ``key`` is the
GEMPACK index tuple already put in Python order by ``Q_TO_VAR[gvar]["reorder"]``.

The Julia model uses GEMPACK-style names (``qfd``, ``qxs``, …) indexed by set-member
name in the model's native axis order (e.g. ``qfd[comm, acts, reg]``). This shim:
  - loads both CSVs into ``{var: {index_tuple: value}}``,
  - maps each block Var name back to its Julia var (via the inverse of Q_TO_VAR["var"]),
  - re-orders each Julia index with the SAME reorder lambda the gate uses on GEMPACK keys,
so ``m.xda[(usa, rice, rice, "base")]`` resolves to Julia ``qfd[rice, rice, usa]`` at base.

Usage:
    from gtap_julia.gempack_adapter import JuliaSolutionModel
    m = JuliaSolutionModel(base_csv, shock_csv)
    within, med = _measure_pp(m, sl4dump_har)   # gate's own measurer, unchanged
"""
from __future__ import annotations

import sys
from pathlib import Path

# Q_TO_VAR lives in scripts/gtap; import it so the block-name↔julia-name map and the
# per-var reorder lambdas are the SINGLE source of truth (no duplication/drift).
_GTAP_SCRIPTS = Path(__file__).resolve().parents[1] / "gtap"
if str(_GTAP_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_GTAP_SCRIPTS))
from gempack_reference import Q_TO_VAR  # noqa: E402

# block-Var basename (what the gate reads) -> Julia var name (what the CSV holds).
# GEMPACK var == Julia var here (both qfd/qxs/…), so invert Q_TO_VAR[g]["var"].
_BLOCK_TO_JULIA: dict[str, str] = {spec["var"]: gvar for gvar, spec in Q_TO_VAR.items()}
_REORDER: dict[str, object] = {spec["var"]: spec["reorder"] for spec in Q_TO_VAR.values()}


def _load_csv(path: Path) -> dict[str, dict[tuple[str, ...], float]]:
    """Parse a Julia dump CSV -> {var: {(idx…): value}}. Skips NaN and SET_ rows."""
    out: dict[str, dict[tuple[str, ...], float]] = {}
    for line in Path(path).read_text().splitlines():
        if not line.strip() or line.startswith("SET_"):
            continue
        parts = line.split(",")
        name, val_s = parts[0], parts[-1]
        try:
            val = float(val_s)
        except ValueError:
            continue
        if val != val:  # NaN (masked/inactive cell) — leave it absent
            continue
        idx = tuple(parts[1:-1])
        out.setdefault(name, {})[idx] = val
    return out


def _lc(key: tuple) -> tuple:
    """Lower-case every string element of an index tuple. GEMPACK keys are CamelCase
    (('USA','Rice')); run_from_csv.jl lower-cases set members (('usa','rice')). Match
    case-insensitively so the gate's keys resolve against Julia cells."""
    return tuple(x.lower() if isinstance(x, str) else x for x in key)


class _VarView:
    """Mimics a Pyomo Var: ``self[(*key, stage)]`` -> float, KeyError if absent.

    ``key`` arrives already reordered to Python order (the gate reorders GEMPACK keys
    before indexing). We stored Julia cells reordered the SAME way and lower-cased, so
    lookup lower-cases the incoming key to match.
    """

    def __init__(self, base: dict, shock: dict):
        self._stage = {"base": base, "shock": shock}

    def __getitem__(self, full_key):
        *key, stage = full_key
        d = self._stage.get(stage)
        if d is None:
            raise KeyError(stage)
        return d[_lc(tuple(key))]


class JuliaSolutionModel:
    """Exposes ``m.xda``, ``m.xw``, … (block names) backed by Julia base+shock CSVs."""

    def __init__(self, base_csv: str | Path, shock_csv: str | Path):
        base_raw = _load_csv(Path(base_csv))
        shock_raw = _load_csv(Path(shock_csv))
        for block_name, jname in _BLOCK_TO_JULIA.items():
            reorder = _REORDER[block_name]
            b = {_lc(reorder(k)): v for k, v in base_raw.get(jname, {}).items()}
            s = {_lc(reorder(k)): v for k, v in shock_raw.get(jname, {}).items()}
            if b or s:
                setattr(self, block_name, _VarView(b, s))
