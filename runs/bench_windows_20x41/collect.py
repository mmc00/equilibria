"""Junta ``results/*.json`` en ``results/RESULTADOS.md``.

    uv run python runs/bench_windows_20x41/collect.py

Usa la mediana de las repeticiones validas. Una corrida que no convergio no se
promedia: aparece como fallo.
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from _bench import EXPERIMENT, RESULTS  # noqa: E402


def _load(name: str) -> dict | None:
    f = RESULTS / f"{name}.json"
    return json.loads(f.read_text(encoding="utf-8")) if f.exists() else None


def _min(xs: list[float]) -> str:
    return f"{statistics.median(xs) / 60:.2f}" if xs else "—"


def _rows() -> list[str]:
    rows = []
    for mode in ("warm", "cold"):
        d = _load(f"equilibria_{mode}")
        if not d:
            continue
        ok = [r for r in d["runs"] if r.get("ok")]
        rows.append(
            f"| equilibria ({mode}) | {len(ok)}/{len(d['runs'])} | "
            f"{_min([r['seconds']['build'] + r['seconds']['load_har'] for r in ok])} | "
            f"{_min([r['seconds']['solve_check_shock'] for r in ok])} | "
            f"{_min([r['wall_process'] for r in ok])} | 1 hilo (Python) |"
        )
    g = _load("gempack")
    if g:
        for variant, runs in g["runs"].items():
            ok = [r for r in runs if r.get("ok")]
            hilos = {r.get("openmp_threads") for r in ok}
            rows.append(
                f"| GEMPACK ({variant}) | {len(ok)}/{len(runs)} | — | — | "
                f"{_min([r['wall_process'] for r in ok])} | OpenMP {sorted(hilos)} |"
            )
    a = _load("gams")
    if a:
        ok = [r for r in a["runs"] if r.get("ok")]
        solve = [sum(float(s["resource_s"] or 0) for s in r["solves"]) for r in ok]
        rows.append(
            f"| GAMS + PATH | {len(ok)}/{len(a['runs'])} | — | {_min(solve)} | "
            f"{_min([r['wall_process'] for r in ok])} | |"
        )
    return rows


def main() -> int:
    env = _load("env")
    m = (env or {}).get("machine", {})
    lines = [
        "# Benchmark GTAP 20x41 — misma maquina",
        "",
        f"- Maquina: {m.get('cpu')} · {m.get('logical_cpus')} hilos logicos · "
        f"{m.get('ram_gb')} GB RAM · {m.get('os')}",
        f"- Experimento: {EXPERIMENT['dataset']}, {EXPERIMENT['shock']}, "
        f"{EXPERIMENT['closure']}, ifSUB={EXPERIMENT['ifsub']}",
        "",
        "Minutos, mediana de las corridas que convergieron.",
        "",
        "| herramienta | convergio | carga+build | solve | wall total | paralelismo |",
        "|---|---|---|---|---|---|",
        *_rows(),
        "",
        "## Lo que NO es igual entre las tres (leer antes de citar)",
        "",
        "- **Metodo:** equilibria y GAMS resuelven el sistema no lineal en niveles",
        "  (PATH). GEMPACK resuelve el shock linealizado por Gragg 8/16/32 con",
        "  extrapolacion: es otra aproximacion, con su propio error.",
        "- **Periodos:** equilibria y GAMS resuelven check y shock; GEMPACK solo el",
        "  shock (parte del benchmark).",
        "- **Arranque:** `equilibria (warm)` siembra con una solucion previa del",
        "  20x41 (como la medicion de ~7 min en la Mac); `cold`, GEMPACK y GAMS",
        "  arrancan del benchmark. La comparacion justa es `cold`.",
        "- **Hilos:** GEMPACK usa OpenMP; equilibria, un hilo. Por eso GEMPACK se",
        "  mide tambien con `OMP_NUM_THREADS=1`.",
        "- **GAMS:** `iterlim` subido de 1000 a 1.000.000 (con 1000 PATH corta y",
        "  GAMS lo reporta como infactible).",
    ]
    out = RESULTS / "RESULTADOS.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(out.read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
