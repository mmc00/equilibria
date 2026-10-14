"""Benchmark del 20x41 contra la linea base: correr antes de mergear un PR de rendimiento.

Los conteos de tests/templates/gtap/test_perf_budget.py atrapan el trabajo repetido
en segundos; esto mide el tiempo y la memoria de verdad, que solo valen en una
maquina limpia. Por eso la corrida se ANULA (exit 2) si hubo swap: con swap la Mac
dio 15-22 min para un codigo que limpio tarda 8-12 (octubre 2026).

    <python con pymumps> scripts/gtap/perf_budget_20x41.py            # mide y compara
    <python con pymumps> scripts/gtap/perf_budget_20x41.py --update   # nueva linea base

Ruta medida: Newton-TR + MUMPS, base calibrada, en frio (sin seed cache ni cache del
modelo), la misma de las cifras del paper. Exit 0 = dentro del presupuesto, 1 =
regresion, 2 = medicion invalida (swap, falta pymumps o el modelo no convergio).

Necesita pymumps, que viene de conda-forge (PyPI solo trae el fuente y pide un
MUMPS del sistema; ver ROADMAP, marca needs_mumps). Sin el, Newton-TR registra "MUMPS
FAILED" en cada iteracion, no converge en 300 y la corrida muere minutos despues
con un error que no nombra la causa (octubre 2026, venv del proyecto).
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASELINE = Path(__file__).with_name("perf_baseline_20x41.json")
TOLERANCIA = 0.10  # +10 % de tiempo o de memoria es regresion
SWAP_MAX_MB = 100.0  # mas swap que esto y la medicion no vale

NEWTON_MUMPS = {
    "EQUILIBRIA_GTAP_SOLVE_NLP": "1",
    "EQUILIBRIA_GTAP_SOLVER": "scipy_newton_tr",
    "EQUILIBRIA_GTAP_TR_LINSOLVE": "mumps",
    "EQUILIBRIA_GTAP_NLP_NO_JACSCALE": "1",
    "EQUILIBRIA_GTAP_TR_GATE": "1",
    "EQUILIBRIA_GTAP_TR_FTOL": "1e-7",
    "EQUILIBRIA_GTAP_SCIPY_MAXITER": "300",
    "EQUILIBRIA_GTAP_TR_DELTA0": "10.0",
    "EQUILIBRIA_GTAP_GMIN": "1e-9",
    "EQUILIBRIA_GTAP_TR_RELTOL": "1e-6",
    "EQUILIBRIA_SEED_CACHE_DISABLE": "1",
    "GTAP_GATES_SKIP": "1",
}

_HIJO = """
import json, sys, time
sys.path[:0] = [{src!r}, {scripts!r}, {tests!r}, {root!r}]
from templates.gtap.test_gtap7_gempack_parity import _solve_shock
t = time.perf_counter()
_, code = _solve_shock("gtap7_20x41", 1, "capFix", solve_nlp=True)
print("PERF " + json.dumps({{"code": code, "wall_min": (time.perf_counter() - t) / 60}}))
"""


def veredicto(medido: dict, base: dict, swap_mb: float) -> tuple[int, str]:
    """(exit code, mensaje) de comparar una medicion con la linea base."""
    if swap_mb > SWAP_MAX_MB:
        return 2, f"medicion invalida: {swap_mb:.0f} MB de swap (max {SWAP_MAX_MB:.0f})"
    if medido.get("code") != 1:
        return (
            2,
            f"medicion invalida: el modelo no convergio (code={medido.get('code')})",
        )
    malas = []
    for k in ("wall_min", "peak_gb"):
        techo = base[k] * (1 + TOLERANCIA)
        if medido[k] > techo:
            malas.append(f"{k} {medido[k]:.2f} > {techo:.2f} (base {base[k]:.2f})")
    if malas:
        return 1, "REGRESION: " + "; ".join(malas)
    return 0, (
        f"OK: {medido['wall_min']:.2f} min, {medido['peak_gb']:.2f} GB "
        f"(base {base['wall_min']:.2f} min, {base['peak_gb']:.2f} GB)"
    )


def _swap_in_mb() -> float:
    """Paginas traidas desde swap hasta ahora, en MB (macOS o Linux)."""
    if platform.system() == "Darwin":
        out = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
        tam = int(re.search(r"page size of (\d+)", out).group(1))
        n = int(re.search(r"Swapins:\s+(\d+)", out).group(1))
        return n * tam / 1e6
    texto = Path("/proc/vmstat").read_text()
    return int(re.search(r"pswpin (\d+)", texto).group(1)) * 4096 / 1e6


def _falta_mumps() -> str | None:
    """Mensaje si el interprete de la medicion no puede importar pymumps."""
    p = subprocess.run([sys.executable, "-c", "import mumps"], capture_output=True)
    if p.returncode == 0:
        return None
    return (
        f"medicion invalida: {sys.executable} no tiene pymumps "
        "(conda install -c conda-forge pymumps; o correr con un python que lo tenga)"
    )


def medir() -> tuple[dict, float]:
    codigo = _HIJO.format(
        src=str(ROOT / "src"),
        scripts=str(ROOT / "scripts" / "gtap"),
        tests=str(ROOT / "tests"),
        root=str(ROOT),
    )
    env = {**os.environ, **NEWTON_MUMPS}
    env.pop("EQUILIBRIA_GTAP_MODEL_CACHE", None)
    cmd = [sys.executable, "-c", codigo]
    darwin = platform.system() == "Darwin"
    if darwin:  # el pico de footprint (incluye lo comprimido) solo lo da time -l
        cmd = ["/usr/bin/time", "-l", *cmd]
    swap0 = _swap_in_mb()
    p = subprocess.run(cmd, env=env, capture_output=True, text=True)
    swap = _swap_in_mb() - swap0
    linea = next((l for l in p.stdout.splitlines() if l.startswith("PERF ")), None)
    if linea is None:
        sys.stderr.write(p.stderr[-4000:])
        return {"code": None, "wall_min": 0.0, "peak_gb": 0.0}, swap
    medido = json.loads(linea[5:])
    if darwin:
        pico = re.search(r"(\d+)\s+peak memory footprint", p.stderr)
        medido["peak_gb"] = int(pico.group(1)) / 1e9
    else:
        import resource

        medido["peak_gb"] = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1e6
    return medido, swap


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--update", action="store_true", help="guardar como linea base")
    args = ap.parse_args()
    if falta := _falta_mumps():
        print(falta)
        return 2
    medido, swap = medir()
    # Siempre, aunque la medicion se anule: con swap el numero sigue orientando.
    print(
        f"medido: {medido['wall_min']:.2f} min, {medido['peak_gb']:.2f} GB, "
        f"swap {swap:.0f} MB, code={medido.get('code')}"
    )
    if args.update:
        rc, msg = veredicto(medido, {"wall_min": 1e9, "peak_gb": 1e9}, swap)
        if rc != 0:
            print(msg)
            return rc
        commit = subprocess.run(
            ["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
        ).stdout.strip()
        base = {
            "wall_min": round(medido["wall_min"], 2),
            "peak_gb": round(medido["peak_gb"], 2),
            "commit": commit,
            "maquina": platform.platform(),
        }
        BASELINE.write_text(json.dumps(base, indent=2) + "\n")
        print(f"linea base nueva: {base}")
        return 0
    rc, msg = veredicto(medido, json.loads(BASELINE.read_text()), swap)
    print(msg)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
