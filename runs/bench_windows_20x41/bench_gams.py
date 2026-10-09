"""Cronometra GAMS (GTAP v7 de GAMS + PATH) en el GTAP 20x41 (tm +10%, capFix).

    uv run python runs/bench_windows_20x41/bench_gams.py --reps 1

``gams/comp_gtap7_20x41_gtap_shock_ifsub1.gms.gz`` es el bundle de
``scripts/gtap/build_gtap7_pure_local_bundle.py --dataset gtap7_20x41 --ifsub 1``
(comp.gms: savfFlag capFix, ifSUB 1; datos en linea; shock tm +10% en el
periodo shock). Se commitea generado porque necesita las fuentes de GAMS de
cge_babel, que no estan en el repo. Resuelve check y shock (no base), igual que
equilibria con ``base_calibrated=True``.

Antes de correr se sube ``iterlim = 1000`` a 1.000.000: con 1000, PATH corta
por iteraciones y GAMS lo reporta como infactible (fue la causa del falso fallo
de ME9D). Eso queda anotado en el resultado.

OJO: el bundle .gms.gz es FIJO (20x41). Este script IGNORA
EQUILIBRIA_BENCH_DATASET, asi que cuando el resto del grid mide otro
agregado, la fila de GAMS no es del mismo experimento. Generar el bundle del
dataset con build_gtap7_pure_local_bundle.py --dataset <ds> si se lo necesita.

Nunca se ha resuelto el 20x41 en GAMS: con licencia demo/community no entra
(~400.000 filas por periodo) y un intento en NEOS (ifSUB=0) termino "Locally
Infeasible". Si falla, se registra el motivo en vez de abortar. Escribe
``results/gams.json`` y deja el ``.lst`` en ``gams_work/``.
"""

from __future__ import annotations

import argparse
import gzip
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from _bench import ROOT, save  # noqa: E402

GMS_GZ = HERE / "gams" / "comp_gtap7_20x41_gtap_shock_ifsub1.gms.gz"
WORK = HERE / "gams_work"
ITERLIM = 1_000_000


def _prepare() -> Path:
    WORK.mkdir(exist_ok=True)
    text = gzip.decompress(GMS_GZ.read_bytes()).decode("utf-8")
    patched, n = re.subn(r"iterlim\s*=\s*1000\b", f"iterlim = {ITERLIM}", text)
    if n == 0:
        raise SystemExit("no encontre 'iterlim = 1000' en el .gms")
    gms = WORK / GMS_GZ.name.removesuffix(".gz")
    gms.write_text(patched, encoding="utf-8")
    return gms


def _parse_lst(lst: Path) -> dict:
    text = lst.read_text(errors="replace") if lst.exists() else ""
    solves = []
    for block in text.split("S O L V E      S U M M A R Y")[1:]:

        def grab(pat: str) -> str:
            mm = re.search(pat, block)
            return mm.group(1).strip() if mm else ""

        solves.append(
            {
                "solver_status": grab(r"\*\*\*\* SOLVER STATUS\s+(.+)"),
                "model_status": grab(r"\*\*\*\* MODEL STATUS\s+(.+)"),
                "resource_s": grab(r"RESOURCE USAGE, LIMIT\s+([\d.]+)"),
                "iterations": grab(r"ITERATION COUNT, LIMIT\s+(\d+)"),
            }
        )
    lic = [
        ln.strip()
        for ln in text.splitlines()
        if re.search(r"licen[cs]|demo|size limit|exceeds", ln, re.IGNORECASE)
    ][:10]
    return {"solves": solves, "license_lines": lic}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--gams", default=shutil.which("gams") or "gams")
    args = ap.parse_args()

    gms = _prepare()
    runs = []
    for i in range(args.reps):
        lst = WORK / f"bench_{i + 1}.lst"
        log = WORK / f"bench_{i + 1}.log"
        print(f"[gams {i + 1}/{args.reps}] {gms.name}", flush=True)
        t0 = time.perf_counter()
        with log.open("w", encoding="utf-8", errors="replace") as fh:
            proc = subprocess.run(
                [args.gams, gms.name, f"o={lst.name}", "lo=3", f"curDir={WORK}"],
                cwd=WORK,
                stdout=fh,
                stderr=subprocess.STDOUT,
                check=False,
            )
        wall = time.perf_counter() - t0
        r = _parse_lst(lst)
        r["gams_exit"] = proc.returncode
        r["wall_process"] = wall
        r["ok"] = (
            proc.returncode == 0
            and len(r["solves"]) >= 2
            and all("Normal Completion" in s["solver_status"] for s in r["solves"])
            and all(s["model_status"].startswith(("1 ", "2 ")) for s in r["solves"])
        )
        runs.append(r)
        print(f"  exit={proc.returncode} ok={r['ok']} wall={wall / 60:.2f} min")
        for s in r["solves"]:
            print(
                f"    {s['model_status']} | {s['solver_status']} | {s['resource_s']} s"
            )
        for ln in r["license_lines"]:
            print(f"    licencia: {ln}")
    save("gams", {"tool": "gams", "iterlim": ITERLIM, "runs": runs})
    return 0 if all(r["ok"] for r in runs) else 1


if __name__ == "__main__":
    raise SystemExit(main())
