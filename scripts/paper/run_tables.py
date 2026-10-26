"""Corre los resultados del paper GTAP (JGEA), un paso por tabla, y deja la salida en --out.

Cada paso es el comando de la Seccion 9 del paper. Los que necesitan GAMS (Tabla 1, Tabla 3)
o tardan mucho (20x41) se pueden saltar con --skip. Uso:

    PYTHONPATH=src:. python scripts/paper/run_tables.py --out paper_out
    PYTHONPATH=src:. python scripts/paper/run_tables.py --out paper_out --only table2,table3a
    PYTHONPATH=src:. python scripts/paper/run_tables.py --dry-run --out paper_out

Las referencias GEMPACK vienen en tests/fixtures/gtap7_gempack/ (se generan en Windows con
scripts/windows/corridas_paper.ps1 -Pasos 4,5,7 -Cierre capflex); las de GAMS de la Tabla 3
las genera el paso table3_refs (scripts/gtap/gen_burfisher_gams.py).
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FIX = "tests/fixtures/gtap7_gempack"
TABLE2 = {  # dataset -> solver de la Tabla 2
    "3x3": "path",
    "3x4": "path",
    "5x5": "path",
    "10x7": "path",
    "15x10": "path",
    "20x41": "newton",
}


@dataclass
class Step:
    name: str
    what: str
    cmds: list[list[str]]
    files: list[str] = field(default_factory=list)  # archivos del repo que necesita


def steps(out: Path) -> list[Step]:
    py = sys.executable
    t2 = [
        [
            py, "scripts/gtap/compare_gempack_levels.py",
            "--dataset", f"gtap7_{ds}", "--closure", "capFlex", "--solver", solver,
            "--har", f"{FIX}/sl4levels6_gtap7_{ds}_tm10_s8-16-32.har",
            "--basedata", f"datasets/gtap7_{ds}/basedata.har",
            "--updated", f"{FIX}/updated_gtap7_{ds}_tm10_s8-16-32.har",
            "--save-eq", str(out / f"eq_gtap7_{ds}_capflex.json.gz"),
            "--out", str(out / f"table2_gtap7_{ds}.json"),
        ]
        for ds, solver in TABLE2.items()
    ]  # fmt: skip
    t2_files = ["scripts/gtap/compare_gempack_levels.py"] + [
        f for ds in TABLE2 for f in (
            f"{FIX}/sl4levels6_gtap7_{ds}_tm10_s8-16-32.har",
            f"{FIX}/updated_gtap7_{ds}_tm10_s8-16-32.har",
            f"datasets/gtap7_{ds}/basedata.har",
        )
    ]  # fmt: skip
    refs = out / "gams_burfisher"
    return [
        Step("table1", "Tabla 1: paridad con GAMS (MCP y NLP)",
             [[py, "scripts/gtap/run_parity_gates.py"]], ["scripts/gtap/run_parity_gates.py"]),
        Step("table2", "Tabla 2: niveles contra GEMPACK, 6 datasets, capFlex", t2, t2_files),
        Step("table3a", "Tabla 3a: ejercicios del libro contra GEMPACK (Gragg)",
             [[py, "scripts/gtap/compare_burfisher_gempack.py",
               "--gempack-dir", f"{FIX}/burfisher_gragg", "--json", str(out / "table3a.json")]],
             ["scripts/gtap/compare_burfisher_gempack.py", f"{FIX}/burfisher_gragg", "datasets/nus333"]),
        Step("table3_refs", "Referencias GAMS de la Tabla 3 (necesita GAMS)",
             [[py, "scripts/gtap/gen_burfisher_gams.py", str(refs)]],
             ["scripts/gtap/gen_burfisher_gams.py"]),
        Step("table3", "Tabla 3: ejercicios del libro contra GAMS",
             [[py, "scripts/gtap/run_burfisher.py", "--gams-dir", str(refs),
               "--json", str(out / "table3.json")]], ["scripts/gtap/run_burfisher.py"]),
        Step("tbl65b_me3c", "TBL65B y ME3C (cierres por notebook)",
             [[py, "-m", "pytest", "-q", "tests/templates/gtap/test_nus333_tbl65b_desempleo.py"]],
             ["tests/templates/gtap/test_nus333_tbl65b_desempleo.py"]),
        Step("homogeneity", "Seccion 6.4: homogeneidad",
             [[py, "-m", "pytest", "-q", "tests/templates/gtap/test_nus333_homogeneidad.py"],
              [py, "scripts/gtap/compare_burfisher_gempack.py", "--only", "GTAPV7",
               "--json", str(out / "homogeneity_gempack.json")]],
             ["tests/templates/gtap/test_nus333_homogeneidad.py"]),
        Step("model_size", "Seccion 2: tamano del modelo",
             [[py, "scripts/paper/model_size.py"]], ["scripts/paper/model_size.py", "datasets/gtap7_3x3"]),
        Step("scale_20x41", "Seccion 4.3: tiempo y memoria del 20x41 (maquina limpia, sin swap)",
             [[py, "scripts/gtap/perf_budget_20x41.py"],
              [py, "scripts/gtap/phase_times.py", "gtap7_20x41", "--out", str(out / "phases_20x41.json")]],
             ["scripts/gtap/perf_budget_20x41.py", "scripts/gtap/phase_times.py"]),
        Step("readers", "Seccion 3: lectores HAR y GDX",
             [[py, "-m", "pytest", "-q", "tests/babel"]], ["tests/babel"]),
    ]  # fmt: skip


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--out", type=Path, required=True, help="carpeta de salida")
    ap.add_argument("--only", default="", help="pasos separados por coma")
    ap.add_argument("--skip", default="", help="pasos a saltar, separados por coma")
    ap.add_argument(
        "--dry-run", action="store_true", help="imprimir los comandos sin correrlos"
    )
    a = ap.parse_args(argv)

    only = {s for s in a.only.split(",") if s}
    skip = {s for s in a.skip.split(",") if s}
    todo = [
        s for s in steps(a.out) if (not only or s.name in only) and s.name not in skip
    ]
    if not a.dry_run:
        a.out.mkdir(parents=True, exist_ok=True)
    failed = []
    for s in todo:
        print(f"== {s.name}: {s.what}", flush=True)
        for cmd in s.cmds:
            print("   " + " ".join(cmd), flush=True)
            if a.dry_run:
                continue
            log = a.out / f"{s.name}.log"
            with log.open("a") as fh:
                rc = subprocess.call(cmd, cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT)
            if rc != 0:
                print(f"   FALLO (rc={rc}), ver {log}", flush=True)
                failed.append(s.name)
                break
    if failed:
        print("pasos que fallaron: " + ", ".join(failed))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
