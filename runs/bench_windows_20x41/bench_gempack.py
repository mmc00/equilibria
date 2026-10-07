"""Cronometra GEMPACK (RunGTAP gtapv7.exe) en el GTAP 20x41 (tm +10%, capFix).

    uv run python runs/bench_windows_20x41/bench_gempack.py --reps 3

Genera el ``.cmf`` con ``scripts/gtap/run_gempack_matrix.prepare`` y RORDELTA=0
(capFix), que es el experimento del fixture
``tests/fixtures/gtap7_gempack/updated_gtap7_20x41_tm10_s8-16-32_capfix.har``:
cierre estandar, Gragg 8/16/32. Ojo: el ``tm10.cmf`` y el ``noswap.cmf`` de
``runs/gempack_20x41_validation/`` son OTROS cierres (swap de dpsave / capFlex).

GEMPACK usa OpenMP (el log dice "OPENMP number of threads: N") y equilibria corre
en un hilo, asi que se mide dos veces: con los hilos por defecto y con
``OMP_NUM_THREADS=1``. Escribe ``results/gempack.json``.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from _bench import DATASET, ROOT, save  # noqa: E402

sys.path.insert(0, str(ROOT / "scripts" / "gtap"))
import run_gempack_matrix as gm  # noqa: E402

_ELAPSED = re.compile(
    r"Total elapsed time is:\s*(?:(\d+)\s*minutes?,?\s*)?(?:(\d+)\s*seconds?)?",
    re.IGNORECASE,
)


def _parse(text: str) -> dict:
    threads = re.search(r"OPENMP number of threads:\s*(\d+)", text)
    resid = next(
        (
            ln.strip()
            for ln in text.splitlines()
            if "maximum residual ratio" in ln.lower()
        ),
        "",
    )
    elapsed = None
    for mm in _ELAPSED.finditer(text):
        if mm.group(1) or mm.group(2):
            elapsed = int(mm.group(1) or 0) * 60 + int(mm.group(2) or 0)
    return {
        "ok": "completed without error" in text.lower(),
        "openmp_threads": int(threads.group(1)) if threads else None,
        "max_residual_ratio": resid,
        "gempack_total_elapsed_s": elapsed,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument(
        "--gtapv7",
        type=Path,
        default=Path(os.environ.get("GTAPV7", str(gm.DEFAULT_GTAPV7))),
    )
    args = ap.parse_args()
    if not args.gtapv7.exists():
        raise SystemExit(f"no encuentro gtapv7.exe en {args.gtapv7} (usar --gtapv7)")

    run_dir, tag = gm.prepare(DATASET, 10.0, "8 16 32", rordelta=0)
    variants = {"default_threads": None, "one_thread": "1"}
    out: dict[str, list] = {}
    for name, omp in variants.items():
        out[name] = []
        for i in range(args.reps):
            env = dict(os.environ)
            if omp is None:
                env.pop("OMP_NUM_THREADS", None)
            else:
                env["OMP_NUM_THREADS"] = omp
            console = run_dir / f"bench_{name}_{i + 1}.txt"
            print(f"[{name} {i + 1}/{args.reps}] gtapv7 -cmf {tag}.cmf", flush=True)
            t0 = time.perf_counter()
            with console.open("w", encoding="utf-8", errors="replace") as fh:
                subprocess.run(
                    [str(args.gtapv7), "-cmf", f"{tag}.cmf"],
                    cwd=run_dir,
                    env=env,
                    stdout=fh,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
            wall = time.perf_counter() - t0
            r = _parse(console.read_text(encoding="utf-8", errors="replace"))
            r["wall_process"] = wall
            r["console"] = str(console.relative_to(ROOT))
            out[name].append(r)
            print(
                f"  ok={r['ok']} hilos={r['openmp_threads']} "
                f"wall={wall / 60:.2f} min  {r['max_residual_ratio']}"
            )
    save("gempack", {"tool": "gempack", "cmf": f"{tag}.cmf", "runs": out})
    return 0 if all(r["ok"] for rs in out.values() for r in rs) else 1


if __name__ == "__main__":
    raise SystemExit(main())
