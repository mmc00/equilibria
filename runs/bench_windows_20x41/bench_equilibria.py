"""Cronometra equilibria en el GTAP 20x41 (tm +10%, capFix, ifSUB=1, PATH).

    uv run python runs/bench_windows_20x41/bench_equilibria.py --mode warm --reps 3
    uv run python runs/bench_windows_20x41/bench_equilibria.py --mode cold --reps 1

Es el mismo experimento que dio los ~7,0 min en la Mac (Pedro ROADMAP, 2026-09-19):
``tests/templates/gtap/test_gtap7_gempack_parity.py::_solve_shock`` con
``savf_flag="capFix"``, ``base_calibrated=True``, sin resolver base y resolviendo
check y shock.

- ``warm``: como esa medicion, con ``ref_gdx`` = el GDX del 20x41 del repo
  (solucion previa de equilibria) para sembrar el check y el shock.
- ``cold``: sin ``ref_gdx``; arranca del benchmark, como GEMPACK y GAMS.

Cada repeticion corre en un proceso nuevo, con cache de semillas vacia y sin la
cache del modelo, para que ninguna se aproveche de la anterior. Escribe
``results/equilibria_<mode>.json``.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from _bench import DATASET, ROOT, save  # noqa: E402

GDX_WARM = ROOT / "tests/fixtures/gtap7" / DATASET / "out_gtap_shock_ifsub1.gdx"


def _path_env() -> dict[str, str]:
    """Rutas de PATH para esta plataforma.

    El driver toma por defecto los nombres de macOS (``libpath50.silicon.dylib``)
    si no se le pasan ``PATH_CAPI_LIBPATH`` / ``PATH_CAPI_LIBLUSOL``.
    """
    sys.path.insert(0, str(ROOT / "src"))
    from equilibria._local_refs import (
        path_capi_lib,
        path_capi_lib_dir,
        path_capi_lusol_names,
    )

    env: dict[str, str] = {}
    if "PATH_CAPI_LIBPATH" not in os.environ:
        env["PATH_CAPI_LIBPATH"] = str(path_capi_lib())
    if "PATH_CAPI_LIBLUSOL" not in os.environ:
        d = path_capi_lib_dir()
        lusol = next((d / n for n in path_capi_lusol_names() if (d / n).exists()), None)
        if lusol is not None:
            env["PATH_CAPI_LIBLUSOL"] = str(lusol)
    return env


def one_run(mode: str) -> dict:
    """Una corrida, cronometrada por fase. Corre dentro del proceso hijo."""
    sys.path.insert(0, str(ROOT / "src"))
    t0 = time.perf_counter()
    from equilibria.templates.gtap import GTAPParameters
    from equilibria.templates.gtap.gtap_block_model import build_block_model
    from equilibria.templates.gtap.gtap_contract import GTAPClosureConfig
    from equilibria.templates.gtap.gtap_multiperiod_driver import solve_multiperiod

    t_import = time.perf_counter()
    d = ROOT / "datasets" / DATASET
    p = GTAPParameters()
    p.load_from_har(
        basedata_path=d / "basedata.har",
        sets_path=d / "sets.har",
        default_path=d / "default.prm",
        baserate_path=d / "baserate.har",
    )
    rr = list(p.sets.r)[-1]
    ac = GTAPClosureConfig(
        name="base",
        closure_type="MCP",
        capital_mobility="sluggish",
        fix_endowments=False,
        fix_taxes=False,
        fix_technology=False,
        if_sub=True,
        savf_flag="capFix",
        numeraire="pnum",
    )
    gdx = GDX_WARM if mode == "warm" else None
    t_load = time.perf_counter()
    m, _ = build_block_model(p, p.sets, ac, rr, base_calibrated=True, ref_gdx=gdx)
    t_build = time.perf_counter()
    res = solve_multiperiod(
        m,
        p,
        ac,
        ref_gdx=gdx,
        skip_base_solve=True,
        mute_welfare=True,
        seed_from_prior=False,
        holdfix_cd=True,
        mode="gtap",
        solve_check=True,
    )
    t_solve = time.perf_counter()
    return {
        # code 1 = resuelto. Un 5 es el limite de tiempo de PATH (1 h por solve,
        # su default): paso en una Mac con 12 GB de swap, no es el modelo.
        "codes": {
            k: int(v["code"])
            for k, v in res.items()
            if isinstance(v, dict) and "code" in v
        },
        "residuals": {
            k: float(v.get("residual", float("nan")))
            for k, v in res.items()
            if isinstance(v, dict) and "code" in v
        },
        "seconds": {
            "import": t_import - t0,
            "load_har": t_load - t_import,
            "build": t_build - t_load,
            "solve_check_shock": t_solve - t_build,
            "total": t_solve - t0,
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("warm", "cold"), default="warm")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--one", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args.one:
        print("BENCH_RESULT " + json.dumps(one_run(args.mode)), flush=True)
        return 0

    if args.mode == "warm" and not GDX_WARM.exists():
        raise SystemExit(f"falta {GDX_WARM}")
    runs = []
    for i in range(args.reps):
        env = {**os.environ, **_path_env()}
        env.pop("EQUILIBRIA_GTAP_MODEL_CACHE", None)
        with tempfile.TemporaryDirectory(prefix="seedcache_") as seeds:
            env["EQUILIBRIA_SEED_CACHE"] = seeds
            print(f"[{args.mode} {i + 1}/{args.reps}] corriendo...", flush=True)
            t0 = time.perf_counter()
            out = subprocess.run(
                [sys.executable, __file__, "--mode", args.mode, "--one"],
                cwd=ROOT,
                env=env,
                capture_output=True,
                text=True,
                # Sin encoding explicito, text=True usa el codec de la locale:
                # en un Windows es-ES es cp1252, que no decodifica el 0x8d que
                # emite el solve. La excepcion ocurre en el reader thread, asi
                # que stdout vuelve None y el .write_text de abajo explota con
                # TypeError -- despues de haber corrido el solve entero.
                encoding="utf-8",
                errors="replace",
            )
            wall = time.perf_counter() - t0
        # El log completo de cada corrida queda en results/ (los .log no se
        # commitean): es lo que hace falta para diagnosticar un code != 1.
        log = HERE / "results" / f"equilibria_{args.mode}_rep{i + 1}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        # `or ""`: si stdout/stderr vuelven None por cualquier motivo, perder
        # el log de una corrida que ya tardo minutos (u horas, en cold) es el
        # peor desenlace posible.
        log.write_text(
            (out.stdout or "") + "\n--- stderr ---\n" + (out.stderr or ""),
            encoding="utf-8",
        )
        line = next(
            (ln for ln in out.stdout.splitlines() if ln.startswith("BENCH_RESULT ")),
            None,
        )
        if out.returncode != 0 or line is None:
            print(f"  fallo (exit {out.returncode}); log en {log.relative_to(ROOT)}")
            runs.append({"ok": False, "wall_process": wall, "log": log.name})
            continue
        r = json.loads(line.removeprefix("BENCH_RESULT "))
        r["ok"] = all(c == 1 for c in r["codes"].values())
        r["wall_process"] = wall
        r["log"] = log.name
        runs.append(r)
        s = r["seconds"]
        print(
            f"  codes={r['codes']}  build {s['build'] / 60:.2f} min  "
            f"solve {s['solve_check_shock'] / 60:.2f} min  total {s['total'] / 60:.2f} min"
        )
    save(
        f"equilibria_{args.mode}",
        {"tool": "equilibria", "mode": args.mode, "runs": runs},
    )
    return 0 if all(r["ok"] for r in runs) else 1


if __name__ == "__main__":
    raise SystemExit(main())
