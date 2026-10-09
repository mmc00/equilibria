"""Lo comun del benchmark: rutas, datos de la maquina y guardado de resultados."""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RESULTS = HERE / "results"
# El 20x41 no es medible en esta maquina: warm 1 rep = 2 h 1 min y el shock
# igual corta en code=5 (tope de 1 h de PATH, residual 1.914). Con
# EQUILIBRIA_BENCH_DATASET se mide un agregado que si cierra.
DATASET = os.environ.get("EQUILIBRIA_BENCH_DATASET", "gtap7_20x41")

# El experimento, igual para las tres herramientas.
EXPERIMENT = {
    "dataset": DATASET,
    "shock": "tm +10% uniforme (imptx_new = (1+imptx)*1.10 - 1)",
    "closure": "capFix (RORDELTA=0), capital sluggish, residual = ROW",
    "ifsub": 1,
}


def machine() -> dict[str, Any]:
    """CPU, nucleos y RAM: sin esto los tiempos no se pueden comparar con nada."""
    info: dict[str, Any] = {
        "hostname": platform.node(),
        "os": platform.platform(),
        "python": sys.version.split()[0],
        "cpu": platform.processor(),
        "logical_cpus": os.cpu_count(),
    }
    try:
        import psutil

        info["ram_gb"] = round(psutil.virtual_memory().total / 2**30, 1)
        info["physical_cpus"] = psutil.cpu_count(logical=False)
    except ImportError:
        info["ram_gb"] = None
    if sys.platform.startswith("win"):
        try:
            out = subprocess.run(
                [
                    "powershell",
                    "-NoProfile",
                    "-Command",
                    "(Get-CimInstance Win32_Processor).Name",
                ],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=30,
            )
            if out.stdout.strip():
                info["cpu"] = out.stdout.strip()
        except (OSError, subprocess.SubprocessError):
            pass
    return info


def save(name: str, payload: dict[str, Any]) -> Path:
    """Guarda ``results/<name>.json`` con la maquina, el experimento y la fecha."""
    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / f"{name}.json"
    payload = {
        "when": datetime.now().isoformat(timespec="seconds"),
        "machine": machine(),
        "experiment": EXPERIMENT,
        **payload,
    }
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"-> {out.relative_to(ROOT)}")
    return out
