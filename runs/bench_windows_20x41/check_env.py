"""Que hay en esta maquina para el benchmark, y que falta.

    uv run python runs/bench_windows_20x41/check_env.py

No instala nada: dice que encontro y, si falta algo, como conseguirlo. Escribe
``results/env.json``.
"""

from __future__ import annotations

import ctypes
import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _bench import DATASET, ROOT, save  # noqa: E402

sys.path.insert(0, str(ROOT / "src"))

GTAPV7 = Path(os.environ.get("GTAPV7", r"C:\runGTAP375\gtapv7.exe"))


def _ok(cond: bool, label: str, detail: str = "", fix: str = "") -> dict:
    mark = "OK  " if cond else "FALTA"
    print(f"[{mark}] {label}" + (f": {detail}" if detail else ""))
    if not cond and fix:
        print(f"        -> {fix}")
    return {"ok": cond, "detail": detail}


def check_dataset() -> dict:
    d = ROOT / "datasets" / DATASET
    files = ("basedata.har", "sets.har", "default.prm", "baserate.har")
    missing = [f for f in files if not (d / f).exists()]
    return _ok(
        not missing, "datos 20x41", str(d) if not missing else f"faltan {missing}"
    )


def check_path() -> dict:
    from equilibria._local_refs import (
        path_capi_lib,
        path_capi_lib_dir,
        path_capi_lusol_names,
        path_capi_src,
    )

    lib = path_capi_lib()
    lusol = next(
        (
            path_capi_lib_dir() / n
            for n in path_capi_lusol_names()
            if (path_capi_lib_dir() / n).exists()
        ),
        None,
    )
    loads = False
    detail = f"{lib}"
    if lib.exists() and lusol is not None:
        try:
            ctypes.CDLL(str(lusol), mode=getattr(ctypes, "RTLD_GLOBAL", 0))
            ctypes.CDLL(str(lib))
            loads = True
        except OSError as e:
            detail = f"{lib} no carga: {e}"
    r = {
        "lib": _ok(
            loads,
            "libreria PATH (+ LUSOL)",
            detail,
            "bajar https://pages.cs.wisc.edu/~ferris/path/path_5.0.05_Win64.zip, "
            "copiar path_5/pathlib/lib/{path50,lusol}.dll a .cache/path_capi/ "
            "(o definir EQUILIBRIA_PATH_CAPI_LIB_DIR)",
        )
    }
    src = path_capi_src()
    r["src"] = _ok(
        src.exists(),
        "path-capi-python",
        str(src),
        "git clone https://github.com/mmc00/path-capi-python y definir "
        "EQUILIBRIA_PATH_CAPI_SRC=<clon>/src",
    )
    r["license"] = _ok(
        bool(os.environ.get("PATH_LICENSE_STRING")),
        "PATH_LICENSE_STRING",
        "definida" if os.environ.get("PATH_LICENSE_STRING") else "",
        "ver README (licencia de cortesia publica, hasta 2035)",
    )
    return r


def check_asl() -> dict:
    try:
        from pyomo.contrib.pynumero.asl import AmplInterface

        ok = bool(AmplInterface.available())
    except Exception:
        ok = False
    return _ok(
        ok,
        "PyNumero ASL (Jacobiano del driver)",
        "",
        "uv run pyomo download-extensions",
    )


def check_gempack() -> dict:
    return _ok(
        GTAPV7.exists(),
        "GEMPACK gtapv7.exe",
        str(GTAPV7),
        "instalar RunGTAP 3.75 o definir GTAPV7=<ruta a gtapv7.exe>",
    )


def check_gams() -> dict:
    gams = shutil.which("gams")
    if not gams:
        return _ok(False, "GAMS", "", "instalar GAMS y ponerlo en el PATH")
    sysdir = Path(gams).resolve().parent
    lic = sysdir / "gamslice.txt"
    lic_text = lic.read_text(errors="replace") if lic.exists() else ""
    head = " | ".join(lic_text.splitlines()[:2])
    version = ""
    try:
        out = subprocess.run([gams, "?"], capture_output=True, text=True, timeout=60)
        version = next(
            (ln.strip() for ln in out.stdout.splitlines() if "GAMS Release" in ln), ""
        )
    except (OSError, subprocess.SubprocessError):
        pass
    demo = "demo" in lic_text.lower() or "community" in lic_text.lower()
    r = _ok(True, "GAMS", f"{gams} {version}")
    r["license_head"] = head
    r["looks_limited"] = demo or not lic_text
    _ok(
        not r["looks_limited"],
        "licencia GAMS completa",
        head or "sin gamslice.txt",
        "con licencia demo/community el 20x41 (~400.000 filas) no se puede "
        "resolver: bench_gams.py lo va a registrar como 'licencia insuficiente'",
    )
    return r


def check_gdxdump() -> dict:
    exe = shutil.which("gdxdump")
    return _ok(
        exe is not None,
        "gdxdump (lo usa el modo warm para leer el GDX semilla)",
        exe or "",
        "viene con GAMS; o descomprimir la wheel gamspy_base de PyPI y poner "
        "su carpeta gamspy_base/ en el PATH",
    )


def main() -> int:
    print(f"Python {sys.version.split()[0]} en {sys.platform}\n")
    res = {
        "equilibria": _ok(
            importlib.util.find_spec("equilibria") is not None,
            "equilibria importable",
            "",
            "uv sync --dev desde la raiz del repo",
        ),
        "dataset": check_dataset(),
        "path": check_path(),
        "asl": check_asl(),
        "gdxdump": check_gdxdump(),
        "gempack": check_gempack(),
        "gams": check_gams(),
    }
    save("env", {"checks": res})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
