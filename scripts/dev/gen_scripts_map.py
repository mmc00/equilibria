"""Genera el mapa de ``scripts/`` (la parte generada de ``scripts/readme.md``).

Para cada script lee la primera linea de su docstring (o del comentario de
cabecera), quien lo nombra (CI, tests, docs, otro codigo) y, para los de GTAP,
si arma el modelo con bloques o con el monolito. El mapa se reescribe entre los
marcadores ``<!-- MAPA:INICIO -->`` y ``<!-- MAPA:FIN -->``; el resto del readme
se escribe a mano.

Uso::

    uv run python scripts/dev/gen_scripts_map.py          # reescribe scripts/readme.md
    uv run python scripts/dev/gen_scripts_map.py --check  # falla si el readme quedo viejo
"""

from __future__ import annotations

import ast
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
README = ROOT / "scripts" / "readme.md"
INICIO, FIN = "<!-- MAPA:INICIO -->", "<!-- MAPA:FIN -->"
EXTS = (".py", ".sh", ".jl", ".gms", ".ps1")


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=ROOT, capture_output=True, text=True, check=False
    ).stdout


def _primera_linea(path: Path) -> str:
    texto = path.read_text(errors="ignore")
    if path.suffix == ".py":
        try:
            doc = ast.get_docstring(ast.parse(texto))
        except SyntaxError:
            return "**no compila** (SyntaxError)"
        return doc.strip().splitlines()[0] if doc else ""
    for linea in texto.splitlines()[:10]:
        s = linea.strip().lstrip("#*!<>").strip()
        if (
            len(s) > 8
            and not s.startswith(("/bin", "/usr", "usr/", "env "))
            and set(s) != {"-"}
        ):
            return s
    return ""


def _nombrado_por(rel: str) -> list[str]:
    nombre = Path(rel).name
    stem = Path(rel).stem
    quien = []
    if _git("grep", "-l", "-F", nombre, "--", ".github", "Makefile").strip():
        quien.append("CI")
    if _git("grep", "-l", "-w", "-F", stem, "--", "tests").strip():
        quien.append("tests")
    if _git(
        "grep",
        "-l",
        "-F",
        nombre,
        "--",
        "docs",
        "README.md",
        "CONTRIBUTING.md",
        "ROADMAP.md",
    ).strip():
        quien.append("docs")
    otros = _git("grep", "-l", "-w", "-F", stem, "--", "scripts", "src", f":!{rel}")
    if [o for o in otros.split() if o != "scripts/readme.md"]:
        quien.append("código")
    return quien


def _camino(path: Path) -> str:
    if path.suffix != ".py":
        return ""
    texto = path.read_text(errors="ignore")
    bloques = "build_block_model" in texto
    # GTAPMultiPeriodModel(...) construido a mano arma el SP con el monolito
    # (_sp_source = "monolith"); los bloques pasan por build_block_model.
    monolito = "GTAPModelEquations" in texto or (
        "GTAPMultiPeriodModel(" in texto and not bloques
    )
    if bloques and monolito:
        return "ambos"
    return "bloques" if bloques else ("monolito" if monolito else "")


def _celda(s: str) -> str:
    return s.replace("|", "\\|")


def mapa() -> str:
    archivos = [f for f in _git("ls-files", "scripts").split() if f.endswith(EXTS)]
    por_carpeta: dict[str, list[str]] = defaultdict(list)
    for f in archivos:
        partes = Path(f).parts
        por_carpeta[partes[1] if len(partes) > 2 else "(raíz)"].append(f)
    salida = []
    for carpeta in sorted(por_carpeta):
        filas = sorted(por_carpeta[carpeta])
        gtap = carpeta == "gtap"
        salida.append(
            f"### `{carpeta}/` ({len(filas)})\n"
            if carpeta != "(raíz)"
            else f"### raíz ({len(filas)})\n"
        )
        cab = "| Script | Qué hace | Lo nombra |" + (" Modelo |" if gtap else "")
        sep = "|---|---|---|" + ("---|" if gtap else "")
        salida += [cab, sep]
        for f in filas:
            p = ROOT / f
            quien = ", ".join(_nombrado_por(f)) or "—"
            fila = f"| `{Path(f).name}` | {_celda(_primera_linea(p))} | {quien} |"
            if gtap:
                fila += f" {_camino(p)} |"
            salida.append(fila)
        salida.append("")
    return "\n".join(salida).rstrip() + "\n"


def _reescrito(actual: str) -> str:
    a, b = actual.index(INICIO) + len(INICIO), actual.index(FIN)
    return actual[:a] + "\n" + mapa() + actual[b:]


def main() -> int:
    actual = README.read_text()
    nuevo = _reescrito(actual)
    if "--check" in sys.argv:
        if nuevo != actual:
            print("scripts/readme.md quedo viejo: corre scripts/dev/gen_scripts_map.py")
            return 1
        return 0
    README.write_text(nuevo)
    print(f"wrote {README.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
