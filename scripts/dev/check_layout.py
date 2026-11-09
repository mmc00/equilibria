#!/usr/bin/env python3
"""Repository layout rules (ADR-0010). Runs in pre-commit and CI.

1. Closed root: only the entries below may sit at the repository root. Every
   non-code file goes under ``data/`` (see ``data/README.md``).
2. Lowercase names, with the legacy and upstream exceptions below.
3. English names: no path segment may contain a token (split on ``_ - .``)
   from ``SPANISH_WORDS``, an accent or an ``ñ``. Only names are checked, not
   file contents.
4. No loose ``scripts/*.py``: scripts live in ``scripts/<area>/``.

Changing an allow-list here is a reviewed edit, never a side effect.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

ROOT_DIRS = frozenset(
    {"src", "data", "tests", "scripts", "examples", "docs", "specs", ".github"}
)
ROOT_FILES = frozenset(
    {
        "README.md",
        "CHANGELOG.md",
        "CONTRIBUTING.md",
        "LICENSE",
        "NOTICE",
        "CITATION.cff",
        "Makefile",
        "pyproject.toml",
        "uv.lock",
        ".gitignore",
        ".pre-commit-config.yaml",
        ".readthedocs.yaml",
    }
)
# Emptied by later phases of specs/data/changes/one-data-folder; each phase
# removes its entry.
ROOT_DIRS_TEMPORARY = frozenset({"results"})

# The top level of docs/ (phase 6 of specs/data/changes/one-data-folder).
DOCS_TOP = frozenset(
    {
        "readme.md",
        "glossary.md",
        "roadmap.md",
        "adr",
        "agents",
        "assets",
        "site",
        "notes",
        "archive",
    }
)

SPANISH_WORDS = frozenset(
    [
        "de",
        "del",
        "la",
        "las",
        "el",
        "los",
        "sin",
        "con",
        "para",
        "por",
        "y",
        "que",
        "un",
        "una",
        "punto",
        "residuos",
        "bloques",
        "registro",
        "bienestar",
        "desempleo",
        "corridas",
        "resultado",
        "resultados",
        "cada",
        "sobre",
        "entre",
        "cuando",
        "como",
        "donde",
        "fijo",
        "libre",
        "paso",
        "pasos",
        "precio",
        "precios",
        "ecuacion",
        "ecuaciones",
        "prueba",
        "pruebas",
        "datos",
        "semilla",
        "asentado",
        "cierre",
        "choque",
        "modelo",
        "nuevo",
        "viejo",
        "viejos",
        "lectura",
        "escritura",
        "diccionario",
        "nombres",
        "ejemplos",
        "pagina",
        "cobertura",
        "salida",
        "entrada",
        "hecho",
        "mapa",
        "guia",
        "notas",
    ]
)
_ACCENTED = re.compile("[ñáéíóúü]", re.IGNORECASE)
_TOKEN_SPLIT = re.compile(r"[_\-.\s]+")

# Spanish names that existed when the rule was introduced; the phase that moves
# or renames each one removes it from here.
SPANISH_LEGACY: frozenset[str] = frozenset()

ALLOWED_EXACT: set[str] = {
    "CHANGELOG.md",
    "CONTRIBUTING.md",
    "Makefile",
    "NOTICE",
    "README.md",
    "LICENSE",
    "CITATION.cff",
    # The map of data/ (ADR-0010), named like the root README.
    "data/README.md",
    "src/equilibria/babel/gdx/README.md",
    "src/equilibria/babel/har/README.md",
    "src/equilibria/templates/gtap/README.md",
    "scripts/windows/README.md",
    "data/gdx_tests/sam_v2_0/SAM-V2_0.gdx",
    # Sphinx documentation site:
    #   - Makefile is the GNU Make / Sphinx convention; renaming would
    #     break the standard `make html` workflow developers expect.
    #   - sphinx-gallery looks specifically for README.rst inside each
    #     examples directory; the name is hard-coded by the extension.
    "docs/site/Makefile",
    "docs/site/examples_src/README.rst",
    # GENERATE_* is a fixture-generation how-to. Deliberate name.
    "scripts/gtap/GENERATE_MISSING_LEVELS_FIXTURES.md",
    # GitHub reads the sponsor button only from this exact name.
    ".github/FUNDING.yml",
}
ALLOWED_PREFIXES: tuple[str, ...] = (
    "src/equilibria/templates/data/pep/",
    "src/equilibria/templates/reference/pep/",
    "src/equilibria/templates/reference/pep2/data/",
    "src/equilibria/templates/reference/pep2/scripts/",
    # GTAP 7 GAMS distribution: upstream file names (GFTCET.gms, ...).
    "data/gtap7/gams/",
    "data/gtap7/gams_jgea/",
    # getData.gms reads %BaseName%Dat/Prm/Sets.gdx: GAMS fixes these names.
    "data/gtap7/9x10/9x10",
    # Horridge's SIMPLE distribution (tpmh0103): simpleMCP.gms / simpleNLP.gms
    # keep the author's upstream CamelCase — renaming would break fidelity to
    # the published sources and the references in dogams.bat / the README.
    "data/horridge_simple/",
)


def spanish_tokens(name: str) -> list[str]:
    """The Spanish words of one name (whole tokens) and any accented token."""
    tokens = [t for t in _TOKEN_SPLIT.split(name.lower()) if t]
    return [t for t in tokens if t in SPANISH_WORDS or _ACCENTED.search(t)]


def _upstream(path: str) -> bool:
    return any(path.startswith(prefix) for prefix in ALLOWED_PREFIXES)


def violations(paths: list[str]) -> list[str]:
    bad: list[str] = []
    for path in paths:
        parts = path.split("/")
        top = parts[0]
        if len(parts) == 1:
            if top not in ROOT_FILES:
                bad.append(
                    f"{path}: new file at the root (see data/README.md for where things go)"
                )
        elif top not in ROOT_DIRS and top not in ROOT_DIRS_TEMPORARY:
            bad.append(
                f"{path}: new folder {top}/ at the root (see data/README.md for where things go)"
            )

        if top == "docs" and len(parts) > 1 and parts[1] not in DOCS_TOP:
            bad.append(
                f"{path}: docs/{parts[1]} is not in the docs layout (docs/readme.md)"
            )

        name = parts[-1]
        if (
            any(ch.isalpha() and ch.isupper() for ch in name)
            and path not in ALLOWED_EXACT
            and not _upstream(path)
        ):
            bad.append(f"{path}: name is not lowercase")

        if path not in SPANISH_LEGACY and not _upstream(path):
            words = sorted({w for seg in parts for w in spanish_tokens(seg)})
            if words:
                bad.append(f"{path}: name is not English ({', '.join(words)})")

        if len(parts) == 2 and top == "scripts" and name.endswith(".py"):
            bad.append(f"{path}: loose script; use scripts/<area>/")
    return bad


def _tracked(repo: Path = ROOT) -> list[str]:
    # -z: without it git quotes non-ASCII paths ("docs/a\303\261o.md") and the
    # accent rule would never see an accent.
    out = subprocess.check_output(["git", "-C", str(repo), "ls-files", "-z"])
    return [p for p in out.decode("utf-8").split("\0") if p]


def main() -> int:
    bad = violations(_tracked())
    if bad:
        print("Layout check failed (scripts/dev/check_layout.py):")
        for line in bad:
            print(f"  - {line}")
        return 1
    print("Layout check passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
