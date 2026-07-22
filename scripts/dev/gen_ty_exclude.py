#!/usr/bin/env python3
"""Genera el bloque ``exclude:`` del hook ``ty`` en ``.pre-commit-config.yaml``.

El hook ``ty`` es un candado/ratchet: bloquea el commit en los archivos hoy
limpios y excluye los que ya tienen diagnostics (deuda de tipos congelada).
Esa lista es el backlog de la fase F-ty del roadmap (limpieza total de tipos).

Uso::

    uv run python scripts/dev/gen_ty_exclude.py          # imprime el regex exclude:
    uv run python scripts/dev/gen_ty_exclude.py --list    # imprime la lista cruda

Al limpiar los tipos de un archivo excluido, regenerá el bloque y pegalo en el
``exclude:`` del hook ``ty`` de ``.pre-commit-config.yaml``.
"""

from __future__ import annotations

import re
import subprocess
import sys

PATH_RE = re.compile(r"(?:src|tests)/[^\s:]+\.py")


def dirty_files() -> list[str]:
    """Return the sorted, de-duplicated files that ``ty check`` flags."""
    proc = subprocess.run(
        ["uv", "run", "ty", "check", "src", "tests"],
        capture_output=True,
        text=True,
        check=False,
    )
    out = proc.stdout + proc.stderr
    return sorted({m.group(0) for m in PATH_RE.finditer(out)})


def as_regex(files: list[str]) -> str:
    """Render files as a pre-commit verbose ``(?x)`` exclude regex."""
    escaped = [f.replace(".", r"\.") for f in files]
    lines = "\n    ".join(f"{f}|" for f in escaped[:-1])
    return f"(?x)^(\n    {lines}\n    {escaped[-1]}\n  )$"


def main() -> int:
    files = dirty_files()
    if not files:
        print(
            "# ty: sin archivos sucios — el candado ya puede bloquear TODO el repo",
            file=sys.stderr,
        )
        return 0
    if "--list" in sys.argv:
        print("\n".join(files))
    else:
        print(as_regex(files))
    print(f"# {len(files)} archivos excluidos", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
