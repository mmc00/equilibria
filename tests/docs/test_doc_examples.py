"""Los ejemplos de la documentación corren.

Medido el 2026-10-28: de 109 bloques ```python en la documentación funcionaban 45;
pep_quickstart 0/7, gtap_quickstart 1/6 (el paso 2 armaba el monolito), README
6/12. Nada lo detectaba: una página podía quedar rota en silencio.

Cada página del sitio (``docs/site/guide/*.md``) y el README corre sus bloques
```python en orden, en un solo proceso (como los ejecutaría alguien siguiendo la
página), desde un directorio temporal con ``datasets/`` y ``tests/`` del repo a
mano, para que lo que escriben los ejemplos no ensucie el árbol. Una página nueva
entra sola; las que necesitan un solver lo declaran en ``REQUIERE``.

Una página conceptual muestra piezas sueltas (``apply_shock(m, ...)`` sin construir
``m``). Ese bloque lleva ``<!-- fragmento -->`` en la línea de arriba (no se ve en
el sitio): no se ejecuta, pero tiene que parsear y sus ``import`` sí corren, así
que si la pieza se renombra o desaparece el gate falla igual.
"""

from __future__ import annotations

import ast
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
PAGINAS = sorted((ROOT / "docs" / "site" / "guide").glob("*.md")) + [ROOT / "README.md"]

# Lo que cada página necesita para correr; el resto corre en cualquier lado.
REQUIERE: dict[str, tuple[str, ...]] = {
    "gtap_quickstart.md": ("needs_path", "needs_asl"),
    "welfare_decomposition.md": ("needs_path", "needs_asl"),
    "pep_quickstart.md": ("needs_path", "needs_asl", "needs_ipopt"),
    "README.md": ("needs_path", "needs_asl", "needs_ipopt"),
}

FRAGMENTO = "<!-- fragmento -->"
_BLOQUE = re.compile(r"(" + re.escape(FRAGMENTO) + r"\n)?```python\n(.*?)```", re.S)


def _bloques(pagina: Path) -> list[str]:
    return [b for _, b in _BLOQUE.findall(pagina.read_text(encoding="utf-8"))]


def _sus_imports(fragmento: str) -> str:
    arbol = ast.parse(fragmento)  # un fragmento que no parsea es un error
    imports = [n for n in arbol.body if isinstance(n, ast.Import | ast.ImportFrom)]
    return "\n".join(ast.unparse(n) for n in imports)


def _script(pagina: Path) -> str:
    """Los bloques de la página en orden; de un fragmento, sólo sus imports."""
    partes = _BLOQUE.findall(pagina.read_text(encoding="utf-8"))
    return "\n\n".join(_sus_imports(b) if marca else b for marca, b in partes)


def _casos() -> list:
    casos = []
    for pagina in PAGINAS:
        if not _bloques(pagina):
            continue
        marcas = [getattr(pytest.mark, m) for m in REQUIERE.get(pagina.name, ())]
        casos.append(pytest.param(pagina, id=pagina.name, marks=marcas))
    return casos


def test_requiere_nombra_paginas_que_existen() -> None:
    """Una entrada de REQUIERE con un nombre viejo no protege nada: es un error."""
    nombres = {p.name for p in PAGINAS}
    assert set(REQUIERE) <= nombres, sorted(set(REQUIERE) - nombres)


@pytest.mark.parametrize("pagina", _casos())
def test_los_ejemplos_de_la_pagina_corren(pagina: Path, tmp_path: Path) -> None:
    for carpeta in ("datasets", "tests"):
        (tmp_path / carpeta).symlink_to(ROOT / carpeta, target_is_directory=True)
    script = tmp_path / "ejemplos.py"
    script.write_text(_script(pagina), encoding="utf-8")
    env = {**os.environ, "MPLBACKEND": "Agg"}
    r = subprocess.run(
        [sys.executable, str(script)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=1800,
    )
    assert r.returncode == 0, (
        f"{pagina.relative_to(ROOT)}: sus ejemplos fallan\n{r.stderr[-3000:]}"
    )


def _corre(md: str, tmp_path: Path) -> subprocess.CompletedProcess:
    pagina = tmp_path / "pagina.md"
    pagina.write_text(md, encoding="utf-8")
    script = tmp_path / "ejemplos.py"
    script.write_text(_script(pagina), encoding="utf-8")
    return subprocess.run([sys.executable, str(script)], capture_output=True, text=True)


def test_un_fragmento_no_se_ejecuta(tmp_path: Path) -> None:
    """Un fragmento usa nombres que la página no define (la pieza, no el castillo)."""
    md = (
        f"{FRAGMENTO}\n```python\n"
        "from equilibria.templates.gtap import apply_shock\n"
        "apply_shock(m, {})\n```\n"
    )
    r = _corre(md, tmp_path)
    assert r.returncode == 0, r.stderr


def test_un_fragmento_con_un_import_roto_falla(tmp_path: Path) -> None:
    """Pero sus imports sí corren: si la pieza desaparece, el gate lo ve."""
    md = (
        f"{FRAGMENTO}\n```python\n"
        "from equilibria.templates.gtap import pieza_que_no_existe\n"
        "pieza_que_no_existe(m)\n```\n"
    )
    r = _corre(md, tmp_path)
    assert r.returncode != 0
    assert "pieza_que_no_existe" in r.stderr


def test_un_fragmento_que_no_parsea_falla(tmp_path: Path) -> None:
    md = f"{FRAGMENTO}\n```python\napply_shock(m, {{\n```\n"
    with pytest.raises(SyntaxError):
        _corre(md, tmp_path)
