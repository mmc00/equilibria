from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_pep_pyomo_solver_sin_rutas_personales() -> None:
    """src/ es codigo publicado en PyPI: no puede referirse al disco del autor.

    _ensure_path_lib buscaba libpath en /Users/marmol/..., asi que en cualquier
    otra maquina la busqueda fallaba en silencio y el solve MCP quedaba sin
    libreria PATH sin decir por que.
    """
    fuente = (
        ROOT / "src/equilibria/templates/pep_pyomo/pep_pyomo_solver.py"
    ).read_text()

    assert "/Users/marmol" not in fuente, (
        "pep_pyomo_solver.py busca librerias en el disco del autor; "
        "en cualquier otra maquina la busqueda falla en silencio"
    )


def test_src_sin_rutas_absolutas_funcionales() -> None:
    """Ningun modulo de src/ debe resolver rutas del disco del autor en runtime.

    Los comentarios `Reference: ...` son otra cosa y se tratan aparte: aqui solo
    interesan las rutas que el codigo llega a abrir o a poner en sys.path.
    """
    ofensores: list[str] = []
    for py in (ROOT / "src").rglob("*.py"):
        for num, linea in enumerate(py.read_text().splitlines(), start=1):
            despojada = linea.strip()
            if despojada.startswith("#") or despojada.startswith("*"):
                continue
            if "Reference:" in linea:
                continue
            if "/Users/marmol" in linea or "/private/tmp/claude" in linea:
                ofensores.append(f"{py.relative_to(ROOT)}:{num}")

    assert not ofensores, "rutas absolutas del autor en codigo de src/: " + ", ".join(
        ofensores
    )
