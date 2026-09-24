# tests/blocks/test_sin_registro_de_bloques.py
#
# Candado de la decision documentada en
# docs/architecture/registro_de_bloques.md: no hay registro de bloques.
#
# Son dos candados de distinta naturaleza y el segundo es el que aguanta:
#   1. El de nombres es un recordatorio, no una barrera: una lista de simbolos
#      via `hasattr`. Un `getattr` dinamico o un rename se lo salta.
#   2. El INVARIANTE POSITIVO: los bloques se componen por import directo, y
#      `_block_classes` los resuelve en orden de dependencia sin consultar
#      ninguna tabla global. Ese es el mecanismo que sustituye al registro.
import equilibria
import equilibria.blocks
import equilibria.blocks.base

NOMBRES = ("BlockRegistry", "get_registry", "register_block")


def test_el_registro_no_esta_en_la_superficie_publica():
    """Si vuelve, que vuelva con un usuario y con el ADR actualizado."""
    for mod in (equilibria, equilibria.blocks, equilibria.blocks.base):
        presentes = [n for n in NOMBRES if hasattr(mod, n)]
        assert not presentes, (
            f"{mod.__name__} reexpone {presentes}; ver "
            "docs/architecture/registro_de_bloques.md"
        )
        exportados = [n for n in NOMBRES if n in getattr(mod, "__all__", ())]
        assert not exportados, f"{mod.__name__}.__all__ exporta {exportados}"


def test_los_bloques_se_componen_por_import_directo():
    """El invariante que sustituye al registro.

    `_block_classes` resuelve las 7 clases GTAP en orden de dependencia sin
    consultar ninguna tabla global: ese es el mecanismo real de composicion.
    """
    from equilibria.templates.gtap.gtap_block_model import _block_classes

    clases = _block_classes()
    assert len(clases) == 7, f"esperadas 7 clases GTAP, hay {len(clases)}"
    assert clases[-1].__name__ == "ClosureBlock", (
        "closure va al final: depende de todo lo demas"
    )
    for c in clases:
        assert isinstance(c, type), f"{c!r} no es una clase"


def test_ningun_fichero_del_repo_menciona_el_registro():
    """El candado que faltaba.

    El borrado inicial dejo vivo `examples/cge/example_04_custom_blocks.py`
    porque el grep cubrio `src/`, `tests/` y `scripts/` pero no `examples/`, y
    los hooks de pre-commit estan acotados a `^(src|tests)/` a proposito. Nada
    lo habria detectado: lo encontro el CI.

    Este test barre TODO el arbol de codigo, sin lista de directorios.
    """
    import pathlib

    raiz = pathlib.Path(__file__).resolve().parents[2]
    permitidos = {
        raiz / "tests" / "blocks" / "test_sin_registro_de_bloques.py",
        raiz / "docs" / "architecture" / "registro_de_bloques.md",
    }
    infractores = []
    for f in raiz.rglob("*.py"):
        if any(p in f.parts for p in (".git", ".venv", "node_modules", "site")):
            continue
        if f in permitidos:
            continue
        texto = f.read_text(encoding="utf-8", errors="replace")
        encontrados = [n for n in NOMBRES if n in texto]
        if encontrados:
            infractores.append(f"{f.relative_to(raiz)}: {', '.join(encontrados)}")

    assert not infractores, "el registro de bloques sigue vivo en:\n  " + "\n  ".join(
        infractores
    )
