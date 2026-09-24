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
    """El candado que faltaba, y el unico que atrapa la regresion real.

    El borrado inicial dejo vivo `examples/cge/example_04_custom_blocks.py`
    porque el grep cubrio `src/`, `tests/` y `scripts/` pero no `examples/`, y
    los hooks de pre-commit estan acotados a `^(src|tests)/` a proposito. Nada
    lo habria detectado: lo encontro el CI.

    Barre TODO el arbol `.py` sin lista de directorios, y mira el AST —imports,
    nombres, atributos y definiciones— en vez de subcadenas. La diferencia se
    midio: una MENCION en comentario ("antes existia un `get_registry()`") es
    justo lo que el ADR pide que la gente escriba, y con subcadenas ponia la
    suite en rojo; por AST no. Y el uso REAL que rompio el CI si lo atrapa.

    Techo medido de lo que este candado puede dar, reproducido y no deducido:
    un registro con otro nombre (`BlockRegistryV2`), o este mismo montado
    dinamicamente (`globals()["Block" + "Registry"] = _R`), pasa igualmente.
    Un candado asi hace ruidoso el OLVIDO, no la intencion; la decision la
    gobierna el ADR, no el test.
    """
    import ast
    import pathlib

    raiz = pathlib.Path(__file__).resolve().parents[2]
    # El unico fichero .py cuyo trabajo ES nombrar el registro borrado.
    permitidos = {raiz / "tests" / "blocks" / "test_sin_registro_de_bloques.py"}
    ignorados = {"node_modules", "site", "build", "dist"}

    infractores = []
    for f in sorted(raiz.rglob("*.py")):
        if f in permitidos:
            continue
        # Sobre la ruta RELATIVA: la absoluta del worktree puede contener un
        # directorio oculto (`.superset`) y excluiria el repo entero.
        partes = f.relative_to(raiz).parts
        if any(p.startswith(".") or p in ignorados for p in partes):
            continue
        try:
            arbol = ast.parse(f.read_text(encoding="utf-8", errors="replace"))
        except SyntaxError:
            continue  # ficheros de fixture que no son Python valido

        usados = set()
        for n in ast.walk(arbol):
            if isinstance(n, ast.ImportFrom):
                usados.update(a.name for a in n.names)
            elif isinstance(n, ast.Import):
                usados.update(a.name.split(".")[-1] for a in n.names)
            elif isinstance(n, ast.Name):
                usados.add(n.id)
            elif isinstance(n, ast.Attribute):
                usados.add(n.attr)
            elif isinstance(n, ast.ClassDef | ast.FunctionDef):
                usados.add(n.name)

        encontrados = sorted(usados & set(NOMBRES))
        if encontrados:
            infractores.append(f"{f.relative_to(raiz)}: {', '.join(encontrados)}")

    assert not infractores, "el registro de bloques sigue vivo en:\n  " + "\n  ".join(
        infractores
    )
