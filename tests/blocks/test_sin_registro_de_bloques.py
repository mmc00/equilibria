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

    Cubre, medido probando cada caso: import directo, import con alias
    (`as rb`), decorador cualificado (`@blocks.register_block`), definicion, y
    el nombre como string en un `__all__` — que es como estaba declarado el
    registro cuando se borro.

    Techo medido, reproducido y no deducido: se le escapan un registro con otro
    nombre (`BlockRegistryV2`), este mismo montado dinamicamente
    (`globals()["Block" + "Registry"] = _R`) y un `getattr(mod, "get_registry")`
    con el nombre partido. Los tres exigen intencion. Un candado asi hace
    ruidoso el OLVIDO, no la intencion; la decision la gobierna el ADR.
    """
    import ast
    import pathlib

    raiz = pathlib.Path(__file__).resolve().parents[2]
    # El unico fichero .py cuyo trabajo ES nombrar el registro borrado.
    permitidos = {raiz / "tests" / "blocks" / "test_sin_registro_de_bloques.py"}
    ignorados = {"node_modules", "site", "build", "dist"}
    # Como en `test_layering.py`: una excepcion que ya no apunta a nada es un
    # permiso fantasma que tapa regresiones. Si el test se renombra, salta aqui.
    fantasma = [p for p in permitidos if not p.exists()]
    assert not fantasma, f"excepciones que ya no existen: {fantasma}"

    infractores = []
    no_parsean: list[pathlib.Path] = []
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
            # Hoy solo cae aqui `scripts/gtap_julia/jparity.py` (un import
            # intercalado, roto de verdad). Si algun dia cae un fichero que SI
            # deberia revisarse, este candado callaria sobre el: por eso se
            # anota y no se ignora en silencio.
            no_parsean.append(f.relative_to(raiz))
            continue

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
            elif isinstance(n, ast.Constant) and isinstance(n.value, str):
                # Un nombre entre comillas en `__all__` reexporta igual que un
                # import, y asi estaba declarado el registro cuando se borro:
                # sin esto, reañadirlo al `__all__` no lo detectaria nadie
                # (medido: el barrido de nodos sin `Constant` no lo veia).
                # La prosa no molesta porque el cruce de mas abajo es por
                # IGUALDAD, no por subcadena: un docstring que diga "antes
                # existia un register_block" es un Constant, pero su texto
                # entero no es igual al nombre.
                usados.add(n.value)

        encontrados = sorted(usados & set(NOMBRES))
        if encontrados:
            infractores.append(f"{f.relative_to(raiz)}: {', '.join(encontrados)}")

    assert not infractores, "el registro de bloques sigue vivo en:\n  " + "\n  ".join(
        infractores
    )

    # El candado no puede revisar lo que no parsea. Se fija la lista conocida
    # para que un fichero nuevo sin revisar no pase inadvertido.
    assert [p.as_posix() for p in no_parsean] == ["scripts/gtap_julia/jparity.py"], (
        "cambio el conjunto de ficheros que no parsean; el candado no los mira:\n  "
        + "\n  ".join(p.as_posix() for p in no_parsean)
    )
