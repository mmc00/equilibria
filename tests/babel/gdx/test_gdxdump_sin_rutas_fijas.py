# tests/babel/gdx/test_gdxdump_sin_rutas_fijas.py
#
# El paquete PUBLICADO no puede llevar rutas absolutas del Mac del autor.
# Tenia 20: `gdxdump_bin: str = "/Library/.../Versions/48/Resources/gdxdump"`
# como valor POR DEFECTO, asi que quien instalara equilibria desde PyPI sin
# pasar la ruta recibia una carpeta que en su maquina no existe. Y ademas la
# v48 ya no pasa el servidor de licencias de GAMS (HTTP 400).
import ast
import pathlib

from equilibria.babel.gdx.gdxdump import _orden_de_version, locate_gdxdump

RAIZ = pathlib.Path(__file__).resolve().parents[3]
PAQUETE = RAIZ / "src" / "equilibria"

#: Una sola excepcion: el propio resolutor, que necesita los PATRONES de
#: busqueda (con comodin, nunca una version concreta) para saber donde mirar.
#: `test_el_resolutor_solo_usa_patrones_no_versiones_concretas` es su contrapeso.
#:
#: `pep_pyomo_solver.py` estuvo aqui y ya no hace falta: su ruta fija a la
#: libpath de GAMS 53 era la misma clase de bug, asi que se arreglo en vez de
#: eximirla.
PERMITIDOS = {PAQUETE / "babel" / "gdx" / "gdxdump.py"}


def _literales_de(f: pathlib.Path) -> list[str]:
    """Strings de CODIGO del fichero, via AST.

    Ni comentarios ni docstrings: explicar por que una ruta era un problema
    —citandola— es justo lo que estos candados quieren que se escriba, y con
    los docstrings dentro el candado castigaba su propia documentacion.
    """
    try:
        arbol = ast.parse(f.read_text(encoding="utf-8", errors="replace"))
    except SyntaxError:
        return []

    docstrings = {
        id(n.body[0].value)
        for n in ast.walk(arbol)
        if isinstance(
            n, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef
        )
        and n.body
        and isinstance(n.body[0], ast.Expr)
        and isinstance(n.body[0].value, ast.Constant)
        and isinstance(n.body[0].value.value, str)
    }
    return [
        n.value
        for n in ast.walk(arbol)
        if isinstance(n, ast.Constant)
        and isinstance(n.value, str)
        and id(n) not in docstrings
    ]


def test_el_paquete_no_lleva_rutas_absolutas_de_gams():
    """El invariante: ningun .py publicable nombra una instalacion concreta."""
    infractores = []
    for f in sorted(PAQUETE.rglob("*.py")):
        if f in PERMITIDOS or "reference" in f.parts:
            continue
        for s in _literales_de(f):
            if "GAMS.framework" in s or "/opt/gams" in s or "/usr/local/gams" in s:
                infractores.append(f"{f.relative_to(RAIZ)}: {s}")

    assert not infractores, (
        "rutas de GAMS fijas en el paquete publicable — usa `locate_gdxdump()`:\n  "
        + "\n  ".join(infractores)
    )


def test_el_resolutor_solo_usa_patrones_no_versiones_concretas():
    """Contrapeso de su exencion en PERMITIDOS.

    El resolutor puede nombrar directorios de GAMS —los necesita para buscar—
    pero SIEMPRE con comodin. Una ruta a una version concreta ahi dentro seria
    el mismo bug, escondido en el sitio que lo arregla.
    """
    f = PAQUETE / "babel" / "gdx" / "gdxdump.py"
    sin_comodin = [
        s
        for s in _literales_de(f)
        if ("GAMS.framework" in s or "/opt/gams" in s or "/usr/local/gams" in s)
        and "*" not in s
    ]
    assert not sin_comodin, f"el resolutor lleva rutas sin comodin: {sin_comodin}"


def test_ningun_default_de_parametro_es_una_ruta_de_gams():
    """El caso concreto que rompia a los usuarios de PyPI.

    `gdxdump_bin: str = "/Library/..."` le da al usuario una ruta del Mac del
    autor sin que la haya pedido. El centinela correcto es `None`.
    """
    malos = []
    for f in sorted(PAQUETE.rglob("*.py")):
        if f in PERMITIDOS or "reference" in f.parts:
            continue
        try:
            arbol = ast.parse(f.read_text(encoding="utf-8", errors="replace"))
        except SyntaxError:
            continue
        for n in ast.walk(arbol):
            if not isinstance(n, ast.FunctionDef | ast.AsyncFunctionDef):
                continue
            for d in [*n.args.defaults, *n.args.kw_defaults]:
                if (
                    isinstance(d, ast.Constant)
                    and isinstance(d.value, str)
                    and "GAMS.framework" in d.value
                ):
                    malos.append(f"{f.relative_to(RAIZ)}:{n.lineno} {n.name}()")
    assert not malos, "defaults de parametro con ruta de GAMS:\n  " + "\n  ".join(malos)


def test_current_va_antes_que_las_versiones_numericas():
    """Y las numericas se ordenan como NUMEROS.

    Ordenar los directorios como texto pone "9" por delante de "53" y "48" —
    el bug que tenia la version de `scripts/`.
    """
    dirs = [
        "/x/Versions/Current/Resources",
        "/x/Versions/9/Resources",
        "/x/Versions/53/Resources",
        "/x/Versions/48/Resources",
        "/x/Versions/10/Resources",
    ]
    orden = [
        d.split("/Versions/")[1].split("/")[0]
        for d in sorted(dirs, key=_orden_de_version)
    ]
    assert orden == ["Current", "53", "48", "10", "9"], orden


def test_la_version_menor_desempata_no_la_tupla_mas_corta():
    """`53.5` es mas nueva que `53`, asi que va antes.

    Comparando tuplas sin rellenar, `(-53,)` < `(-53, -5)` y `53` ganaba.
    """
    dirs = ["/x/Versions/53/Resources", "/x/Versions/53.5/Resources"]
    orden = [
        d.split("/Versions/")[1].split("/")[0]
        for d in sorted(dirs, key=_orden_de_version)
    ]
    assert orden == ["53.5", "53"], orden


def test_las_instalaciones_de_linux_tambien_ordenan_por_version():
    """Linux pone la version en el NOMBRE, macOS en el directorio padre.

    Mirando solo el padre, `/opt/gams/gams48.1_x64` daba "gams" — sin digitos —
    y TODAS las instalaciones de Linux empataban en "sin version", con el orden
    entre ellas al azar.
    """
    dirs = [
        "/opt/gams/gams45.7_linux",
        "/opt/gams/gams48.1_x64",
        "/opt/gams/beta",
    ]
    orden = [d.rsplit("/", 1)[1] for d in sorted(dirs, key=_orden_de_version)]
    assert orden == ["gams48.1_x64", "gams45.7_linux", "beta"], orden


def test_la_variable_de_entorno_manda(monkeypatch):
    monkeypatch.setenv("EQUILIBRIA_GDXDUMP", "/ruta/elegida/gdxdump")
    assert locate_gdxdump() == "/ruta/elegida/gdxdump"


def test_sin_gams_devuelve_none_no_una_ruta_inventada(monkeypatch, tmp_path):
    """Mejor `None` explicito que una ruta que no existe."""
    monkeypatch.delenv("EQUILIBRIA_GDXDUMP", raising=False)
    monkeypatch.setattr("shutil.which", lambda *_a, **_k: None)
    monkeypatch.setattr("glob.glob", lambda *_a, **_k: [])
    assert locate_gdxdump() is None
