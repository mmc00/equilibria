"""Los ceros del `.prm` y la seleccion CDE/Cobb-Douglas.

Cubre las tres cosas que el PR #89 cambio y que no tenian test:

1. `keep_zeros` conserva los ceros SOLO donde se pide. El bug que esto evita:
   `keep_zeros=True` estaba dentro del helper `_h()`, que alimenta los 12
   headers de elasticidades, asi que los ceros de ESBT/ESBQ/ESBI pasaban de
   AUSENTES a presentes-en-0.0. Y "ausente" no significa lo mismo para todos
   los consumidores — `path_capi.py` y `calibration_compare.py` ponen default
   1.0, `gtap_model_equations.py` pone 0.0 — o sea que dos call sites leian
   0.0 donde antes leian 1.0, sin que nadie lo pidiera.

2. `SUBPAR=0` sobrevive la carga. Es lo que selecciona Cobb-Douglas
   (Burfisher 3e nota 5, pag. 126: "all substitution parameters as zero"), y
   si el lector lo descarta el modelo cae al default 1.0 y corre CDE en
   silencio: el caso "Cobb-Douglas" corria con bh=1.0.

3. El modo CD se decide por el DATO (SUBPAR), como el `%utility%` de GAMS: con
   un .prm CDE ninguna region queda en modo Cobb-Douglas.

Los tests usan `datasets/` del repo, no el nus333 de las notas: lo que se
verifica es el MECANISMO (un cero se conserva / se descarta segun se pida), y
para eso alcanza con construir el caso a mano. Asi no dependen de un fixture
externo ni de un solver.

NO cubierto todavia, a proposito: el aviso de SUBPAR mixto dentro de una region.
El review del PR #89 midio que ese aviso puede dar FALSO POSITIVO (compara contra
`len(sets.i)` mientras que `_cd_requested` solo se puebla sobre los commodities
CON demanda del hogar, cuando GAMS guarda con `$xaFlag(r,i,h)`). Testearlo ahora
congelaria el comportamiento equivocado; primero hay que arreglar el aviso.
"""

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
import pytest

DATASETS = ROOT / "datasets"
DATASET = "gtap7_3x3"


def _prm_paths(dataset: str = DATASET):
    d = DATASETS / dataset
    needed = ["basedata.har", "sets.har", "default.prm", "baserate.har"]
    missing = [n for n in needed if not (d / n).exists()]
    if missing:
        pytest.skip(f"{dataset}: faltan {missing}")
    return {
        "basedata_path": d / "basedata.har",
        "sets_path": d / "sets.har",
        "default_path": d / "default.prm",
        "baserate_path": d / "baserate.har",
    }


def _load():
    from equilibria.templates.gtap import GTAPParameters

    p = GTAPParameters()
    p.load_from_har(**_prm_paths())
    return p


# --------------------------------------------------------------------------
# 1. keep_zeros: el contrato del lector
# --------------------------------------------------------------------------


def test_har_to_dict_descarta_ceros_por_default():
    """Sin keep_zeros un cero es ausencia de flujo: se descarta (ahorra memoria).

    Este es el comportamiento historico y el que los 11 headers restantes
    siguen necesitando; el test lo fija para que nadie lo cambie de nuevo
    sin querer.
    """
    p = _load()
    # ESBT/ESBQ/ESBI son todo ceros en los .prm de GTAP, asi que sin keep_zeros
    # sus dicts quedan VACIOS: afirmar "no hay ceros" sobre un dict vacio es
    # trivialmente cierto y no distingue "no conservo ceros" de "no cargo nada".
    # Por eso se separa: los que tienen datos no deben traer ceros, y de los que
    # son todo-ceros se afirma que estan vacios, que es justamente el efecto.
    todo_ceros, con_datos = [], []
    for nombre in ("esubt", "esubq", "esubi", "esubd", "esubm", "esubva"):
        d = getattr(p.elasticities, nombre)
        (todo_ceros if not d else con_datos).append((nombre, d))

    assert con_datos, "ningun header cargo datos: el .prm o el lector cambiaron"
    for nombre, d in con_datos:
        ceros = {k: v for k, v in d.items() if v == 0.0}
        assert not ceros, (
            f"{nombre} conserva ceros sin que se los pida: {list(ceros)[:5]}. "
            "Eso cambia el significado de 'clave ausente' para sus consumidores."
        )
    # Los todo-ceros TIENEN que llegar vacios: si alguno trae claves, el
    # keep_zeros se volvio a aplicar de mas y `.get(k, 1.0)` empezo a dar 0.0.
    for nombre, d in todo_ceros:
        assert not d, f"{nombre} deberia estar vacio, trae {len(d)} claves"


def test_load_from_har_conserva_los_ceros_de_subpar():
    """El CABLEADO, no sólo el lector: `load_from_har` tiene que pedir keep_zeros.

    Este es el test que le faltaba a la primera version: los otros llaman a
    `_har_to_dict` directo, asi que pasaban igual con el call site de produccion
    revertido a `_h("SUBP", ...)` sin keep_zeros. Verificado por mutacion: sin
    este test, borrar el arreglo dejaba los 4 en verde.

    Hay que ESCRIBIR un .prm con SUBPAR=0 y cargarlo por el camino publico:
    `default.prm` es CDE (SUBPAR>0), asi que con o sin keep_zeros da lo mismo y
    la mutacion no se nota. El .prm Cobb-Douglas del libro vive fuera del repo,
    de modo que se genera acá a partir del de datasets/.
    """
    import tempfile

    from equilibria.babel.har import read_har, write_har
    from equilibria.templates.gtap import GTAPParameters

    paths = _prm_paths()
    har = read_har(paths["default_path"])
    if "SUBP" not in har:
        pytest.skip("default.prm sin header SUBP")

    n_celdas = int(har["SUBP"].array.size)
    with tempfile.TemporaryDirectory() as td:
        destino = pathlib.Path(td) / "cobbdouglas.prm"
        har["SUBP"].array[...] = 0.0  # SUBPAR=0 en todas las celdas => CD
        try:
            write_har(destino, har)
        except Exception as e:  # noqa: BLE001
            pytest.skip(
                f"write_har no pudo escribir este .prm: {type(e).__name__}: {e}"
            )

        p = GTAPParameters()
        p.load_from_har(
            basedata_path=paths["basedata_path"],
            sets_path=paths["sets_path"],
            default_path=destino,
            baserate_path=paths["baserate_path"],
        )

    subpar = p.elasticities.subpar
    assert len(subpar) == n_celdas, (
        f"load_from_har trajo {len(subpar)} de {n_celdas} celdas de SUBPAR: los "
        "ceros se descartaron en el camino, o sea que el call site no esta "
        "pidiendo keep_zeros=True. Con subpar incompleto el modelo cae al "
        "default bh=1.0 y corre CDE creyendo que es Cobb-Douglas."
    )
    assert all(v == 0.0 for v in subpar.values()), (
        f"se esperaban todos 0.0: {[(k, v) for k, v in subpar.items() if v][:5]}"
    )


def test_subpar_conserva_los_ceros():
    """SUBPAR SI pide keep_zeros: un 0 ahi es Cobb-Douglas, no falta de dato.

    No se puede afirmar que gtap7_3x3 tenga ceros en SUBPAR (default.prm es
    CDE, con valores positivos), asi que lo que se verifica es el camino del
    lector: al pedirle conservar ceros, los conserva.
    """
    from equilibria.babel.har import read_har
    from equilibria.templates.gtap.gtap_parameters import GTAPBenchmarkValues

    paths = _prm_paths()
    sets = _load().sets  # los sets ya cargados, sin reconstruirlos a mano
    har = read_har(paths["default_path"])

    args = (har, "SUBP", sets, ["COMM", "REG"], (1, 0))
    con = GTAPBenchmarkValues._har_to_dict(*args, scale=1.0, keep_zeros=True)
    sin = GTAPBenchmarkValues._har_to_dict(*args, scale=1.0, keep_zeros=False)

    # Con keep_zeros no se pierde ninguna clave; sin el, solo se pierden ceros.
    assert set(sin) <= set(con)
    perdidas = set(con) - set(sin)
    assert all(con[k] == 0.0 for k in perdidas), (
        "keep_zeros=False descarto una clave con valor NO cero: "
        f"{[(k, con[k]) for k in perdidas if con[k] != 0.0][:5]}"
    )
    # Y el reves del bug: keep_zeros no debe inventar ni alterar valores.
    for k in sin:
        assert con[k] == sin[k], f"{k}: {con[k]} != {sin[k]}"


def test_un_cero_de_subpar_sobrevive_la_carga_completa():
    """El cero tiene que llegar hasta `elasticities.subpar`, no sólo al lector.

    Es el test que le faltaba al bug original: el lector podia estar bien y el
    cero perderse igual en el camino.

    El caso Cobb-Douglas se construye poniendo el header SUBP en cero EN MEMORIA
    y leyendolo por el mismo camino que usa `GTAPElasticities.load_from_har`. No
    se escribe un .prm ni se depende del 3x3CobbDouglas.prm, que vive fuera del
    repo (en las notas de dev-tools).
    """
    from equilibria.babel.har import read_har
    from equilibria.templates.gtap.gtap_parameters import GTAPBenchmarkValues

    paths = _prm_paths()
    sets = _load().sets
    har = read_har(paths["default_path"])
    if "SUBP" not in har:
        pytest.skip("default.prm sin header SUBP")

    # Se pone el header en cero y se lee por el mismo camino que usa
    # `GTAPElasticities.load_from_har` para SUBP. Asi el caso Cobb-Douglas se
    # construye con datos del repo, sin escribir un .har ni depender del
    # 3x3CobbDouglas.prm, que vive fuera del repo.
    import copy

    har_cd = dict(har)
    arr = copy.deepcopy(har["SUBP"])
    if hasattr(arr, "array"):
        arr.array = arr.array * 0.0  # HeaderArray: se pone a cero su payload
    else:
        arr = arr * 0.0
    har_cd["SUBP"] = arr

    subpar = GTAPBenchmarkValues._har_to_dict(
        har_cd, "SUBP", sets, ["COMM", "REG"], (1, 0), scale=1.0, keep_zeros=True
    )
    assert subpar, (
        "subpar quedo VACIO con SUBPAR=0: el cero se descarto en el camino y el "
        "modelo cae al default 1.0, o sea corre CDE creyendo que es Cobb-Douglas."
    )
    assert all(v == 0.0 for v in subpar.values()), (
        f"se esperaban todos 0.0, hay otros valores: "
        f"{[(k, v) for k, v in subpar.items() if v != 0.0][:5]}"
    )


# --------------------------------------------------------------------------
# 2. La seleccion del modo la decide el dato
# --------------------------------------------------------------------------


def test_default_prm_no_activa_modo_cd():
    """Con un .prm CDE (SUBPAR>0) ninguna region debe quedar en modo CD.

    Es el guardrail de la afirmacion "CDE no cambia": si esto se rompe, el
    camino CDE empezo a correr las ecuaciones de Cobb-Douglas.
    """
    from equilibria.templates.gtap.gtap_model_equations import GTAPModelEquations

    p = _load()
    assert all(v > 0.0 for v in p.elasticities.subpar.values()), (
        "este dataset dejo de ser CDE; el test necesita un .prm con SUBPAR>0"
    )

    eqs = GTAPModelEquations(p.sets, p, residual_region="ROW")
    # `self._cd_regions` se CREA dentro de la calibracion
    # (gtap_model_equations.py:1429, poblado en :1767), asi que no existe hasta
    # que se construye el modelo: no sirve mirarlo antes.
    eqs.build_model()
    assert hasattr(eqs, "_cd_regions"), (
        "_cd_regions no existe tras build_model(): cambio el nombre del atributo"
    )
    assert eqs._cd_regions == set(), (
        f"con SUBPAR>0 ninguna region deberia quedar en modo Cobb-Douglas, "
        f"quedaron: {eqs._cd_regions}"
    )


# --------------------------------------------------------------------------
# 3. El modelo CD se construye y respeta las guardas de GAMS
# --------------------------------------------------------------------------


def _params_cd(tmp_path):
    """Un GTAPParameters cargado desde un .prm con SUBPAR=0 (Cobb-Douglas)."""
    from equilibria.babel.har import read_har, write_har
    from equilibria.templates.gtap import GTAPParameters

    paths = _prm_paths()
    har = read_har(paths["default_path"])
    if "SUBP" not in har:
        pytest.skip("default.prm sin header SUBP")
    har["SUBP"].array[...] = 0.0
    destino = tmp_path / "cobbdouglas.prm"
    try:
        write_har(destino, har)
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"write_har no pudo escribir este .prm: {type(e).__name__}: {e}")

    p = GTAPParameters()
    p.load_from_har(
        basedata_path=paths["basedata_path"],
        sets_path=paths["sets_path"],
        default_path=destino,
        baserate_path=paths["baserate_path"],
    )
    return p


def test_subpar_cero_activa_modo_cd_en_todas_las_regiones(tmp_path):
    """El caso positivo: con SUBPAR=0 TODAS las regiones tienen que quedar en CD.

    El otro test (`..._no_activa_modo_cd`) solo puede cazar un falso positivo.
    Sin este, borrar la deteccion de CD entera dejaba la suite en verde
    (verificado por mutacion: reemplazar el `_cd_requested.add(...)` por `pass`
    no rompia nada).
    """
    from equilibria.templates.gtap.gtap_model_equations import GTAPModelEquations

    p = _params_cd(tmp_path)
    eqs = GTAPModelEquations(p.sets, p, residual_region="ROW")
    eqs.build_model()
    assert eqs._cd_regions == set(p.sets.r), (
        f"con SUBPAR=0 se esperaban todas las regiones en modo CD; "
        f"quedaron {eqs._cd_regions} de {set(p.sets.r)}"
    )


def test_modo_cd_salta_las_ecuaciones_cde_only(tmp_path):
    """`eveq`/`cveq` son CDE-only en GAMS (model.gms:1322 y :1328).

    Bajo CD, con bh=0, degeneran a `sum_i alphaa == 1`: cierto para CUALQUIER
    ev/cv, o sea que dejarlas activas convierte el bienestar en DOF libre. GAMS
    no las tiene bajo CD y Python tampoco debe tenerlas.

    Y como una fila que se salta necesita su variable fijada para no romper la
    cuadratura del MCP, se verifica tambien que ev/cv queden fijadas.
    """
    from equilibria.templates.gtap.gtap_model_equations import GTAPModelEquations

    p = _params_cd(tmp_path)
    m = GTAPModelEquations(p.sets, p, residual_region="ROW").build_model()

    # dict(...) por el mismo motivo que en el test de abajo: el tipo de un
    # atributo dinamico de Pyomo es `Component | Any` y el ratchet de `ty`
    # rechaza `in` sobre eso.
    for r in p.sets.r:
        for nombre in ("eq_ev", "eq_cv"):
            con = dict(getattr(m, nombre).items())
            assert r not in con, (
                f"{nombre}[{r}] sigue activa bajo Cobb-Douglas. Con bh=0 es una "
                "tautologia (sum alphaa == 1) y deja el bienestar sin ancla."
            )
        for nombre in ("ev", "cv"):
            var = getattr(m, nombre)[r]
            assert var.fixed, (
                f"{nombre}[{r}] quedo LIBRE con su ecuacion saltada: el MCP pierde "
                "la cuadratura (una variable sin fila)."
            )


def test_modo_cd_mantiene_las_ecuaciones_de_utilidad(tmp_path):
    """Las tres que SI existen bajo CD deben seguir activas: zcons, phip, uh.

    Contraparte del test anterior: que el guard de CD no se lleve puesto lo que
    la forma funcional necesita (model.gms:765, :781, :794).
    """
    from equilibria.templates.gtap.gtap_model_equations import GTAPModelEquations

    p = _params_cd(tmp_path)
    m = GTAPModelEquations(p.sets, p, residual_region="ROW").build_model()

    # Los componentes se leen con getattr porque el tipo de un atributo dinamico
    # de Pyomo es `Component | Any`, y el ratchet de `ty` rechaza `in` sobre eso.
    for nombre, ref in (("eq_phip", "model.gms:781"), ("eq_uh", "model.gms:794")):
        con = dict(getattr(m, nombre).items())
        for r in p.sets.r:
            assert r in con, f"{nombre}[{r}] falta bajo CD ({ref})"

    zcons = dict(m.eq_zcons.items())
    regiones = set(p.sets.r)
    activas = sum(1 for idx in zcons if idx[0] in regiones)
    assert activas > 0, "eq_zcons no tiene ninguna celda activa bajo CD (model.gms:765)"


# --------------------------------------------------------------------------
# 4. Los avisos: nada se elige en silencio
# --------------------------------------------------------------------------


def test_subpar_mixto_avisa(tmp_path):
    """Un .prm con SUBPAR=0 en unos bienes y >0 en otros TIENE que avisar.

    GAMS no puede representarlo (`%utility%` es una constante de compilacion),
    asi que la region entera va en CD; elegir en silencio daria un resultado que
    no corresponde a ninguna de las dos formas.

    Ojo: la 2da ronda de review reporto este aviso como "falso positivo" por
    comparar contra `len(sets.i)` en vez del equivalente de `$xaFlag`. MEDIDO que
    NO lo es: `_cd_requested` se puebla recorriendo `self.sets.i` completo, sin
    filtrar por demanda del hogar, asi que ambos lados del `<` cuentan lo mismo.
    Este test fija ese comportamiento para que no se "arregle" lo que funciona.
    """
    import warnings

    from equilibria.babel.har import read_har, write_har
    from equilibria.templates.gtap import GTAPParameters
    from equilibria.templates.gtap.gtap_model_equations import GTAPModelEquations

    paths = _prm_paths()
    har = read_har(paths["default_path"])
    if "SUBP" not in har:
        pytest.skip("default.prm sin header SUBP")
    if har["SUBP"].array.shape[0] < 2:
        pytest.skip("hace falta mas de un commodity para armar el caso mixto")

    # SUBPAR=0 en el primer commodity de la primera region, el resto intacto.
    har["SUBP"].array[0, 0] = 0.0
    destino = tmp_path / "mixto.prm"
    try:
        write_har(destino, har)
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"write_har no pudo escribir este .prm: {type(e).__name__}: {e}")

    p = GTAPParameters()
    p.load_from_har(
        basedata_path=paths["basedata_path"],
        sets_path=paths["sets_path"],
        default_path=destino,
        baserate_path=paths["baserate_path"],
    )
    eqs = GTAPModelEquations(p.sets, p, residual_region="ROW")
    with warnings.catch_warnings(record=True) as capturados:
        warnings.simplefilter("always")
        eqs.build_model()
        avisos = [str(x.message) for x in capturados if "SUBPAR=0 en" in str(x.message)]

    assert len(avisos) == 1, (
        f"se esperaba 1 aviso de SUBPAR mixto, hubo {len(avisos)}: {avisos[:2]}"
    )
    # Y la region entera queda en CD, que es lo que el aviso anuncia.
    assert len(eqs._cd_regions) == 1, (
        f"el aviso dice que la region va entera en CD; _cd_regions={eqs._cd_regions}"
    )


def test_prm_sin_header_subp_avisa(tmp_path):
    """Sin header SUBP, bh cae al default 1.0 (CDE) — y hay que avisarlo.

    Un .prm sin SUBP es indistinguible de uno que pida SUBPAR=1 a proposito, asi
    que un dataset Cobb-Douglas correria CDE sin que nadie se enterara: el mismo
    bug que keep_zeros arregla un nivel mas abajo.
    """
    import warnings

    from equilibria.babel.har import read_har, write_har
    from equilibria.templates.gtap import GTAPParameters

    paths = _prm_paths()
    har = read_har(paths["default_path"])
    if "SUBP" not in har:
        pytest.skip("default.prm ya viene sin header SUBP")

    sin_subp = {k: v for k, v in har.items() if k != "SUBP"}
    destino = tmp_path / "sin_subp.prm"
    try:
        write_har(destino, sin_subp)
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"write_har no pudo escribir este .prm: {type(e).__name__}: {e}")

    p = GTAPParameters()
    with warnings.catch_warnings(record=True) as capturados:
        warnings.simplefilter("always")
        p.load_from_har(
            basedata_path=paths["basedata_path"],
            sets_path=paths["sets_path"],
            default_path=destino,
            baserate_path=paths["baserate_path"],
        )
        avisos = [str(x.message) for x in capturados if "header SUBP" in str(x.message)]

    assert avisos, "sin header SUBP hay que avisar, no caer a CDE en silencio"
    assert not p.elasticities.subpar, "se esperaba subpar vacio sin el header"


def test_pairing_y_fix_endowments_juntos_avisan():
    """Pedir los dos cierres es una contradiccion: gana el pairing, y se avisa.

    El pairing deja xft LIBRE (emparejada a eq_xfteq, model.gms:1413) y
    fix_endowments la FIJA. Resolverlo en silencio deja al usuario creyendo que
    corrio el cierre que no corrio.
    """
    import warnings

    from equilibria.templates.gtap.gtap_contract import GTAPClosureConfig
    from equilibria.templates.gtap.gtap_model_equations import GTAPModelEquations
    from equilibria.templates.gtap.gtap_solver import GTAPSolver

    p = _load()
    cl = GTAPClosureConfig(fix_endowments=True, gams_factor_pairing=True)
    m = GTAPModelEquations(p.sets, p, residual_region="ROW", closure=cl).build_model()
    h = GTAPSolver(m, solver_name="path", params=p)
    with warnings.catch_warnings(record=True) as capturados:
        warnings.simplefilter("always")
        h.apply_closure(cl)
        avisos = [
            str(x.message) for x in capturados if "incompatibles" in str(x.message)
        ]

    assert avisos, (
        "fix_endowments + gams_factor_pairing es contradictorio y se resolvia en "
        "silencio a favor del pairing"
    )
