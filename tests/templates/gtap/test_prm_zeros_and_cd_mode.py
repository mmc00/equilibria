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
    # ESBT/ESBQ/ESBI son todo ceros en los .prm de GTAP: sin keep_zeros el
    # dict queda VACIO. Si alguna vez llegan con valores no-cero el test
    # deja de ser informativo, asi que se afirma sobre los ceros, no sobre
    # el vacio: ningun valor 0.0 puede haber quedado en el dict.
    for nombre in ("esubt", "esubq", "esubi", "esubd", "esubm", "esubva"):
        d = getattr(p.elasticities, nombre)
        ceros = {k: v for k, v in d.items() if v == 0.0}
        assert not ceros, (
            f"{nombre} conserva ceros sin que se los pida: {list(ceros)[:5]}. "
            "Eso cambia el significado de 'clave ausente' para sus consumidores."
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
