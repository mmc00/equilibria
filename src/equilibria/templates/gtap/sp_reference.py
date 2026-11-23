"""The single-period reference model of a GTAP period (``SPSnapshot``) and its cache.

The Pyomo period model reads its fixings and bounds from a single-period model
built with the period's closure. Moved here from the old Pyomo multiperiod driver
and its period recipe (PR 4 of gtap-on-core), unchanged.
"""

from __future__ import annotations

import logging
import weakref
from typing import Any

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# the single-period reference model whose fixing and bounds replicate_reference copies
# ---------------------------------------------------------------------------
def sp_ref_reusable(prev_closure, closure) -> bool:
    """True when a reference model built for ``prev_closure`` also serves ``closure``.

    The reference replication (``replicate_reference``) reads a COMPLETE single-period
    model built only to be inspected and discarded — 3.4M vars on 20x41.  In gtap
    mode the check period passes the same ``base_closure`` as the base period, so
    that second build is byte-identical to the first: verified by hashing every
    (name, index, value, lb, ub) of the built model, where base and check give the
    same digest and only shock (shocked params) differs.  MEASURED: the three
    reference builds cost 1.74 min of a 7.46 min 20x41 wall.

    The reuse is CONDITIONAL on purpose.  Under altertax the check period uses
    ``alt_closure``, which differs in name, capital_mobility, fix_taxes and
    fix_technology; reusing there would replicate the wrong fixing pattern and
    change the solution WITHOUT raising — the failure mode this codebase keeps
    meeting.  So the predicate compares the whole config and refuses on any
    difference, rather than special-casing the fields that look relevant today.
    """
    if prev_closure is None or closure is None:
        return False
    if prev_closure is closure:
        return True
    dump = getattr(prev_closure, "model_dump", None)
    other = getattr(closure, "model_dump", None)
    if dump is None or other is None:
        return False
    return dump() == other()


def _drop_duplicate_fisher_aggregates(sp) -> int:
    """Quitar del SP de bloques los agregados Fisher duplicados.

    Son duplicados EXACTOS de los `mq_factr_*`/`mq_factw_*` que declara la
    version cross-periodo: misma suma, mismo ancho, mismo periodo.  La
    maquinaria multiperiodo ya los borra —`gtap_model_multiperiod`
    `build_equations_fisher`, con el diagnostico completo— pero el SP que pide
    este modulo nunca pasa por ahi.

    Sin esto el sistema queda SOBREDETERMINADO y
    `_closure_patches.deactivate_zero_unique_var_eqs` lo cuadra desactivando
    una ecuacion REAL: medido `eq_xseq[USA,VegFruit]` en check y shock —el
    balance fisico xs == xds + xet—, con el gate MCP de gtap7_15x10 pure
    ifSUB=1 en 87,00% contra un piso de 99%.

    Se borra la Var ademas de la ecuacion: quitar la fila y dejar la columna
    deja un HUERFANO, que altera el sistema igual que la fila.  Y se borra en
    vez de fijarla porque son sumas DEFINIDAS — un valor rancio deja residual
    en las filas Fisher (por eso existe `refresh_fisher_aggregates`).  Como no
    las consume nadie, lo correcto es que desaparezcan.
    """
    # Los nombres se IMPORTAN de donde se declaran (blocks/gtap/closure.py) en
    # vez de recopiarlos: la lista vivia duplicada en cinco ficheros, asi que
    # anadir un septimo agregado obligaba a acordarse de todos.  El import va
    # DENTRO de la funcion porque este modulo no tiene imports de `equilibria`
    # a primer nivel — todos son diferidos, para no reintroducir el ciclo que
    # ya mordio en __init__.py.
    from equilibria.blocks.gtap.closure import FISHER_AUX_EQS, FISHER_AUX_VARS

    n = 0
    faltan = []
    for name in (*FISHER_AUX_EQS, *FISHER_AUX_VARS):
        comp = getattr(sp, name, None)
        if comp is None:
            faltan.append(name)
            continue
        sp.del_component(comp)
        n += 1
    if faltan:
        # Un nombre que ya no aparece NO es benigno: si el agregado sigue vivo
        # bajo otro nombre, el sistema vuelve a quedar sobredeterminado y
        # `deactivate_zero_unique_var_eqs` cuadra sacrificando una ecuacion
        # REAL —medido `eq_xseq[USA,VegFruit]`, el balance fisico xs==xds+xet,
        # con el gate MCP de gtap7_15x10 en 87,00%—.  El sintoma aparece lejos
        # de aqui, asi que se levanta en el sitio.
        raise RuntimeError(
            "_drop_duplicate_fisher_aggregates no encontro estos componentes en "
            f"el SP de bloques: {faltan}. Si se renombraron, actualiza "
            "FISHER_AUX_EQS/FISHER_AUX_VARS en blocks/gtap/closure.py; si "
            "dejaron de emitirse, quita la "
            "entrada. Dejarlo pasar reintroduce la sobredeterminacion."
        )
    return n


def build_sp_reference(sets, params, closure, residual_region, model=None):
    """El modelo de periodo simple del que se leen `.fixed` / `lb` / `ub`.

    F3: esto era `GTAPModelEquations(...).build_model()` — el MONOLITO — en
    tres sitios (base, check, shock).  En 20x41 son 3,4M de celdas construidas
    para leer tres atributos, y mientras ocurra el monolito es dependencia de
    RUNTIME.

    Bloques produce el mismo fixing.  Medido (closure altertax, tras d9e6d49),
    SP del monolito vs `build_block_single_period`:

        gtap7_3x3    1.330 celdas comunes, `.fixed` distinto = 0
        gtap7_10x7  20.145 celdas comunes, `.fixed` distinto = 0

    Las celdas que SOLO existen en el monolito (470 y 5.063) son los 30
    shifters muertos (dtxshft, afeall, lambdaf, ...), que no aparecen en
    ninguna ecuacion: no hay nada que replicar, y bloques no las declara.

    `EQUILIBRIA_GTAP_REF_MODEL=monolith` vuelve al camino viejo.

    El corte estuvo bloqueado por dos cosas, las dos ya arregladas:

    - 33 filas Fisher duplicadas que solo sobrevivian en esta ruta
      (`_drop_duplicate_fisher_aggregates`, cf2c1b0).
    - Los 9 vars de reporte ifSUB se congelaban a "su valor actual", lo que
      volvia DEFINITIVA la diferencia de init entre las dos rutas — medido
      `pwmg` en 0.001 (monolito) vs 1.0 (bloques), factor 1000, y el gate MCP
      de gtap7_15x10 pure ifSUB=1 en 87%. Ahora se recalculan desde su macro
      (f654f63).
    """
    import os

    # La fuente TIENE que coincidir con la clase que construyo `model`: un
    # modelo del monolito con un SP de bloques deja el fixing desalineado
    # —medido en gtap7_3x3 pure ifSUB=1: codes {base:1, check:0, shock:0}, y
    # test_ifsub_primary_block_consistent en frac_agree 0.00%—. A model built by
    # hand with the monolith is caught here instead of in each caller.
    # El monolito hay que PEDIRLO: o el modelo se declara "monolith"
    # (GTAPMultiPeriodModel lo hace en build_sets), o se fuerza por entorno.
    #
    # Antes un modelo SIN marcador caia al monolito, que era lo conservador
    # mientras las dos clases convivian sin declararse.  Ahora las dos declaran,
    # asi que el default implicito solo servia para que un modelo nuevo heredara
    # el monolito por olvido — lo contrario de lo que F3 persigue.  Sin marcador
    # => BLOQUES.
    # Los dos valores se VALIDAN.  Antes solo se comparaba `== "monolith"`, asi
    # que cualquier otra cosa —incluido un typo como "monolit", o el "blocks"
    # que un test creia que pedia bloques— caia al default en silencio.  Medido:
    # poner "BASURA_XYZ" en la env var dejaba pasar el test igual.
    _VALIDOS = ("monolith", "blocks")

    _src = getattr(model, "_sp_source", None)
    if _src is not None and _src not in _VALIDOS:
        raise ValueError(
            f"_sp_source={_src!r} no reconocido (esperado uno de {_VALIDOS}). "
            "Un valor desconocido caia a bloques en silencio."
        )

    _env = os.environ.get("EQUILIBRIA_GTAP_REF_MODEL")
    if _env is not None and _env not in _VALIDOS:
        raise ValueError(
            f"EQUILIBRIA_GTAP_REF_MODEL={_env!r} no reconocido (esperado uno de "
            f"{_VALIDOS})."
        )

    # El monolito hay que pedirlo; la env var manda sobre el marcador del modelo.
    _pide_monolito = _env == "monolith" or (_env is None and _src == "monolith")
    if _pide_monolito:
        from equilibria.blocks.gtap.overwrite import require_blocks
        from equilibria.templates.gtap import GTAPModelEquations

        require_blocks("build_sp_reference (monolito)")

        return GTAPModelEquations(
            sets, params, closure, residual_region=residual_region
        ).build_model()

    from equilibria.templates.gtap.gtap_block_model import build_block_single_period

    sp = build_block_single_period(params, sets, closure, residual_region)
    _drop_duplicate_fisher_aggregates(sp)
    return sp


class SPSnapshot:
    """What the reference replication (``replicate_reference``) reads from a
    referencia, sin el modelo: por variable, las celdas fijas con su valor y las
    celdas con cota con ``(lb, ub)``.

    El modelo de referencia (3,4M vars en 20x41) solo se construye para leerle esto;
    guardar la foto en vez del modelo deja liberarlo apenas se construye y permite
    reusarla entre corridas (ver ``SPReferenceCache.cached``)."""

    __slots__ = ("fixed", "bounds")

    def __init__(self, fixed: dict, bounds: dict) -> None:
        # nombre -> (indexada?, [(indice, valor)]) / [(indice, lb, ub)]
        self.fixed = fixed
        self.bounds = bounds


def sp_snapshot(sp_model) -> SPSnapshot:
    """La foto de ``sp_model`` que consumen las dos replicas."""
    from pyomo.environ import Var

    fixed: dict = {}
    bounds: dict = {}
    for sp_v in sp_model.component_objects(Var, active=True):
        indexed = sp_v.is_indexed()
        fx, bd = [], []
        for sp_idx in sp_v.index_set():
            sp_vd = sp_v[sp_idx]
            if sp_vd.fixed:
                fx.append(
                    (sp_idx, float(sp_vd.value) if sp_vd.value is not None else 0.0)
                )
            if sp_vd.lb is not None or sp_vd.ub is not None:
                bd.append((sp_idx, sp_vd.lb, sp_vd.ub))
        fixed[sp_v.name] = (indexed, fx)
        bounds[sp_v.name] = (indexed, bd)
    return SPSnapshot(fixed, bounds)


# Foto del modelo de referencia compartida entre las corridas que reciben el mismo
# objeto ``params``: id(params) -> (ref debil a params, clave, SPSnapshot).  Por id
# y no como clave debil porque ``GTAPParameters`` no es hasheable (dataclass con
# igualdad); la ref debil confirma que el id sigue siendo de ese objeto y un
# finalizador borra la entrada cuando ``params`` muere.  Fuera de los atributos de
# ``params`` y de ``m``: el cache del modelo los serializa, y su clave digiere todos
# los atributos de ``params``.
_SHARED_SP_REF: dict[int, tuple[weakref.ref, str, Any]] = {}


def _shared_get(params, key: str):
    entry = _SHARED_SP_REF.get(id(params))
    if entry is None or entry[0]() is not params or entry[1] != key:
        return None
    return entry[2]


def _shared_put(params, key: str, snap) -> None:
    pid = id(params)
    if pid not in _SHARED_SP_REF:
        weakref.finalize(params, _SHARED_SP_REF.pop, pid, None)
    _SHARED_SP_REF[pid] = (weakref.ref(params), key, snap)


class SPReferenceCache:
    """The reference models (``SPSnapshot``) of one run: the base's is kept and
    reused while the closure does not change (the gtap check uses the same one), and
    shared with other runs on the same ``params`` object. Moved from the old Pyomo
    period recipe's preparer (PR 4 of gtap-on-core), with the same key and the same
    sharing."""

    def __init__(self, *, residual_region: str) -> None:
        self.residual_region = residual_region
        self._sp_ref = None
        self._sp_ref_closure = None

    def build(self, m, params, closure):
        """Foto (``SPSnapshot``) del modelo de un periodo con ``closure``; el modelo
        Pyomo se suelta apenas se lee."""
        return sp_snapshot(
            build_sp_reference(
                params.sets, params, closure, self.residual_region, model=m
            )
        )

    def cached(self, m, params, closure, period: str):
        """Foto del modelo de un periodo con ``closure``.  Se guarda la del base y
        se reutiliza mientras el cierre no cambie (el check gtap usa el mismo); una
        construida para otro cierre se usa y se descarta.

        Also shared between runs on the SAME ``params`` object: the base
        calibration (on another ``m``) and the run each make their own cache, and
        built the same model twice (~46 s each on 20x41; identical, measured)."""
        if self._sp_ref is not None and sp_ref_reusable(self._sp_ref_closure, closure):
            _log.info(
                "%s period: reusing the base reference model (identical closure)",
                period,
            )
            return self._sp_ref
        key = self._shared_key(m, params, closure)
        snap = _shared_get(params, key) if key is not None else None
        if snap is not None:
            _log.info(
                "%s period: reusing the reference model of a previous run (same key)",
                period,
            )
        else:
            snap = self.build(m, params, closure)
            if key is not None:
                _shared_put(params, key, snap)
        if self._sp_ref is None:
            self._sp_ref, self._sp_ref_closure = snap, closure
        return snap

    def _shared_key(self, m, params, closure) -> str | None:
        """Clave por CONTENIDO de todo lo que entra al modelo de referencia (params,
        cierre, region residual, fuente del modelo y codigo), o ``None`` si no se
        puede compartir.

        De ``m``, ``build_sp_reference`` solo lee la fuente (``_sp_source``, que
        ``EQUILIBRIA_GTAP_REF_MODEL`` pisa).  Con un hook ``@overwrite`` activo no
        se comparte: cambia el modelo y no esta en la clave (igual que el cache
        del modelo)."""
        import os

        from equilibria.blocks.gtap.fingerprint import fingerprint
        from equilibria.blocks.gtap.overwrite import overwrite

        if overwrite.active():
            return None
        return fingerprint(
            "sp_ref",
            params=params,
            closure=closure,
            residual_region=self.residual_region,
            sp_source=getattr(m, "_sp_source", None),
            ref_model_env=os.environ.get("EQUILIBRIA_GTAP_REF_MODEL"),
        )
