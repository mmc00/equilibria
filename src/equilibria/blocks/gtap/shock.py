"""GTAP SHOCK block: los instrumentos de shock como Vars fijas.

En GAMS los instrumentos de politica y tecnologia son variables fijadas por
periodo (lambdava, imptx, prdtx, ...; model.gms) o parametros indexados por t
(aft): el shock es `x.fx(...,'shock') = v`, sin tocar ecuaciones. Aca cada
instrumento es una Var que el compositor FIJA en su valor de benchmark y registra
en ``_exogenous_instruments``. El constructor multiperiodo la copia por periodo
como referencia viva (una Var fija no se pliega a literal, a diferencia de un
Param), y el driver no libera ni re-siembra un instrumento registrado. Aplicar un
shock es ``instruments.fix_instrument_shock(m, nombre, indice, factor=...)``.

Solo el CUERPO de las ecuaciones lee el instrumento; la calibracion (shares,
semillas, krat de eq_kstock) sigue leyendo el benchmark.
"""

from __future__ import annotations

from typing import Any

from equilibria.blocks.base import Block
from equilibria.blocks.gtap import _derived_params as dp
from equilibria.core.symbolic_equations import SymbolicEquation
from equilibria.core.variables import Variable

# Nombres de componente de cada instrumento. Fuente unica para el compositor,
# build_block_model y el driver.
SHOCK_INSTRUMENTS: tuple[str, ...] = (
    "lambdava",
    "aft",
    "imptx",
    "prdtx_rai",
    "fcttx",
    "dintx_tgt",
    "mintx_tgt",
    "kappaf",
    "exptx",
    "pop",
    "lambdaf",
    "axp",
    "lambdam",
    "lambdamg",
)


class ShockBlock(Block):
    """Instrumentos de shock (sin ecuaciones propias)."""

    name: str = "GTAP_SHOCK"
    description: str = "GTAP shock instruments as fixed Vars (GAMS x.fx per period)"
    sets: Any = None
    params: Any = None

    def model_post_init(self, __context: Any) -> None:
        self.required_sets = ["r", "a", "f", "i", "aa", "rp", "m"]

    def setup(self, set_manager, parameters, variables) -> list[SymbolicEquation]:
        p, s = self.params, self.sets
        byname = {
            d: list(set_manager.get(d)) for d in ("r", "a", "f", "i", "aa", "rp", "m")
        }

        def _instrument(name: str, data: dict, doms: tuple, default: float) -> None:
            variables[name] = Variable(
                name=name,
                value=dp.to_array(data, [byname[d] for d in doms], default),
                domains=doms,
                domain="Reals",
                lower=float("-inf"),
                upper=float("inf"),
            )

        sh = p.shifts
        # avaall -> lambdava(r,a,t) (model.gms:38, :540, :547).
        _instrument("lambdava", dict(sh.lambdava), ("r", "a"), 1.0)
        # qe -> aft(r,fm,t): Parameter indexado por t en GAMS (model.gms:290),
        # xft = aft*(pft/pabs)**etaf (model.gms:1073). El benchmark para calibrar es
        # aft0 (FactorBlock).
        _instrument("aft", dp.aft_data(p, s), ("r", "f"), 0.0)
        # tms -> imptx(r,i,rp,t) (exportador, bien, importador), cal.gms:333.
        _instrument("imptx", dp.imptx_data(p, s), ("r", "i", "rp"), 0.0)
        # to -> prdtx(r,a,i,t): tasa efectiva makb/maks - 1 (cal.gms:290-291).
        _instrument("prdtx_rai", dp.prdtx_rai_data(p, s), ("r", "a", "i"), 0.0)
        # tfe -> fcttx(r,fp,a,t) (cal.gms:163); fctts sigue siendo Param.
        _instrument("fcttx", dp.fcttx_data(p, s), ("r", "f", "a"), 0.0)
        # tpdall/tfd -> dintx(r,i,aa,t), que GAMS fija en iterloop.gms:32. Aca dintx
        # es Var emparejada con eq_dintxeq; el instrumento es su objetivo.
        _instrument(
            "dintx_tgt",
            {
                (r, i, aa): dp._dintx_target(p, s, r, i, aa)
                for r in byname["r"]
                for i in byname["i"]
                for aa in byname["aa"]
            },
            ("r", "i", "aa"),
            0.0,
        )
        # tpmall/tfm -> mintx(r,i,aa,t), igual que dintx: Var emparejada con
        # eq_mintxeq; el instrumento es su objetivo.
        _instrument(
            "mintx_tgt",
            {
                (r, i, aa): dp._mintx_target(p, s, r, i, aa)
                for r in byname["r"]
                for i in byname["i"]
                for aa in byname["aa"]
            },
            ("r", "i", "aa"),
            0.0,
        )
        # tinc -> kappaf(r,fp,a,t) (cal.gms:143): potencia EVFB/EVOS = 1/(1-kappaf).
        _instrument(
            "kappaf",
            {
                (r, f, a): dp._kappaf(p, r, f, a)
                for r in byname["r"]
                for f in byname["f"]
                for a in byname["a"]
            },
            ("r", "f", "a"),
            0.0,
        )
        # txs -> exptx(r,i,rp,t) (exportador, bien, importador), cal.gms:316.
        _instrument("exptx", dict(p.taxes.rtxs), ("r", "i", "rp"), 0.0)
        # pop -> pop(r,t), variable fija en GAMS (cal.gms:233).
        _instrument("pop", {(r,): dp.pop_value(p, r) for r in byname["r"]}, ("r",), 1.0)
        # afeall -> lambdaf(r,fp,a,t) (model.gms:1356).
        _instrument("lambdaf", dict(sh.lambdaf), ("r", "f", "a"), 1.0)
        # aoall -> axp(r,a,t) (model.gms:1341).
        _instrument("axp", dict(sh.axp), ("r", "a"), 1.0)
        # ams -> lambdam(rp,i,r,t) (origen, bien, destino), model.gms:941/947.
        from equilibria.blocks.gtap.trade_armington_bilateral import _safe

        _instrument(
            "lambdam",
            {
                (e, i, d): max(_safe(p, "lambdam", (e, i, d), 1.0), 1e-12)
                for e in byname["r"]
                for i in byname["i"]
                for d in byname["rp"]
            },
            ("r", "i", "rp"),
            1.0,
        )
        # atd/ats/atf/atm -> lambdamg(m,r,i,rp,t) (margen, origen, bien, destino):
        # eficiencia en el uso de margenes, xmgm = amgm*xwmg/lambdamg y
        # pwmg = sum_m amgm*ptmg/lambdamg (model.gms:1000/1007). GAMS declara
        # atd(r,t) (model.gms:206) pero ninguna ecuacion lo usa.
        _instrument("lambdamg", dp.lambdamg_data(p, s), ("m", "r", "i", "rp"), 1.0)
        return []
