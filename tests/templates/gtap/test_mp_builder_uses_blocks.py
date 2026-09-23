"""F3: la construccion multiperiodo no debe instanciar el MONOLITO.

`GTAPMultiPeriodModel` construye su modelo de periodo simple con
`GTAPModelEquations` — el monolito.  `GTAPBlockMultiPeriodModel` sobrescribe
ese unico punto y compone bloques.

Medido (gtap7_3x3, closure altertax): los dos producen 3.417 filas, cero solo
en uno u otro, y 3.417/3.417 con el MISMO cuerpo algebraico.  Son el mismo
sistema; lo unico que cambia es quien lo construye.

Con `closure=None` bloques emite 27 celdas MAS: son las 9 ecuaciones de ifSUB
(eq_pp_rai, eq_xwmg, eq_xmgm, eq_pwmg, eq_pefobeq, eq_pmcifeq, eq_pmeq,
eq_pfaeq, eq_pfyeq) que un closure real desactiva al fijar sus vars a los
macros `_m_*` — ver blocks/gtap/__init__.py item 1.  Por eso este contrato se
mide con un closure real, no con None.
"""

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
from test_multiperiod_sets import _load_3x3_params


def _altertax_closure():
    from equilibria.templates.gtap.gtap_contract import GTAPClosureConfig

    return GTAPClosureConfig(
        name="altertax",
        closure_type="MCP",
        capital_mobility="mobile",
        fix_endowments=False,
        fix_taxes=True,
        fix_technology=True,
        if_sub=False,
        numeraire="pnum",
    )


def _build(cls, p_alt, closure):
    rr = list(p_alt.sets.r)[-1]
    mp = cls(p_alt.sets, p_alt, closure, residual_region=rr)
    m = mp.build_sets()
    mp.build_vars(m)
    for t in ("base", "check", "shock"):
        mp.build_equations_intra(m, t)
    mp.build_equations_fisher(m)
    return m


def test_block_multiperiod_builder_never_instantiates_the_monolith(monkeypatch):
    """El contrato de F3: cero `GTAPModelEquations` por la via de bloques.

    Verificado que muerde: apuntando el mismo cuerpo a
    `GTAPMultiPeriodModel` cuenta 4 construcciones y falla.
    """
    from equilibria.templates.gtap import gtap_model_equations as ME
    from equilibria.templates.gtap.altertax import apply_altertax_elasticities
    from equilibria.templates.gtap.gtap_block_model import GTAPBlockMultiPeriodModel

    p_alt = apply_altertax_elasticities(_load_3x3_params(), in_place=False)

    built: list[int] = []
    orig = ME.GTAPModelEquations.__init__

    def counting(self, *a, **k):
        built.append(1)
        return orig(self, *a, **k)

    monkeypatch.setattr(ME.GTAPModelEquations, "__init__", counting)
    _build(GTAPBlockMultiPeriodModel, p_alt, _altertax_closure())

    assert built == [], (
        f"la via de bloques construyo {len(built)} monolito(s); "
        "debe componer bloques y no tocar GTAPModelEquations"
    )


def test_blocks_and_monolith_build_the_same_system():
    """La red del corte: el MISMO cuerpo algebraico, fila por fila.

    Compara el esqueleto (nombres, operadores, estructura) EXACTO y los
    coeficientes con tolerancia relativa 1e-12.

    Las dos versiones anteriores fallaron por extremos opuestos, ambos
    verificados a mano:

    - Normalizar TODO numero a "#" la hacia ciega: ``tmarg`` se hornea como
      literal en el cuerpo, asi que duplicarlo en la ecuacion de bloques no
      cambiaba el string.  Saboteado ``eq_pmcifeq`` (tmarg x2): pasaba verde.
    - Comparar el string COMPLETO la hacia ruidosa: 255 filas diferian en el
      ultimo digito de coma flotante (0.9506885123289279 vs ...78), que es
      orden de operaciones, no otra formula.

    Los init SI difieren a proposito (pm/pmcif/pefob se re-valuan tras el
    escalado en el monolito y se siembran del benchmark en bloques —
    blocks/gtap/__init__.py item 6), pero el init no vive en el cuerpo de la
    restriccion, asi que no contamina esta comparacion.
    """
    import re

    from pyomo.core.expr.visitor import expression_to_string
    from pyomo.environ import Constraint

    from equilibria.templates.gtap.altertax import apply_altertax_elasticities
    from equilibria.templates.gtap.gtap_block_model import GTAPBlockMultiPeriodModel
    from equilibria.templates.gtap.gtap_model_multiperiod import GTAPMultiPeriodModel

    p_alt = apply_altertax_elasticities(_load_3x3_params(), in_place=False)
    gc = _altertax_closure()

    def bodies(cls):
        m = _build(cls, p_alt, gc)
        return {
            f"{c.name}[{k}]": expression_to_string(cd.body)
            for c in m.component_objects(Constraint, active=True)
            for k, cd in c.items()
        }

    mono, blk = bodies(GTAPMultiPeriodModel), bodies(GTAPBlockMultiPeriodModel)

    assert set(mono) == set(blk), (
        f"filas solo-monolito: {sorted(set(mono) - set(blk))[:5]}; "
        f"solo-bloques: {sorted(set(blk) - set(mono))[:5]}"
    )
    assert len(mono) > 3000, f"comparacion vacua o truncada: solo {len(mono)} filas"

    _NUM = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")

    def split(expr: str) -> tuple[str, list[float]]:
        """Esqueleto con los numeros sacados, y los numeros aparte."""
        nums = [float(x) for x in _NUM.findall(expr)]
        return _NUM.sub("#", expr), nums

    distintas = []
    for k in mono:
        sa, na = split(mono[k])
        sb, nb = split(blk[k])
        if sa != sb or len(na) != len(nb):
            distintas.append((k, "estructura"))
            continue
        for x, y in zip(na, nb, strict=True):
            if abs(x - y) > 1e-12 * max(1.0, abs(x), abs(y)):
                distintas.append((k, f"coef {x!r} vs {y!r}"))
                break

    assert not distintas, (
        f"{len(distintas)} de {len(mono)} filas con cuerpo distinto; "
        f"ej {distintas[0][0]} ({distintas[0][1]}):\n"
        f"  mono={mono[distintas[0][0]][:200]}\n  blk ={blk[distintas[0][0]][:200]}"
    )
