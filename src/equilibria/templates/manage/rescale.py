"""What GAMS rescaled before writing GHA_BaU.gdx, and how to undo it.

MANAGE calibrates and solves in "solve units", the SAM scaled so that GDP is
1e5 (``scaleSAM``; ``inScale = 1e5/defPar('gdp')``, inc/Base.inc:544-555). After
the year loop, ``inc/cal/rescale.inc`` multiplies the ``.l`` of a listed set of
symbols (and some parameters) by ``reScale``, ``reEscale`` and ``usd``, and
run.gms:402 does the same for ``lCovArea``. Bounds (``.lo``/``.up``) are never
rescaled, nor any symbol or domain the list leaves out (``atFmSup`` is rescaled
only for ``inst``, so its ``'sumInst'`` entries stay in solve units).

``RESCALED`` transcribes those statements one per line, with the line it comes
from. ``to_solve_units`` divides a GDX value by the factor of its rule; every
other value is returned as it is. Python works in solve units (design,
decision 4), so this is the only place a GDX value is converted.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass

Domain = tuple[tuple[str | None, int], ...]


@dataclass(frozen=True)
class RescaleRule:
    """One statement of rescale.inc: ``symbol = symbol * factor``.

    ``factor = reScale**rescale * reEscale**reescale * usd**usd *
    inScale**inscale``. ``domain`` lists, from the first index on, the GAMS set
    each group of ``width`` indices must belong to (``None``: any element); it is
    written only where rescale.inc narrows the symbol's own records, and the
    trailing year index is never restricted (``tScale`` holds both years).
    """

    symbol: str
    source: str
    rescale: int = 0
    reescale: int = 0
    usd: int = 0
    inscale: int = 0
    domain: Domain = ()

    def factor(self, scale: Mapping[str, float]) -> float:
        return (
            scale["reScale"] ** self.rescale
            * scale["reEscale"] ** self.reescale
            * scale["usd"] ** self.usd
            * scale["inScale"] ** self.inscale
        )

    def applies(
        self, key: tuple[str, ...], member: Callable[[str, tuple[str, ...]], bool]
    ) -> bool:
        """Whether rescale.inc touched the record ``key`` (``member(set, part)``
        says whether ``part`` belongs to the GAMS set)."""
        pos = 0
        for name, width in self.domain:
            part = key[pos : pos + width]
            if name is not None and not member(name, part):
                return False
            pos += width
        return True


def _r(symbol: str, line: int, **kw: object) -> RescaleRule:
    return RescaleRule(symbol=symbol, source=f"rescale.inc:{line}", **kw)  # type: ignore[arg-type]


_RULES: tuple[RescaleRule, ...] = (
    _r("xfac", 23, rescale=1, domain=(("flagXFac", 3),)),
    _r("xfdRD", 24, rescale=1),
    _r("p_xRD0", 25, rescale=1),
    _r("xp", 26, rescale=1),
    _r("x", 27, rescale=1),
    _r("xs", 28, rescale=1),
    _r("xsExp", 29, rescale=1, domain=(("ist", 1),)),
    _r("st", 30, rescale=1),
    _r("delst", 31, rescale=1),
    _r("mdelst", 32, rescale=1),
    _r("facty", 33, rescale=1),
    _r("kapy", 34, rescale=1),
    _r("yh", 35, rescale=1),
    _r("yhDsp", 36, rescale=1),
    _r("savh", 37, rescale=1),
    _r("fundT", 38, rescale=1),
    _r("fundG", 39, rescale=1),
    _r("fundP", 40, rescale=1),
    _r("fundF", 41, rescale=1),
    _r("totTr", 42, rescale=1),
    _r("debtStk", 43, rescale=1),
    _r("debtStkT", 44, rescale=1),
    _r("debtPay", 45, rescale=1),
    _r("debtInterest", 46, rescale=1),
    _r("debtPrincipal", 47, rescale=1),
    _r("debtNew", 48, rescale=1),
    _r("debtNewT", 49, rescale=1),
    _r("kstocke", 50, rescale=1),
    _r("yg", 51, rescale=1),
    _r("ygov", 52, rescale=1),
    _r("gTax", 53, rescale=1),
    _r("ygEmi", 54, rescale=1),
    _r("ygEmiXP", 55, rescale=1),
    _r("ygPit", 56, rescale=1),
    _r("ygCit", 57, rescale=1),
    _r("ygPaf", 58, rescale=1),
    _r("ygtot", 59, rescale=1),
    _r("fiscalBalance", 60, rescale=1),
    _r("primaryBalance", 61, rescale=1),
    _r("govBalance", 62, rescale=1),
    _r("rgovBalance", 63, rescale=1),
    _r("entrY", 64, rescale=1),
    _r("entrTax", 65, rescale=1),
    _r("savEntr", 66, rescale=1),
    _r("savf", 67, rescale=1),
    _r("bot", 68, rescale=1),
    _r("transfers", 69, rescale=1),
    _r("rowtransfers", 70, rescale=1),
    _r("yf", 71, rescale=1),
    _r("p_yf0", 72, rescale=1),
    _r("p_GDP", 73, rescale=1),
    _r("atFmSup", 74, rescale=1, domain=((None, 1), ("inst", 1), (None, 1))),
    _r("p_gammaFundF", 75, rescale=1),
    _r("p_pfIni", 76, rescale=1),
    _r("xfd", 77, rescale=1),
    _r("mmuh", 78, rescale=1),
    _r("xak", 79, rescale=1),
    _r("xmarg", 80, rescale=1),
    _r("xa", 81, rescale=1),
    _r("xat", 82, rescale=1),
    _r("xd", 83, rescale=1),
    _r("xe", 84, rescale=1),
    _r("xm", 85, rescale=1),
    _r("xfmSup", 86, rescale=1),
    _r("kaps", 87, rescale=1),
    _r("kstock", 88, rescale=1),
    _r("emiInt", 89, rescale=-1, reescale=1),
    _r("emiIntHH", 90, rescale=-1, reescale=1),
    _r("emiIntSec", 91, rescale=-1, reescale=1),
    _r("tdelst", 93, rescale=1),
    _r("rtdelst", 94, rescale=1),
    _r("totExp", 95, rescale=1),
    _r("rTotExp", 96, rescale=1),
    _r("totimp", 97, rescale=1),
    _r("rtotimp", 98, rescale=1),
    _r("oldCap", 99, rescale=1),
    _r("gdpmp", 100, rescale=1),
    _r("rgdpmp", 101, rescale=1),
    _r("gdpfc", 102, rescale=1),
    _r("rgdpfc", 103, rescale=1),
    _r("walras", 104, rescale=1),
    _r("SectTot", 105, rescale=1),
    _r("hPop", 106, rescale=1),
    _r("p_hPop0", 107, rescale=1),
    _r("Pop", 108, rescale=1),
    _r("p_Pop0", 109, rescale=1),
    _r("p_popT", 110, rescale=1),
    _r("PopAge", 111, rescale=1),
    _r("unEmp", 112, rescale=1),
    _r("p_rhoEmi", 116, rescale=-1, reescale=1),
    _r("p_rhoEmiXP", 117, rescale=-1, reescale=1),
    _r(
        "p_rhoEmiFac",
        118,
        rescale=-1,
        reescale=1,
        domain=((None, 1), ("actFp", 1), (None, 1)),
    ),
    _r("tEmi", 120, rescale=1, reescale=-1),
    _r("rtEmi", 121, rescale=1, reescale=-1),
    _r("tEmiD0", 122, rescale=1, reescale=-1),
    _r("emiComm", 123, reescale=1, domain=(("emiCommFlag", 3),)),
    _r("emiFact", 124, reescale=1, domain=((None, 1), ("actFp", 1), ("aact", 1))),
    _r("emiXP", 125, reescale=1, domain=((None, 1), ("aact", 1))),
    _r("emiGHG", 126, reescale=1),
    _r("emiTot", 127, reescale=1),
    _r("emiInv", 128, reescale=1),
    _r("dCarbon", 129, reescale=1),
    _r("carbCnt", 130, rescale=-1, reescale=1, usd=1),
    _r("emiSeca", 131, reescale=1),
    _r("p_land", 133, rescale=1, domain=(("actLnd", 1), (None, 1))),
    _r("p_labor", 134, rescale=1, domain=(("l", 1), (None, 1))),
    _r("p_DebtStkF0", 135, rescale=1),
    _r("p_DebtStkD0", 136, rescale=1),
    _r("sam", 137, rescale=1),
    RescaleRule("lCovArea", "run.gms:402", inscale=-1, usd=-1),
)

# Keyed by the lower-case name: GAMS names are case-insensitive, and the GDX
# keeps the case of the declaration (rescale.inc writes gTax, the GDX gtax).
RESCALED: Mapping[str, RescaleRule] = {r.symbol.lower(): r for r in _RULES}


def to_solve_units(
    name: str,
    key: tuple[str, ...],
    value: float,
    scale: Mapping[str, float],
    member: Callable[[str, tuple[str, ...]], bool],
) -> float:
    """``value`` of the GDX record ``name(key)`` in GAMS's solve units."""
    rule = RESCALED.get(name.lower())
    if rule is None or not rule.applies(key, member):
        return value
    return value / rule.factor(scale)
