"""ALTERTAX on nus333 through the array driver: load ``altertax.prm``, solve base,
check and shock. No Pyomo here."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

from equilibria.family.altertax.closure import ALTAX_RESIDUAL_REGION, altertax_closure
from equilibria.family.driver import FamilyResult, solve_gtap_family
from equilibria.templates.gtap import GTAPParameters

# ``Shock to("AGR","AGR",USA") = 0.0012`` of ``altax.exp``: ``to(i,a,r)`` is the
# instrument ``prdtx_rai[r,a,i]`` (percent change in the power of the output tax,
# ``shocks.GEMPACK_KIND[prdtx_rai] == power``). AGR/AGR is symmetric in (a, i).
ALTAX_SHOCK: dict[str, dict[tuple[str, ...], float]] = {
    "prdtx_rai": {("USA", "AGR", "AGR"): 0.0012}
}

# set of each position of a shocked cell (as ``blocks/gtap/shock.py`` registers them)
_CELL_SETS = {"prdtx_rai": ("r", "a", "i")}


@dataclass
class AltertaxRun:
    params: GTAPParameters
    result: FamilyResult
    codes: dict[str, int]
    minutes: float
    dataset_dir: Path


# Elasticities that ``altertax.prm`` sets to exactly 1, which the 1.01 guard of #192
# (cal.gms, as for the production nests) must also cover: ``esubm`` (``eq_pmteq``) and
# ``sigmas`` = 1/esubq (``eq_xs``) are CES rows with exponent 1 - sigma that the blocks
# drop at sigma = 1 (a free ``pmt`` and ``ps`` per cell: the system is not square);
# ``esubd`` (the domestic/imported Armington nest) leaves ``pa`` undetermined at 1, and
# check drifts away from base.
UNIT_TRADE_ELASTICITIES = ("esubd", "esubm", "sigmas")


def load_altertax_params(
    dataset_dir: Path, prm: str = "altertax.prm", *, unit_sigma: float = 1.01
) -> GTAPParameters:
    """``GTAPParameters`` of the nus333 package with ``prm`` as the parameter file.

    The production nests ``sigmap``/``sigmand``/``sigmav`` come out of the loader at 1.01
    (#192); ``esubm`` and ``sigmas`` equal to 1 are set to ``unit_sigma`` here and the
    shares recalibrated (the loader's last two steps)."""
    d = Path(dataset_dir)
    baserate = d / "baserate.har"
    p = GTAPParameters()
    p.load_from_har(
        basedata_path=d / "basedata.har",
        sets_path=d / "sets.har",
        default_path=d / prm,
        baserate_path=baserate if baserate.exists() else None,
    )
    changed = 0
    for name in UNIT_TRADE_ELASTICITIES:
        table = getattr(p.elasticities, name)
        for key, value in table.items():
            if value == 1.0:
                table[key] = unit_sigma
                changed += 1
    # GEMPACK GTAPv7 E_qia (gtapv7.tab, lines ~1717-1720): qia(c,r) = qinv(r), investment
    # demand across commodities is Leontief and ESBI is not read. ``altertax.prm`` carries
    # ESBI = 1, so the ALTERTAX entry point sets esubi = 0 in every region.
    for key in p.elasticities.esubi:
        if p.elasticities.esubi[key] != 0.0:
            p.elasticities.esubi[key] = 0.0
            changed += 1
    if changed:
        p.shares.calibrate(p.benchmark, p.elasticities, p.sets, p.taxes)
        p._calibrate()
    return p


def _check_cells(params: GTAPParameters, shock: dict) -> None:
    """Refuse a shock cell that names no element of the model, before any solve."""
    s = params.sets
    for name, cells in shock.items():
        if name not in _CELL_SETS:
            continue  # other instruments: the driver checks them against the arrays
        names = _CELL_SETS[name]
        for cell in cells:
            if len(cell) != len(names):
                raise ValueError(f"{name}{tuple(cell)}: expected {len(names)} indices")
            for x, dim in zip(cell, names, strict=True):
                if x not in getattr(s, dim):
                    raise ValueError(
                        f"{name}{tuple(cell)}: {x!r} is not in the {dim} set "
                        f"{list(getattr(s, dim))}"
                    )


def solve_altertax(
    dataset_dir: Path,
    *,
    prm: str = "altertax.prm",
    shock: dict = ALTAX_SHOCK,
    cache_root: Path | None = None,
    params: GTAPParameters | None = None,
) -> AltertaxRun:
    """Base, check and shock of ALTERTAX on ``dataset_dir`` (``params`` overrides the
    load, e.g. with other elasticities)."""
    p = load_altertax_params(dataset_dir, prm) if params is None else params
    _check_cells(p, shock)
    t0 = time.perf_counter()
    res = solve_gtap_family(
        p,
        altertax_closure(),
        ALTAX_RESIDUAL_REGION,
        shock=shock,
        cache_root=cache_root,
        # GEMPACK's del_tbal(USA) = 0: the nominal balance, not savf = pigbl * savf_bar
        nominal_trade_balance=True,
    )
    minutes = (time.perf_counter() - t0) / 60.0
    codes = {k: int(res.results[k]["code"]) for k in ("check", "shock")}
    return AltertaxRun(p, res, codes, minutes, Path(dataset_dir))
