"""ALTERTAX on nus333 through the array driver: load ``altertax.prm``, solve base,
check and shock. No Pyomo here."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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
    # ``solve_altertax(decomp_steps=...)``: the path-integral decomposition
    # (``family.altertax.path.DecompPath``); None = the one-step ``decomp_headers``
    decomp_path: Any = None


# Elasticities that ``altertax.prm`` sets to exactly 1. The route's closure has
# ``unit_sigma="gempack"`` (``altertax_closure``): the blocks write every one of them as
# an exact Cobb-Douglas nest (``blocks/gtap/_cd.py``) and ``load_altertax_params`` keeps
# them at 1. The 1.01 stand-in of GAMS #192 (cal.gms) stays selectable for A/B runs
# (``load_altertax_params(..., unit_sigma=1.01)``): ``esubm`` (``eq_pmteq``) and
# ``sigmas`` = 1/esubq (``eq_xs``) are CES rows with exponent 1 - sigma that drop at
# sigma = 1 outside the CD form, and ``esubd`` leaves ``pa`` undetermined there.
UNIT_TRADE_ELASTICITIES = ("esubd", "esubm", "sigmas")
# guarded to 1.01 inside the blocks under unit_sigma="gams" (sigmag, sigmamg)
UNIT_BLOCK_ELASTICITIES = ("esubg", "sigmam")
_BLOCK_GUARD = 1.01


def apply_unit_stand_in(
    params: GTAPParameters, unit_sigma: float = 1.01, *, recalibrate: bool = True
) -> int:
    """The 1.01 stand-in of the route before ``unit_sigma="gempack"``, so that a run with
    it reproduces that route under the ALTERTAX closure too:

    - every unit ``esubd``/``esubm``/``sigmas`` (:data:`UNIT_TRADE_ELASTICITIES`) becomes
      ``unit_sigma``;
    - ``esubg`` and ``sigmam`` equal to 1 become 1.01 (the blocks' own guard);
    - the production nests stay at the loader's 1.01 (#192), made deliberate: the
      closure's ``set_unit_sigma("gempack")`` no longer restores them.

    With ``recalibrate`` the shares are recalibrated. Returns the number of
    elasticities changed."""
    changed = 0
    el = params.elasticities
    for name in UNIT_TRADE_ELASTICITIES:
        table = getattr(el, name)
        for key, value in table.items():
            if value == 1.0:
                table[key] = unit_sigma
                changed += 1
    for name in UNIT_BLOCK_ELASTICITIES:
        table = getattr(el, name)
        for key, value in table.items():
            if value == 1.0:
                table[key] = _BLOCK_GUARD
                changed += 1
    before = {n: dict(getattr(el, n)) for n in ("sigmap", "sigmand", "sigmav")}
    el.guard_unit_production_elasticities()
    changed += sum(before[n][k] != v for n in before for k, v in getattr(el, n).items())
    el.keep_unit_production_guard()
    if changed and recalibrate:
        params.shares.calibrate(
            params.benchmark, params.elasticities, params.sets, params.taxes
        )
        params._calibrate()
    return changed


def load_altertax_params(
    dataset_dir: Path,
    prm: str | Path = "altertax.prm",
    *,
    unit_sigma: float | None = None,
) -> GTAPParameters:
    """``GTAPParameters`` of the nus333 package with ``prm`` as the parameter file.

    ``prm`` is a name inside ``dataset_dir`` or, when no such file exists there, a path
    (e.g. the ``parameters.prm`` of a RunGTAP reference folder). The factor mobility sets
    come from its ``EFLG`` header (``GTAPSets.load_from_har``), not from the closure.

    ``unit_sigma=None`` (default): every elasticity ``prm`` sets to 1 stays exactly 1,
    for the closure's exact Cobb-Douglas nests (``altertax_closure``,
    ``unit_sigma="gempack"``): the production nests the loader moved to 1.01 (#192) are
    put back here (``GTAPParameters.set_unit_sigma``; the load runs before any closure
    is known). A number is the 1.01 stand-in of :func:`apply_unit_stand_in` instead
    (A/B against the exact form). The shares are recalibrated either way."""
    d = Path(dataset_dir)
    prm_path = d / prm if (d / prm).exists() else Path(prm)
    baserate = d / "baserate.har"
    p = GTAPParameters()
    p.load_from_har(
        basedata_path=d / "basedata.har",
        sets_path=d / "sets.har",
        default_path=prm_path,
        baserate_path=baserate if baserate.exists() else None,
    )
    if unit_sigma is None:
        changed = p.elasticities.restore_unit_production_elasticities()
    else:
        changed = apply_unit_stand_in(p, unit_sigma, recalibrate=False)
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
    prm: str | Path = "altertax.prm",
    shock: dict = ALTAX_SHOCK,
    cache_root: Path | None = None,
    params: GTAPParameters | None = None,
    trade_balance: str = "nominal",
    decomp_steps: int | str | None = None,
) -> AltertaxRun:
    """Base, check and shock of ALTERTAX on ``dataset_dir`` (``params`` overrides the
    load, e.g. with other elasticities).

    ``trade_balance`` is the closure of the external balance of every non-residual region:
    ``"nominal"`` (default, ``altax.exp`` of nus333 Burfisher: ``del_tbal = 0``) holds
    ``savf`` at its check value; ``"ratio"`` (RunGTAP's ``del_tbalry = 0``) holds
    ``savf / world income``. The residual region (``cgdslack``) is the same in both.

    ``decomp_steps``: ``None`` (default) leaves the welfare decomposition to the one-step
    :func:`decomp.decomp_headers`, exact to first order (the Burfisher ALTAX shock is
    ~1e-5). An int K, or ``"auto"`` (K doubled until the EV settles), solves the shock
    path in K sub-steps and integrates the decomposition along it
    (:func:`path.solve_decomp_path`, ``run.decomp_path``); the run keeps its solver
    session (``run.result.session``) for that."""
    if trade_balance not in ("nominal", "ratio"):
        raise ValueError(
            f"trade_balance={trade_balance!r}: expected 'nominal' or 'ratio'"
        )
    p = load_altertax_params(dataset_dir, prm) if params is None else params
    _check_cells(p, shock)
    t0 = time.perf_counter()
    res = solve_gtap_family(
        p,
        altertax_closure(),
        ALTAX_RESIDUAL_REGION,
        shock=shock,
        cache_root=cache_root,
        # GEMPACK's del_tbal(USA) = 0: the nominal balance, not savf = pigbl * savf_bar;
        # "ratio": del_tbalry(r) = 0 (savf / world income)
        nominal_trade_balance=trade_balance,
        keep_session=decomp_steps is not None,
    )
    minutes = (time.perf_counter() - t0) / 60.0
    codes = {k: int(res.results[k]["code"]) for k in ("check", "shock")}
    run = AltertaxRun(p, res, codes, minutes, Path(dataset_dir))
    if decomp_steps is not None:
        from equilibria.family.altertax.path import solve_decomp_path

        run.decomp_path = solve_decomp_path(run, shock, decomp_steps)
    return run
