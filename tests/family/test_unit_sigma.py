"""Exact sigma = 1 (Cobb-Douglas) nests under ``GTAPClosureConfig.unit_sigma``.

``unit_sigma="gams"`` (default) keeps the 1.01 guards of GAMS #192; ``"gempack"`` writes
every CES nest whose elasticity is exactly 1 in its exact CD price form

    log(P) == c + sum_i w_i * log(q_i),   w_i = alpha_i / S,  S = sum_i alpha_i

with ``c`` taken from the base prices of the calibration that built ``alpha``. The
ALTERTAX closure sets ``"gempack"``; its default parameter load keeps the unit
elasticities at exactly 1.

``golden/unit_sigma_rows.json`` pins the recorded rows of the default closure (and of
the ALTERTAX route with the explicit 1.01 stand-in) as they were before the switch
existed: run this file as a script to rewrite it (only ever from the commit before the
switch).
"""

from __future__ import annotations

import hashlib
import json
import math
from functools import cache
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("jax")

GOLDEN = Path(__file__).parent / "golden" / "unit_sigma_rows.json"
ALTAX_DATASETS = ("nus333", "gtap7_10x7")
RUNGTAP = "gempack_altertax-rungtap_tms-rate10_s8-16-32"

# the price duals that take the CD form at sigma = 1, with the elasticity each reads
CD_FAMILIES = (
    "eq_pxeq",
    "eq_pndeq",
    "eq_pvaeq",
    "eq_paa",
    "eq_pmteq",
    "eq_xs",
    "eq_pg",
    "eq_ptmg",
)


def _data_dir(dataset: str) -> Path:
    from equilibria.sources import dataset_dir

    return dataset_dir(dataset)


def _altax_params(dataset: str, **kw):
    """ALTERTAX parameters of ``dataset``: nus333's own ``altertax.prm``; elsewhere the
    RunGTAP reference folder's ``parameters.prm`` (as the measure script loads them)."""
    from equilibria.family.altertax.run import load_altertax_params

    d = _data_dir(dataset)
    if dataset == "nus333":
        return load_altertax_params(d, **kw)
    return load_altertax_params(d, d / RUNGTAP / "parameters.prm", **kw)


def _default_closure():
    from equilibria.templates.gtap.gtap_contract import GTAPClosureConfig

    return GTAPClosureConfig(
        name="base",
        closure_type="MCP",
        capital_mobility="sluggish",
        fix_endowments=False,
        fix_taxes=False,
        fix_technology=False,
        if_sub=True,
        savf_flag="capFix",
        numeraire="pnum",
    )


def _record(params, closure, residual_region: str):
    """``(backend, layout, system)`` of the single-period family, built the way
    ``family.driver.prepare_run`` builds it (the closure's switches on params first)."""
    from equilibria.backends.family_backend import FamilyBackend
    from equilibria.templates.gtap.gtap_block_model import compose_block_state

    params.set_va_subsidy_basis(getattr(closure, "va_subsidy_basis", "gams"))
    if hasattr(params, "set_unit_sigma"):
        params.set_unit_sigma(getattr(closure, "unit_sigma", "gams"))
    model, state = compose_block_state(params, params.sets, closure, residual_region)
    backend = FamilyBackend()
    backend.build(model, skip=state.ifsub_skipped)
    layout = backend.layout(state)
    return backend, layout, backend.system(layout)


def _h(*arrays) -> str:
    h = hashlib.sha256()
    for a in arrays:
        if isinstance(a, (list, tuple)):
            h.update("\x1f".join(map(str, a)).encode())
        else:
            h.update(np.ascontiguousarray(np.asarray(a, dtype=float)).tobytes())
        h.update(b"\x1e")
    return h.hexdigest()[:32]


def _snapshot(backend, layout, system) -> dict:
    """Per family: hash of its row labels and of F at the seed and at a perturbed point;
    plus the columns, the fixed values and J at the perturbed point."""
    z0 = np.asarray(layout.free_values, float)
    zp = z0 * (1.0 + 0.01 * np.sin(np.arange(z0.size) + 1.0))
    f0, fp = system.F(z0), system.F(zp)
    names = system.row_names
    fams: dict[str, str] = {}
    start = 0
    for f in system.families:
        stop = start + f.n_rows
        fams[f.name] = _h(names[start:stop], f0[start:stop], fp[start:stop])
        start = stop
    jac = system.jac(zp).tocsr()
    jac.sort_indices()
    return {
        "families": fams,
        "columns": _h(layout.col_names),
        "fixed": _h(layout.fixed_values),
        "jac": _h(jac.indptr, jac.indices, jac.data),
    }


def _golden_cases():
    """name -> (params, closure, residual region) of each pinned case."""
    from tests.family._data import load_params

    out = {}
    for ds in ("gtap7_3x3", "gtap7_10x7"):
        p = load_params(ds)
        out[f"{ds}_default"] = (p, _default_closure(), str(list(p.sets.r)[-1]))
    from equilibria.family.altertax.closure import (
        ALTAX_RESIDUAL_REGION,
        altertax_closure,
    )

    out["nus333_altertax_s1.01"] = (
        _altax_params("nus333", unit_sigma=1.01),
        altertax_closure(),
        ALTAX_RESIDUAL_REGION,
    )
    return out


@cache
def _altax_system(dataset: str):
    """The ALTERTAX family on ``dataset`` with the route's default load (exact 1s)."""
    from equilibria.family.altertax.closure import altertax_closure

    p = _altax_params(dataset)
    rr = "ROW" if dataset == "nus333" else str(list(p.sets.r)[-1])
    return (p, *_record(p, altertax_closure(), rr))


# ----------------------------------------------------------------------------------
# 0. the switch
# ----------------------------------------------------------------------------------


def test_closure_field_defaults_to_gams():
    from equilibria.templates.gtap.gtap_contract import GTAPClosureConfig

    assert GTAPClosureConfig().unit_sigma == "gams"
    assert GTAPClosureConfig(unit_sigma="gempack").unit_sigma == "gempack"
    with pytest.raises(ValueError):
        GTAPClosureConfig.model_validate({"unit_sigma": "exact"})


def test_altertax_closure_is_gempack_and_gtap7_gempack_is_not():
    from equilibria.family.altertax.closure import altertax_closure
    from equilibria.templates.gtap.gtap_contract import build_gtap_closure_config

    assert altertax_closure().unit_sigma == "gempack"
    assert build_gtap_closure_config("gtap7_gempack").unit_sigma == "gams"


def test_unit_sigma_is_in_the_cache_key():
    from equilibria.blocks.gtap.fingerprint import _closure_digest

    c = _default_closure()
    g = c.model_copy(update={"unit_sigma": "gempack"})
    assert _closure_digest(c) != _closure_digest(g)


@pytest.mark.parametrize("dataset", ALTAX_DATASETS)
def test_altertax_default_load_keeps_exact_ones(dataset):
    """No 1.01 anywhere in the route's default load: every nest ALTERTAX sets to 1
    is exactly 1 (the production nests the loader guarded are restored)."""
    el = _altax_params(dataset).elasticities
    for nest in ("sigmap", "sigmand", "sigmav", "esubd", "esubm", "sigmas"):
        vals = set(getattr(el, nest).values())
        assert vals == {1.0}, (nest, sorted(vals)[:4])
    assert set(el.esubg.values()) == {1.0}
    assert set(el.sigmam.values()) == {1.0}


@pytest.mark.parametrize("dataset", ALTAX_DATASETS)
def test_altertax_stand_in_is_still_selectable(dataset):
    """``unit_sigma=1.01`` keeps today's A/B: no elasticity left at exactly 1, also
    after the gempack closure is applied to the parameters."""
    p = _altax_params(dataset, unit_sigma=1.01)
    p.set_unit_sigma("gempack")
    el = p.elasticities
    for nest in ("sigmap", "sigmand", "sigmav", "esubd", "esubm", "sigmas"):
        assert 1.0 not in set(getattr(el, nest).values()), nest
    assert 1.0 not in set(el.esubg.values())
    assert 1.0 not in set(el.sigmam.values())


# ----------------------------------------------------------------------------------
# 1. the ALTERTAX family builds square and the CD rows hold at the seed
# ----------------------------------------------------------------------------------


@pytest.mark.parametrize("dataset", ALTAX_DATASETS)
def test_pmteq_and_xs_have_a_row_for_every_active_cell(dataset):
    p, backend, _, _ = _altax_system(dataset)
    s = p.sets
    pmt = {tuple(k) for k, _ in backend.records["eq_pmteq"]}
    assert pmt == {(str(r), str(i)) for r in s.r for i in s.i}
    gx, makb = p.calibrated.gx_param, p.benchmark.makb
    active = {
        (str(r), str(i))
        for r in s.r
        for i in s.i
        if any(
            gx.get((r, a, i), 0.0) > 0.0 or makb.get((r, a, i), 0.0) > 0.0
            for a in s.commodity_activities.get(i, list(s.a))
        )
    }
    xs = {tuple(k) for k, _ in backend.records["eq_xs"]}
    assert active and xs == active


@pytest.mark.parametrize("dataset", ALTAX_DATASETS)
def test_production_duals_build(dataset):
    _, backend, _, _ = _altax_system(dataset)
    for name in ("eq_pxeq", "eq_pndeq", "eq_pvaeq"):
        assert backend.records.get(name), name


@pytest.mark.parametrize("dataset", ALTAX_DATASETS)
def test_cd_rows_hold_at_the_base_seed(dataset):
    _, _, layout, system = _altax_system(dataset)
    f0 = system.F(np.asarray(layout.free_values, float))
    names = system.row_names
    seen, bad = set(), []
    for k, name in enumerate(names):
        fam = name.split("[")[0]
        if fam in CD_FAMILIES:
            seen.add(fam)
            if not abs(f0[k]) < 1e-12:
                bad.append((name, float(f0[k])))
    assert seen == set(CD_FAMILIES), sorted(set(CD_FAMILIES) - seen)
    assert not bad, (len(bad), sorted(bad, key=lambda t: -abs(t[1]))[:10])


@pytest.mark.parametrize("dataset", ALTAX_DATASETS)
def test_cd_rows_are_log_rows(dataset):
    """The CD rows carry ``log`` terms (not a 1 = 1 power row): eq_pndeq has one log term
    per input plus the log of its own price."""
    _, backend, _, _ = _altax_system(dataset)
    from equilibria.family.recorder import residual_terms

    def is_log(t) -> bool:  # log(x), or w * log(x)
        op = getattr(t, "op", None)
        return op == "log" or (op == "mul" and any(is_log(a) for a in t.args))

    _, res = backend.records["eq_pndeq"][0]
    terms = [t for _, t in residual_terms(res)]
    assert sum(map(is_log, terms)) >= 2, [getattr(t, "op", t) for t in terms]
    assert not any(getattr(t, "op", None) == "pow" for t in terms)


# ----------------------------------------------------------------------------------
# 2. the CD weights sum to one before normalising
# ----------------------------------------------------------------------------------


def _share_sums(p) -> dict[str, dict]:
    """S = sum alpha per CD nest, from the same sources the price duals read."""
    from equilibria.blocks.gtap import _derived_params as dp

    s = p.sets
    cal, sh = p.calibrated, p.shares
    out: dict[str, dict] = {k: {} for k in ("and_ava", "io", "af", "amw", "arm")}
    for k in ("ax", "g", "tmg"):
        out[k] = {}
    for r in s.r:
        for a in s.a:
            and_v = cal.and_param.get((r, a), 0.0)
            ava_v = cal.ava_param.get((r, a), 0.0)
            if and_v > 0.0 or ava_v > 0.0:
                out["and_ava"][(r, a)] = and_v + ava_v
            io = [cal.io_param.get((r, i, a), 0.0) for i in s.i]
            if any(v > 0.0 for v in io):
                out["io"][(r, a)] = sum(v for v in io if v > 0.0)
            af = [cal.af_param.get((r, f, a), 0.0) for f in s.f]
            if any(v > 0.0 for v in af):
                out["af"][(r, a)] = sum(v for v in af if v > 0.0)
    amw = sh.normalized.import_source_share
    for r in s.r:
        for i in s.i:
            vals = [amw.get((r, i, rp), 0.0) for rp in s.r]
            if any(v > 0.0 for v in vals):
                out["amw"][(r, i)] = sum(v for v in vals if v > 0.0)
            ax = [sh.p_ax.get((r, a, i), 0.0) for a in s.a]
            if any(v > 0.0 for v in ax):
                out["ax"][(r, i)] = sum(v for v in ax if v > 0.0)
    for key, (ad, am) in dp.armington_shares(p, s).items():
        if ad > 0.0 or am > 0.0:
            out["arm"][key] = ad + am
    g = dp.demand_income_params(p, s, "ROW")["g_share"]
    for r in s.r:
        vals = [g.get((r, i), 0.0) for i in s.i]
        if any(v > 0.0 for v in vals):
            out["g"][(r,)] = sum(v for v in vals if v > 0.0)
    for m in s.marg or s.m:
        vst = [dp._vst_value(p, str(r), str(m)) for r in s.r]
        if sum(vst) > 1e-12:
            out["tmg"][(m,)] = sum(v / sum(vst) for v in vst)
    return out


@cache
def _altax_share_sums(dataset: str) -> dict[str, dict]:
    p = _altax_params(dataset)
    p.set_unit_sigma("gempack")
    return _share_sums(p)


@pytest.mark.parametrize(
    "nest", ("and_ava", "io", "af", "amw", "arm", "ax", "g", "tmg")
)
@pytest.mark.parametrize("dataset", ALTAX_DATASETS)
def test_cd_share_sums_are_one(dataset, nest):
    by = _altax_share_sums(dataset)[nest]
    assert by, nest
    bad = sorted(
        ((k, v) for k, v in by.items() if abs(v - 1.0) > 1e-9),
        key=lambda kv: -abs(kv[1] - 1.0),
    )
    assert not bad, (len(bad), len(by), bad[:5])


def test_cd_coefficients_normalise():
    from equilibria.blocks.gtap._derived_params import cd_coefficients

    w, c = cd_coefficients([0.2, 0.6], [2.0, 0.5], p0=1.5)
    assert w == pytest.approx([0.25, 0.75], abs=1e-15)
    assert c == pytest.approx(
        math.log(1.5) - 0.25 * math.log(2.0) - 0.75 * math.log(0.5), abs=1e-15
    )


# ----------------------------------------------------------------------------------
# 3. the default closure (and the explicit 1.01 stand-in) records the same rows
# ----------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "case", ["gtap7_3x3_default", "gtap7_10x7_default", "nus333_altertax_s1.01"]
)
def test_rows_match_golden(case):
    gold = json.loads(GOLDEN.read_text())[case]
    params, closure, rr = _golden_cases()[case]
    snap = _snapshot(*_record(params, closure, rr))
    assert snap["columns"] == gold["columns"]
    assert snap["fixed"] == gold["fixed"]
    diff = sorted(
        set(snap["families"].items()) ^ set(gold["families"].items()),
        key=lambda kv: kv[0],
    )
    assert not diff, diff[:10]
    assert snap["jac"] == gold["jac"]


if __name__ == "__main__":
    data = {name: _snapshot(*_record(*case)) for name, case in _golden_cases().items()}
    GOLDEN.write_text(json.dumps(data, indent=1, sort_keys=True) + "\n")
    print(f"wrote {GOLDEN}: {list(data)}")
