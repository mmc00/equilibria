"""Lever A — eliminate calibrate_base's double solve.

HARD GATE: settle_only (cut the settle at check) must produce a settled_seed
IDENTICAL to the full base->check->shock settle. Baseline captured on gtap7_10x7.
"""

from __future__ import annotations

import contextlib
import gzip
import hashlib
import json
import os
from pathlib import Path

import pytest

from equilibria.blocks.gtap.factor import FactorBlock
from equilibria.templates.gtap import GTAPParameters
from equilibria.templates.gtap.gtap_contract import GTAPClosureConfig

pytestmark = pytest.mark.needs_path

DATA = Path("datasets/gtap7_10x7")
# Full base->check->shock settle baseline (gtap7_10x7).
# settle_only must reproduce this EXACTLY.
#
# Recapturado 2026-09-18. El valor previo (20138, ded5651b6133f788) se tomo el
# 2026-08-25 y caduco con `eacc53f` (2026-09-07), que NOMBRO los agregados Fisher
# cruzados: dejaron de ser expresiones inline y pasaron a ser Vars (Hessiano 13.3x
# mas disperso). El seed las incluye porque ahora SON variables.
#
# Medido celda a celda contra el ultimo commit que reproduce la firma vieja
# (769bcf3): de las 20138 originales no desaparecio NINGUNA, la peor diferencia
# relativa es 6.4e-11 (ruido de convergencia), y las 56 que se suman son
# exactamente los agregados nuevos -- mq_abs_{bs,sb,ss}, mq_factr_{bs,sb,ss} y
# mq_gdp_{bs,sb}, 7 regiones cada uno. El corte sigue sin mover el seed, que es
# lo que este gate existe para vigilar.
BASELINE_COUNT = 20194
# Recalculada al pasar a %.13g (antes %.10e -> b82d6f9b530a70bf en macOS y
# d29fceb45a79c2f2 en Linux, con el MISMO conteo: la diferencia era el ultimo
# bit, no el seed). El conteo no cambia: 20194.
BASELINE_SIG = "6e261fe5b8773a5c"


def _load_params():
    p = GTAPParameters()
    p.load_from_har(
        basedata_path=DATA / "basedata.har",
        sets_path=DATA / "sets.har",
        default_path=DATA / "default.prm",
        baserate_path=DATA / "baserate.har",
    )
    return p


def _closure(p):
    return GTAPClosureConfig(
        name="base",
        closure_type="MCP",
        capital_mobility="sluggish",
        fix_endowments=False,
        fix_taxes=False,
        fix_technology=False,
        if_sub=False,
        savf_flag="capFix",
        numeraire="pnum",
    )


# Tolerancia relativa del gate del seed. El ruido REAL aqui es de CONVERGENCIA
# del solver, no de 1-2 ULP: la nota de BASELINE_COUNT lo midio celda a celda
# contra 769bcf3 y la peor diferencia era 6.4e-11. 1e-9 deja ~2 ordenes de
# margen sobre eso y sigue siendo ~7 ordenes mas estricto que cualquier cambio
# real del seed (el que cazo eacc53f movio celdas enteras, no decimales).
SEED_TOL = 1e-9
BASELINE_SEED = (
    Path(__file__).resolve().parents[2] / "fixtures/gtap7_10x7_settled_seed.json.gz"
)


def _seed_signature(seed):
    """Firma de un seed. Solo para comparar DOS seeds de la MISMA corrida
    (cache hit vs miss): ahi el bit-a-bit es exacto y es lo que se quiere.
    El gate contra el baseline NO la usa —cruza plataformas— - ver _seed_diffs.
    """
    parts = [
        f"{name}|{body}|{float(val):.13g}"
        for name, cells in seed.items()
        for body, val in cells.items()
    ]
    parts.sort()
    return len(parts), hashlib.sha256("\n".join(parts).encode()).hexdigest()[:16]


def _flat(seed):
    return {
        f"{name}|{body}": float(val)
        for name, cells in seed.items()
        for body, val in cells.items()
    }


def _load_baseline():
    with gzip.open(BASELINE_SEED, "rt") as fh:
        return json.load(fh)


def _seed_diffs(actual: dict, baseline: dict, tol: float = SEED_TOL):
    """Celdas que difieren mas de `tol` relativo, mas las que sobran/faltan."""
    faltan = sorted(set(baseline) - set(actual))
    sobran = sorted(set(actual) - set(baseline))
    movidas = []
    for k in set(actual) & set(baseline):
        a, b = actual[k], baseline[k]
        if a == b:
            continue
        denom = max(abs(a), abs(b))
        rel = abs(a - b) / denom if denom else abs(a - b)
        if rel > tol:
            movidas.append((k, b, a, rel))
    movidas.sort(key=lambda t: -t[3])
    return faltan, sobran, movidas


def _calibrate(**kw):
    p = _load_params()
    rr = list(p.sets.r)[-1]
    fb = FactorBlock(sets=p.sets, params=p)
    return fb.calibrate_base(p, p.sets, _closure(p), rr, ref_gdx=None, **kw)


@pytest.mark.skipif(not DATA.exists(), reason="gtap7_10x7 dataset not present")
def test_full_settle_baseline():
    os.environ["EQUILIBRIA_SEED_CACHE_DISABLE"] = "1"
    seed = _calibrate()
    count, h = _seed_signature(seed)
    assert count > 0, "settle produced an empty seed"
    print(f"BASELINE settled_seed: {count} cells, sig={h}")


def _solve_multiperiod_result(settle_only):
    """Build the block model + solve, return the results dict (base/check/[shock])."""
    from equilibria.templates.gtap.gtap_block_model import (
        build_block_model,
        solve_block_model,
    )

    p = _load_params()
    rr = list(p.sets.r)[-1]
    m, mp = build_block_model(p, p.sets, _closure(p), rr)
    return solve_block_model(
        m, p, _closure(p), None, mode="gtap", settle_only=settle_only
    )


@pytest.mark.skipif(not DATA.exists(), reason="gtap7_10x7 dataset not present")
def test_settle_only_skips_shock():
    os.environ["EQUILIBRIA_SEED_CACHE_DISABLE"] = "1"
    res = _solve_multiperiod_result(settle_only=True)
    assert "check" in res, "settle_only must still solve the check phase"
    assert "shock" not in res, "settle_only must NOT solve the shock phase"


@pytest.mark.skipif(not DATA.exists(), reason="gtap7_10x7 dataset not present")
def test_settle_only_seed_identical_to_full():
    os.environ["EQUILIBRIA_SEED_CACHE_DISABLE"] = "1"
    # calibrate_base's seed must equal the full-settle baseline (Task 0 constants).
    # Before Step 3 this is trivially true (calibrate_base is still full-settle);
    # after Step 3 (calibrate_base uses settle_only) it is the byte-identical gate.
    actual = _flat(_calibrate())
    baseline = _load_baseline()

    # El conteo sigue siendo gate DURO: una celda de mas o de menos es un cambio
    # estructural del seed, no ruido.
    assert len(actual) == BASELINE_COUNT, (
        f"HARD GATE: calibrate_base seed tiene {len(actual)} celdas, "
        f"baseline {BASELINE_COUNT}. The cut changed the seed. STOP."
    )

    faltan, sobran, movidas = _seed_diffs(actual, baseline)
    assert not faltan and not sobran, (
        f"HARD GATE: el seed cambio de CLAVES. faltan={faltan[:5]} "
        f"sobran={sobran[:5]}. The cut changed the seed. STOP."
    )
    assert not movidas, (
        "HARD GATE: calibrate_base seed != full-settle baseline en "
        f"{len(movidas)} celda(s) por encima de {SEED_TOL:g} relativo. "
        f"Peores: {[(k, f'{b:.12g}->{a:.12g}', f'{r:.2e}') for k, b, a, r in movidas[:5]]}. "
        "The cut changed the seed. STOP."
    )


@contextlib.contextmanager
def _monkey(obj, name, new):
    orig = getattr(obj, name)
    setattr(obj, name, new)
    try:
        yield
    finally:
        setattr(obj, name, orig)


def _sparse_solution(force_full_settle):
    from pyomo.environ import Var
    from pyomo.environ import value as V

    from equilibria.templates.gtap.gtap_block_model import solve_block_model as _sbm
    from equilibria.templates.gtap.gtap_multiperiod_driver import solve_multiperiod
    from equilibria.templates.gtap_sparse.multiperiod import build_sparse_model_mp

    p = _load_params()
    rr = list(p.sets.r)[-1]
    patch = contextlib.ExitStack()
    if force_full_settle:
        # calibrate_base imports solve_block_model LOCALLY from gtap_block_model, so
        # patch it at the source module (not factor). Force settle_only=False so the
        # settle runs the full base->check->shock stack (the pre-lever-A behavior).
        import equilibria.templates.gtap.gtap_block_model as _bm

        def _full(m, params, closure, ref_gdx, *, mode="gtap", settle_only=False):
            return _sbm(m, params, closure, ref_gdx, mode=mode, settle_only=False)

        patch.enter_context(_monkey(_bm, "solve_block_model", _full))
    with patch:
        m, mp, _ = build_sparse_model_mp(
            p, p.sets, _closure(p), rr, base_calibrated=True
        )
        solve_multiperiod(
            m,
            p,
            _closure(p),
            ref_gdx=None,
            skip_base_solve=True,
            mute_welfare=True,
            seed_from_prior=False,
            holdfix_cd=True,
            mode="gtap",
        )
    out = {}
    for v in m.component_objects(Var, active=True):
        for idx in v:
            with contextlib.suppress(Exception):
                out[f"{v.name}{idx}"] = float(V(v[idx]))
    return out


@pytest.mark.skipif(not DATA.exists(), reason="gtap7_10x7 dataset not present")
def test_end_to_end_solve_parity():
    os.environ["EQUILIBRIA_SEED_CACHE_DISABLE"] = "1"
    sol_cut = _sparse_solution(force_full_settle=False)
    sol_full = _sparse_solution(force_full_settle=True)
    shared = set(sol_cut) & set(sol_full)
    assert shared, "no shared var keys"
    worst, key = 0.0, None
    for k in shared:
        rel = abs(sol_cut[k] - sol_full[k]) / (abs(sol_full[k]) + 1e-12)
        if rel > worst:
            worst, key = rel, k
    assert worst < 1e-8, f"final solve diverged at {key}: rel={worst:.2e}"


# ---------------------------- disk cache ---------------------------- #


def test_seed_cache_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setenv("EQUILIBRIA_SEED_CACHE", str(tmp_path))
    monkeypatch.delenv("EQUILIBRIA_SEED_CACHE_DISABLE", raising=False)
    from equilibria.blocks.gtap import seed_cache

    seed = {
        "pf": {("USA", "Land", "Food"): 1.25, ("EU", "Land", "Food"): 0.9},
        "kstock": {"USA": 42.0},
    }
    key = "k-abc123"
    assert seed_cache.load(key) is None
    seed_cache.save(key, seed)
    got = seed_cache.load(key)
    assert got == seed, f"roundtrip mismatch: {got} != {seed}"


def test_seed_cache_roundtrip_preserves_key_types():
    """A cached key must come back as the SAME object it went in as.

    The old encoder joined the tuple with \\x1f and split it back, which turned
    every element into a str and collapsed a 1-tuple into a bare scalar.  The
    consumer (gtap_multiperiod_driver, base-calibrated seeding) looks the key up
    inside ``except (KeyError, TypeError, ValueError): pass`` — so a mistyped key
    does not raise, it silently seeds NOTHING and the model solves from a
    different starting point.  That is the failure this pins.
    """
    from equilibria.blocks.gtap import seed_cache

    for key in [
        ("USA", 2020),  # int element -> came back as "2020"
        ("USA",),  # 1-tuple -> came back as the bare str "USA"
        (2020,),
        ("USA", "Land", "Food"),  # the all-str case that always worked
        "USA",  # bare scalar stays a bare scalar
        42,
    ]:
        enc = seed_cache._enc_key(key)
        assert isinstance(enc, str), f"{key!r} must encode to a str, got {enc!r}"
        got = seed_cache._dec_key(enc)
        assert got == key, f"key round-trip lost information: {key!r} -> {got!r}"
        assert type(got) is type(key), (
            f"key round-trip changed the type: {key!r} ({type(key).__name__}) "
            f"-> {got!r} ({type(got).__name__})"
        )


def test_seed_cache_roundtrip_through_disk_preserves_key_types(tmp_path, monkeypatch):
    """End-to-end: save() then load() must hand back the identical key objects."""
    monkeypatch.setenv("EQUILIBRIA_SEED_CACHE", str(tmp_path))
    monkeypatch.delenv("EQUILIBRIA_SEED_CACHE_DISABLE", raising=False)
    from equilibria.blocks.gtap import seed_cache

    seed = {
        "pf": {("USA", "Land", "Food"): 1.25, ("EU", "Land", "Food"): 0.9},
        "kstock": {"USA": 42.0, ("EU",): 7.0},
        "xp": {("USA", 2020): 3.5},
    }
    seed_cache.save("k-types", seed)
    got = seed_cache.load("k-types")
    assert got == seed, f"disk round-trip mismatch: {got} != {seed}"


def test_seed_cache_reads_a_legacy_cache_file(tmp_path, monkeypatch):
    """A cache written by the OLD \\x1f encoder must still load, not be dropped.

    Users have these files sitting in ~/.cache already.  Their keys are all
    strings -- the one case the old encoding round-tripped correctly -- so they
    are still valid; refusing to read them would silently re-run every settle.
    """
    import json as _json

    monkeypatch.setenv("EQUILIBRIA_SEED_CACHE", str(tmp_path))
    monkeypatch.delenv("EQUILIBRIA_SEED_CACHE_DISABLE", raising=False)
    from equilibria.blocks.gtap import seed_cache

    legacy = {"pf": {"USA\x1fLand\x1fFood": 1.25, "USA": 42.0, "2020": 7.0}}
    (tmp_path / "k-legacy.json").write_text(_json.dumps(legacy))

    got = seed_cache.load("k-legacy")
    assert got == {"pf": {("USA", "Land", "Food"): 1.25, "USA": 42.0, "2020": 7.0}}, got
    # "2020" was written by an encoder that stringified everything, so it must come
    # back as the STRING it was -- decoding it as int 2020 would be the same
    # silent-mistype bug this module was fixed for.
    assert "2020" in got["pf"], (
        f"legacy numeric-looking key mistyped: {list(got['pf'])}"
    )


def test_seed_cache_file_is_versioned(tmp_path, monkeypatch):
    """New files carry a format marker, so the reader never has to guess.

    Without it, load() would have to sniff each key -- and a legacy "2020" is
    valid JSON, so sniffing silently turns it into an int.
    """
    import json as _json

    monkeypatch.setenv("EQUILIBRIA_SEED_CACHE", str(tmp_path))
    monkeypatch.delenv("EQUILIBRIA_SEED_CACHE_DISABLE", raising=False)
    from equilibria.blocks.gtap import seed_cache

    seed_cache.save("k-ver", {"pf": {("USA", "Land"): 1.0}})
    raw = _json.loads((tmp_path / "k-ver.json").read_text())
    assert raw.get("_fmt") == 2, f"missing/!=2 format marker: {raw!r}"
    assert seed_cache.load("k-ver") == {"pf": {("USA", "Land"): 1.0}}


def test_seed_cache_key_changes_with_input():
    from equilibria.blocks.gtap import seed_cache

    p = _load_params()
    rr = list(p.sets.r)[-1]
    c1 = GTAPClosureConfig(
        name="base",
        closure_type="MCP",
        capital_mobility="sluggish",
        fix_endowments=False,
        fix_taxes=False,
        fix_technology=False,
        if_sub=False,
        savf_flag="capFix",
        numeraire="pnum",
    )
    c2 = GTAPClosureConfig(
        name="base",
        closure_type="MCP",
        capital_mobility="sluggish",
        fix_endowments=False,
        fix_taxes=False,
        fix_technology=False,
        if_sub=True,
        savf_flag="capFix",
        numeraire="pnum",
    )  # if_sub differs
    k1 = seed_cache.cache_key("gtap7_10x7", c1, rr, p)
    k2 = seed_cache.cache_key("gtap7_10x7", c2, rr, p)
    assert k1 != k2, "cache key must change when a settle-affecting input changes"


def test_seed_cache_disabled_is_noop(tmp_path, monkeypatch):
    monkeypatch.setenv("EQUILIBRIA_SEED_CACHE", str(tmp_path))
    monkeypatch.setenv("EQUILIBRIA_SEED_CACHE_DISABLE", "1")
    from equilibria.blocks.gtap import seed_cache

    assert seed_cache.disabled() is True
    seed_cache.save("k-x", {"pf": {("USA",): 1.0}})  # must write nothing
    assert list(tmp_path.iterdir()) == [], "disabled cache still wrote a file"
    assert seed_cache.load("k-x") is None, "disabled cache still read"


@pytest.mark.skipif(not DATA.exists(), reason="gtap7_10x7 dataset not present")
def test_cache_hit_skips_settle(tmp_path, monkeypatch):
    monkeypatch.setenv("EQUILIBRIA_SEED_CACHE", str(tmp_path))
    monkeypatch.delenv("EQUILIBRIA_SEED_CACHE_DISABLE", raising=False)
    seed1 = _calibrate()  # miss → computes + writes
    # calibrate_base imports build_block_model LOCALLY from gtap_block_model, so
    # spy at the source module (patching factor would never see the call).
    import equilibria.templates.gtap.gtap_block_model as _bm

    called = {"n": 0}
    orig = _bm.build_block_model

    def _spy(*a, **k):
        called["n"] += 1
        return orig(*a, **k)

    monkeypatch.setattr(_bm, "build_block_model", _spy)
    seed2 = _calibrate()  # hit → must NOT build/solve
    assert called["n"] == 0, "cache hit still built the model"
    assert _seed_signature(seed1) == _seed_signature(seed2), "cached seed differs"
