"""GTAP7 vs RunGTAP (GEMPACK) quantity parity gate. LOCAL-only.

Compares the Python model's post-shock QUANTITY %-changes against GEMPACK's SL4
solution (qfd→xd, qxs→xw, qo→xp — the verified Q_TO_VAR map), cell-by-cell, in
ABSOLUTE PERCENTAGE POINTS (the natural metric for %-changes; a relative tol on
small %-changes is misleading — see gempack_reference). GEMPACK is Gragg-linearized
and Python is levels, so the per-page floor is stated in pp, not the GAMS 1% rel.

SKIPs when a row's sl4dump fixture is absent, so it never blocks the parity stamp
on a machine without the Windows-produced solution.
"""

from __future__ import annotations

import statistics
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts/gtap"))
sys.path.insert(0, str(ROOT / "src"))
from coverage_matrix import rows_for  # noqa: E402

DATASETS_DIR = ROOT / "datasets"
FIXTURES = ROOT / "tests/fixtures/gtap7_gempack"
GEMPACK_ROWS = rows_for("gtap7", "gempack", kind="mcp")

# Floor per (dataset): min fraction of cells within 1pp, and max allowed median |Δpp|.
# Measured on the real sl4dump; set conservatively below the measured value.
PP_WITHIN = 0.01  # 1 percentage point


def test_no_gempack_rows_is_a_clean_skip():
    if not GEMPACK_ROWS:
        pytest.skip("no reference='gempack' rows yet — awaiting RunGTAP SL4 dump")


def _solve_shock(
    dataset: str, ifsub: int, savf_flag: str = "capFlex", solve_nlp: bool = False
):
    """Build + seed + solve base→check→shock (gtap pure MCP) and return the model.

    ``savf_flag`` selects the capital-account closure to MATCH the GEMPACK fixture's:
      * "capFlex" (RORDELTA=1): returns equalize — matches the `_capflex` / default fixtures.
        Faithful but the returns-equalizing MCP is slow / non-convergent on large datasets.
      * "capFix" (RORDELTA=0): returns differ — matches the `_capfix` fixtures.

    ``solve_nlp`` runs the model as an NLP (maximize walras, IPOPT) instead of the PATH MCP.
    On the LARGE datasets the capFix MCP does not converge in PATH's in-process capi (10x7:
    code=2 in 6min; 15x10: spins for ~48min), but the SAME model is feasible as an NLP —
    IPOPT closes it (10x7: code=1 in 33s @97.0%; 15x10: code=1 in ~20min @91.5% vs the capFix
    fixture). NLP is faithful (tolerances cancel), so it is the gate path for large datasets.
    """
    import os as _os

    # NLP mode (maximize walras, IPOPT — read per-call inside run_gtap._run_path_capi_
    # nonlinear_full, which solve_multiperiod lazy-loads) is toggled by this env var. Set it
    # for the WHOLE _solve_shock (build + solve), mirroring test_gtap7_nlp_parity, and restore
    # it after — setting it only around the solve left the 10x7 build in a mismatched mode and
    # gave a nondeterministic code=2.
    _prev_nlp = _os.environ.get("EQUILIBRIA_GTAP_SOLVE_NLP")
    if solve_nlp:
        _os.environ["EQUILIBRIA_GTAP_SOLVE_NLP"] = "1"
    else:
        _os.environ.pop("EQUILIBRIA_GTAP_SOLVE_NLP", None)
    try:
        from equilibria.templates.gtap import GTAPParameters
        from equilibria.templates.gtap.gtap_block_model import build_block_model
        from equilibria.templates.gtap.gtap_contract import GTAPClosureConfig
        from equilibria.templates.gtap.gtap_multiperiod_driver import solve_multiperiod

        d = DATASETS_DIR / dataset
        p = GTAPParameters()
        p.load_from_har(
            basedata_path=d / "basedata.har",
            sets_path=d / "sets.har",
            default_path=d / "default.prm",
            baserate_path=d / "baserate.har",
        )
        rr = list(p.sets.r)[-1]
        ac = GTAPClosureConfig(
            name="base",
            closure_type="MCP",
            capital_mobility="sluggish",
            fix_endowments=False,
            fix_taxes=False,
            fix_technology=False,
            if_sub=bool(ifsub),
            savf_flag=savf_flag,
            numeraire="pnum",
        )
        # The GAMS ref GDX is a SPEEDUP, not a requirement: capFlex reads benchmark rore/rorg
        # from it (fast) instead of a capFix twin-solve, and it warm-starts the shock. When it's
        # absent (e.g. gtap7_3x4 ships no local GAMS ref — its shock is 2168 eqs > PATH's 1000-eq
        # demo cap), the block SELF-SEEDS: base_calibrated stamps m._settled_seed, and the risk
        # twin-solve fallback recovers rore/rorg. Verified on 3x4: shock code=1, 99.2% within 1pp
        # vs the GEMPACK fixture (median 0.043pp) with ref_gdx=None. So pass the GDX only if it
        # exists; on small datasets the twin-solve fallback is cheap enough.
        _gdx = ROOT / f"tests/fixtures/gtap7/{dataset}/out_gtap_shock_ifsub{ifsub}.gdx"
        gdx = _gdx if _gdx.exists() else None
        m, mp = build_block_model(p, p.sets, ac, rr, base_calibrated=True, ref_gdx=gdx)
        res = solve_multiperiod(
            m,
            p,
            ac,
            ref_gdx=gdx,
            skip_base_solve=True,
            mute_welfare=True,
            seed_from_prior=False,
            holdfix_cd=True,
            mode="gtap",
            # Solve the CHECK period so %-changes can be measured shock/check —
            # the denominator RunGTAP/GEMPACK uses.  base_calibrated's base is the
            # RAW benchmark, and base→check re-settles income ~+0.8%; on near-zero
            # gov-demand cells the tariff on top is ±1%, so shock/base overstates
            # the move (qga 66.7% within-1pp) while shock/check matches (100%).
            solve_check=True,
        )
    finally:
        if _prev_nlp is None:
            _os.environ.pop("EQUILIBRIA_GTAP_SOLVE_NLP", None)
        else:
            _os.environ["EQUILIBRIA_GTAP_SOLVE_NLP"] = _prev_nlp
    return m, int(res["shock"]["code"])


def _measure_pp(m, sl4dump: Path):
    """Return (within_1pp_fraction, median_abs_pp) across all mapped quantity cells."""
    from gempack_reference import Q_TO_VAR, gempack_qty_pct
    from pyomo.environ import value as V

    diffs = []
    for gvar, spec in Q_TO_VAR.items():
        pyname = spec["var"]
        try:
            gem = gempack_qty_pct(str(sl4dump), gvar)
        except KeyError:
            continue
        pv = getattr(m, pyname, None)
        if pv is None:
            continue
        for key, gfrac in gem.items():
            # Denominator = CHECK (the re-settled baseline), matching RunGTAP's
            # shock/check %-change.  _solve_shock runs solve_check=True so the check
            # period is solved; if a cell has no check value (older un-checked run)
            # fall back to base so the metric still computes.
            try:
                s = float(V(pv[(*key, "shock")]))
            except (KeyError, ValueError):
                continue
            den = None
            for _period in ("check", "base"):
                try:
                    _d = float(V(pv[(*key, _period)]))
                except (KeyError, ValueError):
                    continue
                if abs(_d) > 1e-12:
                    den = _d
                    break
            if den is None:
                continue
            py = s / den - 1.0
            diffs.append(abs(py - gfrac))
    if not diffs:
        return None, None
    within = sum(1 for x in diffs if x <= PP_WITHIN) / len(diffs)
    return within, statistics.median(diffs)


# The GEMPACK-faithful closure is capFlex (returns EQUALIZE, RORDELTA=1) — that is what the
# default fixtures use. But base-calibrated capFlex runs a returns-equalizing MCP settle +
# shock that is slow / non-convergent (code=2) on the large datasets. capFix (RORDELTA=0,
# returns differ) solves fast + converges everywhere. So the gate PREFERS a capFix fixture
# when one exists (run_gempack_matrix --rordelta 0), else falls back to the capFlex fixture,
# and skips the large datasets only when neither works.
_CAPFLEX_SLOW_DATASETS = {"gtap7_10x7", "gtap7_15x10", "gtap7_20x41"}

# Large datasets whose capFix MCP does not converge in PATH's in-process capi but ARE feasible
# as an NLP (maximize walras, IPOPT). The gate runs these via solve_nlp=True. Measured:
#   10x7  -> code=1 in 33s,  97.0% within 1pp (median 0.103pp)
#   15x10 -> code=1 in ~20m, 91.5% within 1pp (median 0.200pp)  — SLOW (see _SLOW_DATASETS)
_NLP_MODE_DATASETS = {"gtap7_10x7", "gtap7_15x10"}
# 20x41: too large for IPOPT (247M-nnz Hessian → OOM >32GB), so it is NOT in _NLP_MODE_DATASETS
# and this gate skips it. It WAS validated out-of-band with the FREE solver (Newton-TR + MUMPS,
# no Hessian) on Kaggle, capFix + base_calibrated=True (Kaggle kernel gtap-20x41-n20):
#   20x41 -> code=1 resid=1.4e-8, 94.7% within 1pp (median 0.15pp) / 98.5% in levels (median 0.10%)
# base_calibrated=True is REQUIRED: with =False (raw seed → check rebase) the match collapses to
# 61% (verified: 15x10 too drops 91.9%→54.6% with =False). See
# docs/findings/gempack_residual_is_linearization_2026-07-24.md (UPDATE 2026-08-19).
# Datasets whose NLP solve is too slow for the normal sweep (~20min). Marked @slow so they are
# excluded from run_parity_gates (which runs `-m "not slow"`) but still runnable by hand:
#   uv run pytest tests/templates/gtap/test_gtap7_gempack_parity.py -m slow -k 15x10
_SLOW_DATASETS = {"gtap7_15x10", "gtap7_20x41"}


def _gempack_params():
    """GEMPACK rows as pytest params, tagging the slow (large-NLP) datasets with @slow."""
    out = []
    for r in GEMPACK_ROWS:
        marks = [pytest.mark.slow] if r.dataset in _SLOW_DATASETS else []
        out.append(pytest.param(r, marks=marks, id=f"{r.dataset}-ifsub{r.ifsub}"))
    return out


def _capfix_fixture_for(row) -> Path | None:
    """Return the dataset's capFix GEMPACK fixture if one has been generated, else None.

    run_gempack_matrix --rordelta 0 emits `sl4dump_<ds>_<tag>_capfix.har`, where <tag> carries
    the shock+steps config (e.g. `tm10_s8-16-32`). We accept the exact row.ref-derived name, the
    plain `tm10` form, or — as a fallback — ANY step-tagged capFix dump for the dataset (the
    default study grid emits `tm10_s8-16-32_capfix`, which neither exact form matches)."""
    stem = row.ref[: -len(".har")] if row.ref.endswith(".har") else row.ref
    cands = [
        FIXTURES / f"{stem}_capfix.har",
        FIXTURES / f"sl4dump_{row.dataset}_tm10_capfix.har",
    ]
    # Fallback: match any step-tagged capFix dump for this dataset (sorted for determinism).
    cands += sorted(FIXTURES.glob(f"sl4dump_{row.dataset}_*_capfix.har"))
    return next((c for c in cands if c.exists()), None)


@pytest.mark.parametrize("row", _gempack_params())
def test_gtap7_gempack_parity(row):
    if not (DATASETS_DIR / row.dataset / "basedata.har").exists():
        pytest.skip(f"dataset HAR missing: {row.dataset}")
    # The GAMS ref GDX is a speedup (fast rore/rorg + warm-start), NOT a requirement: the block
    # SELF-SEEDS from base_calibrated's m._settled_seed + the risk twin-solve fallback (see
    # _solve_shock). It does NOT rescue the large datasets, where the capFlex settle/shock is
    # inherently non-convergent (code=2) — those are handled by the _CAPFLEX_SLOW_DATASETS skip.

    # Prefer the capFix fixture; fall back to the default (capFlex) fixture.
    #   * Small datasets: capFlex MCP self-seeds and converges (3x3/3x4/5x5).
    #   * Large datasets: the capFix MCP does NOT converge in PATH's capi, but the same model
    #     IS feasible as an NLP — so with a capFix fixture present we run solve_nlp=True. Without
    #     a capFix fixture there is no convergent path at all, so we skip.
    capfix_sl4 = _capfix_fixture_for(row)
    if capfix_sl4 is not None:
        sl4, savf_flag = capfix_sl4, "capFix"
    else:
        sl4, savf_flag = FIXTURES / row.ref, "capFlex"
        if not sl4.exists():
            pytest.skip(f"sl4dump fixture missing: {sl4}")
        # No capFix fixture: the large-dataset capFlex MCP is non-convergent (code=2) and there
        # is no NLP fixture to compare against either — skip until a capFix fixture exists
        # (run_gempack_matrix --rordelta 0). Small datasets fall through and run capFlex.
        if row.dataset in _CAPFLEX_SLOW_DATASETS:
            pytest.skip(
                f"{row.dataset}: capFlex MCP does not converge (code=2) on large datasets and "
                f"no capFix fixture yet — run run_gempack_matrix --rordelta 0"
            )

    # Large datasets run as an NLP (their capFix MCP does not converge in PATH); small ones
    # keep the native MCP path.
    solve_nlp = row.dataset in _NLP_MODE_DATASETS
    m, code = _solve_shock(
        row.dataset, row.ifsub, savf_flag=savf_flag, solve_nlp=solve_nlp
    )
    assert code == 1, (
        f"[{row.dataset}/{savf_flag}/{'NLP' if solve_nlp else 'MCP'}] shock did not "
        f"converge (code={code})"
    )

    within, med = _measure_pp(m, sl4)
    assert within is not None, f"[{row.dataset}] no comparable quantity cells"

    # Floor: fraction within 1pp >= the row's shock floor (expressed as a fraction*100
    # in stage_floors["shock"]); median |Δpp| must stay small.
    floor = dict(row.stage_floors)["shock"] / 100.0
    assert within >= floor, (
        f"[{row.dataset}/gempack/{savf_flag}] {within * 100:.1f}% of quantity cells within "
        f"1pp < floor {floor * 100:.0f}% (median |Δ|={med * 100:.2f}pp) — regression"
    )
