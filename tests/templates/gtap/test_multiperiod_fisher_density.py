"""The Fisher rows must not carry a quadratic Hessian.

``build_equations_fisher`` deletes the block's ``eq_pabs``/``eq_pfact``/``eq_pwfact``
and rebuilds them across periods. Rebuilt with the wide sums expanded inline under a
square root, ONE row couples every ``pf``/``xf`` in the model with every other: the
second derivative of ``sqrt(f(Σ)·g(Σ))`` crosses each pair, so the row's Hessian grows
as the square of its width.

Measured on the shipped datasets before this contract existed (attribution per equation,
inside the solve): ``eq_pwfact`` alone was 30,625 nnz at 5x5 and 233,289 at 10x7 — 40.6%
of the whole Hessian in a single row, while ``eq_mfw_ss`` (the same sum, named by the
block) was 462. At 20x41 the full Hessian reached 247,390,965 nnz and IPOPT died with
``EXIT: Not enough memory.`` after 21 GB.

Naming each cross-period aggregate with a variable and its defining row keeps the wide
part bilinear (Hessian linear in width) and leaves the square root over a handful of
scalars. This test pins that: the per-row Hessian must grow SUBQUADRATICALLY with the
dataset width.
"""

import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts/gtap"))

np = pytest.importorskip("numpy")
pytest.importorskip("pyomo.contrib.pynumero.interfaces.pyomo_nlp")

DATASETS_DIR = ROOT / "datasets"
# The rows build_equations_fisher rebuilds across periods, plus eq_rgdpmp, which
# carries the same wide-sum-under-a-root shape via _mqgdp.
FISHER_ROWS = ("eq_pwfact", "eq_pfact", "eq_pabs", "eq_rgdpmp")


def _hessian_nnz_by_equation(m):
    """nnz each equation block contributes to the Hessian of the Lagrangian.

    Counts VALUES, not the ``.nl`` structure: with the dual set to 1 on one block and 0
    everywhere else, whatever stays nonzero is what THAT block contributes. Reading
    ``.nnz`` instead would return the fixed sparsity pattern of the whole model.
    """
    from collections import defaultdict

    from pyomo.contrib.pynumero.interfaces.pyomo_nlp import PyomoNLP

    nlp = PyomoNLP(m)
    nrows = nlp.n_constraints()
    nlp.set_primals(nlp.get_primals())

    by_eq = defaultdict(list)
    for k, cd in enumerate(nlp.get_pyomo_constraints()):
        by_eq[cd.name.split("[")[0]].append(k)

    out = {}
    for name, idxs in by_eq.items():
        duals = np.zeros(nrows)
        duals[idxs] = 1.0
        nlp.set_duals(duals)
        H = nlp.evaluate_hessian_lag().tocoo()
        out[name] = int((np.abs(H.data) > 0).sum())
    return out


def _build_mp(dataset: str):
    """Build the multi-period model the way the parity gate does, without solving."""
    from equilibria.templates.gtap import GTAPParameters
    from equilibria.templates.gtap.gtap_model_multiperiod import GTAPMultiPeriodModel

    d = DATASETS_DIR / dataset
    p = GTAPParameters()
    p.load_from_har(
        basedata_path=d / "basedata.har",
        sets_path=d / "sets.har",
        default_path=d / "default.prm",
        baserate_path=d / "baserate.har",
    )
    mp = GTAPMultiPeriodModel(p.sets, p, None, residual_region=list(p.sets.r)[-1])
    m = mp.build_sets()
    mp.build_vars(m)
    for t in ("base", "check", "shock"):
        mp.build_equations_intra(m, t)
    mp.build_equations_fisher(m)
    # PyomoNLP requires exactly one objective; the MP model is a square system with
    # none. A constant objective adds nothing to the Hessian of the Lagrangian, so
    # the per-equation attribution below is unaffected.
    from pyomo.environ import Objective

    m._density_probe_obj = Objective(expr=0.0)
    return m


@pytest.mark.integration
@pytest.mark.parametrize("dataset", ["gtap7_3x3", "gtap7_5x5"])
def test_fisher_rows_exist(dataset):
    """Guard the measurement itself: renaming a row would make the density test vacuous."""
    if not (DATASETS_DIR / dataset / "basedata.har").exists():
        pytest.skip(f"{dataset} not shipped")
    m = _build_mp(dataset)
    for name in FISHER_ROWS:
        assert getattr(m, name, None) is not None, f"{name} missing from the MP model"


@pytest.mark.integration
def test_fisher_hessian_grows_subquadratically():
    """3x3 → 5x5 widens the factor sums ~2.8x; a quadratic row would grow ~7.8x.

    The bound is 4.0x per row: comfortably above the ~2.8x a correctly-named
    (bilinear) aggregate costs, and far below the ~7.8x an inlined sum under a root
    costs. Measured before the fix, eq_pwfact grew 7.6x from 5x5 to 10x7.
    """
    for ds in ("gtap7_3x3", "gtap7_5x5"):
        if not (DATASETS_DIR / ds / "basedata.har").exists():
            pytest.skip(f"{ds} not shipped")

    small = _hessian_nnz_by_equation(_build_mp("gtap7_3x3"))
    large = _hessian_nnz_by_equation(_build_mp("gtap7_5x5"))

    failures = []
    for name in FISHER_ROWS:
        n_small, n_large = small.get(name, 0), large.get(name, 0)
        if n_small == 0:
            continue
        ratio = n_large / n_small
        if ratio > 4.0:
            failures.append(f"{name}: {n_small:,} -> {n_large:,} nnz ({ratio:.1f}x)")

    assert not failures, (
        "Fisher rows carry a quadratic Hessian — the wide sums are inlined under the "
        "square root instead of being named by auxiliary variables:\n  "
        + "\n  ".join(failures)
    )
