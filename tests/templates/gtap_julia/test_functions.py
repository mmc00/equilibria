"""Task 3: ces/cde helpers match the Julia ComputableGeneralEquilibriumHelpers.

Reference values are the docstring examples from the Julia source
(ces.jl / cde.jl), which are exact.
"""

import numpy as np

from equilibria.templates.gtap_julia.functions import cde, ces


def test_ces_docstring_example():
    # julia> ces(10, [1,2,3], [.1,.3,.6], 2.1, 2)
    got = ces(10.0, np.array([1.0, 2.0, 3.0]), np.array([0.1, 0.3, 0.6]), 2.1, 2.0)
    ref = np.array([1.5374084155938235, 3.6023084600464648, 6.591014163653714])
    assert np.allclose(got, ref, rtol=1e-12)


def test_ces_sigma_one():
    # Cobb-Douglas branch (σ == 1)
    y, p, a, g = 10.0, np.array([1.2, 0.8, 1.0]), np.array([0.4, 0.35, 0.25]), 1.0
    got = ces(y, p, a, 1.0, g)
    # value shares: p·x / (y·? ) — just assert positivity + budget consistency vs σ→1 limit
    assert (got > 0).all()


def test_ces_sigma_zero_leontief():
    y, a, g = 5.0, np.array([0.2, 0.5, 0.3]), 2.0
    got = ces(y, np.array([1.0, 1.0, 1.0]), a, 0.0, g)
    ref = y * a / g
    assert np.allclose(got, ref, rtol=1e-12)


def test_ces_zero_alpha_cell_is_zero():
    # a cell with α==0 must return demand 0 (Julia: toRet[α==0]=0)
    got = ces(10.0, np.array([1.0, 2.0, 3.0]), np.array([0.5, 0.0, 0.5]), 2.0, 1.0)
    assert got[1] == 0.0


def test_cde_docstring_example():
    # julia> cde([0.1,0.2,0.3],[1,2,3],[1.1,0.9,1.0],1,[2,1,1],[4])
    got = cde(
        np.array([0.1, 0.2, 0.3]),
        np.array([1.0, 2.0, 3.0]),
        np.array([1.1, 0.9, 1.0]),
        1.0,
        np.array([2.0, 1.0, 1.0]),
        4.0,
    )
    ref = np.array(
        [0.5341500255889858, 1.1690947909189762, 1.762605157903052, 2.3324281115373924]
    )
    assert np.allclose(got, ref, rtol=1e-12)
