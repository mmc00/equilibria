"""CESTree, a nested CES block whose tree is read from a share table (ADR-0013).

The equations are MANAGE-WB's ``e_xfac``, ``e_pxfac`` and ``e_pxfacInd``
(``data/manage/gams/inc/model.inc:1253, 1307, 1361``), on toy trees. The toy
calibration below is written from the identities, not from the block: a parent's
value is the sum of its children's values (``x_p·p_p = Σ x_c·p_c``) and the share
is what makes the child demand hold,
``alpha = x_c/x_p·(λ_c·λ_top)^(1-σ)·(p_c/p_p)^σ``; an additive (volume
preserving) parent keeps ``x_p = Σ x_c`` instead, with the exhaustion price
``P``: ``alpha = x_c/x_p·λ_c^σ·(p_c/P)^σ``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from equilibria.backends.pyomo_backend import PyomoBackend
from equilibria.blocks.production import CESTree
from equilibria.core.sets import Set as ESet
from equilibria.model import Model
from equilibria.templates.manage.residuals import block_residuals

OWNERS = ("o1", "o2")
FAMILIES = ("e_xfac", "e_pxfac", "e_pxfacInd")


def _tree_block(
    nodes: tuple[str, ...],
    parents: tuple[str, ...],
    alpha: dict[tuple[str, str, str], float],
    sigma: dict[tuple[str, str], float],
    *,
    additive: tuple[str, ...] = (),
    top: tuple[str, ...] = (),
    start: dict[str, Any] | None = None,
) -> CESTree:
    nn, npar, no = len(nodes), len(parents), len(OWNERS)
    a = np.zeros((nn, npar, no))
    for (c, p, o), v in alpha.items():
        a[nodes.index(c), parents.index(p), OWNERS.index(o)] = v
    s = np.zeros((npar, no))
    for (p, o), v in sigma.items():
        s[parents.index(p), OWNERS.index(o)] = v
    values = {
        "alpha": a,
        "sigma": s,
        "additive": np.array([p in additive for p in parents], dtype=float),
        "top": np.array([p in top for p in parents], dtype=float),
    }
    return CESTree(
        values=values,
        start=start or {},
        quantity="xfac",
        price="pxfac",
        index_price="pxfacInd",
        shifter="lambdaxfac",
        node_set="N",
        parent_set="P",
        owner_set="O",
        alpha="alpha",
        sigma="sigma",
        additive="additive",
        top="top",
    )


def _build(block: CESTree, nodes: tuple[str, ...], parents: tuple[str, ...]) -> Any:
    model = Model(name="toy_tree")
    model.add_sets(
        [
            ESet(name="N", elements=nodes),
            ESet(name="P", elements=parents),
            ESet(name="O", elements=OWNERS),
        ]
    )
    model.add_block(block)
    backend = PyomoBackend()
    backend.build(model)
    return backend.pyomo_model


def _put(pm: Any, name: str, cells: dict[tuple[str, str], float]) -> None:
    var = getattr(pm, name)
    for key, v in cells.items():
        var[key].value = v


def _families(pm: Any) -> list[str]:
    return [f for f in FAMILIES if pm.find_component(f"{f}_con") is not None]


def _worst(pm: Any) -> float:
    res = block_residuals(pm, _families(pm))
    return max(abs(r) for rows in res.values() for r in rows.values())


# --- a two-level tree: y (top) -> va, nd; va -> k, l -----------------------

NODES = ("y", "va", "nd", "k", "l")
PARENTS = ("y", "va")
CHILDREN = {"y": ("va", "nd"), "va": ("k", "l")}


def _calibrated_two_level(sigma: float) -> tuple[Any, dict[str, dict]]:
    """The two-level tree at a point where every row holds, by construction."""
    x = {("k", "o1"): 3.0, ("l", "o1"): 5.0, ("nd", "o1"): 7.0}
    x |= {("k", "o2"): 1.5, ("l", "o2"): 0.4, ("nd", "o2"): 2.5}
    p = {("k", "o1"): 1.3, ("l", "o1"): 0.8, ("nd", "o1"): 1.1}
    p |= {("k", "o2"): 0.7, ("l", "o2"): 1.6, ("nd", "o2"): 0.9}
    p |= {("va", "o1"): 1.2, ("y", "o1"): 0.95, ("va", "o2"): 0.85, ("y", "o2"): 1.4}
    lam = {(n, o): 1.0 + 0.03 * i + 0.05 * j for i, n in enumerate(NODES)
           for j, o in enumerate(OWNERS)}  # fmt: skip
    alpha: dict[tuple[str, str, str], float] = {}
    for o in OWNERS:
        for par in ("va", "y"):  # bottom up
            kids = CHILDREN[par]
            x[(par, o)] = sum(x[(c, o)] * p[(c, o)] for c in kids) / p[(par, o)]
            for c in kids:
                lam_c = lam[(c, o)] * (lam[(par, o)] if par == "y" else 1.0)
                alpha[(c, par, o)] = (
                    x[(c, o)] / x[(par, o)] * lam_c ** (1 - sigma)
                    * (p[(c, o)] / p[(par, o)]) ** sigma
                )  # fmt: skip
    sig = {(par, o): sigma for par in PARENTS for o in OWNERS}
    block = _tree_block(NODES, PARENTS, alpha, sig, top=("y",))
    pm = _build(block, NODES, PARENTS)
    point = {"xfac": x, "pxfac": p, "lambdaxfac": lam}
    for name, cells in point.items():
        _put(pm, name, cells)
    return pm, point


def test_ces_tree_two_level_cd_limit() -> None:
    pm, point = _calibrated_two_level(sigma=1.01)
    res = block_residuals(pm, ["e_xfac", "e_pxfac"])
    # one demand row per (child, owner), one price row per (parent, owner)
    assert set(res["e_xfac"]) == {
        (c, o) for c in ("va", "nd", "k", "l") for o in OWNERS
    }
    assert set(res["e_pxfac"]) == {(p, o) for p in PARENTS for o in OWNERS}
    assert pm.find_component("e_pxfacInd_con") is None  # no additive parent
    assert _worst(pm) < 1e-12

    # doubling a parent's quantity doubles its children's demands
    x = point["xfac"]
    pm.xfac["va", "o1"].value = 2 * x[("va", "o1")]
    res = block_residuals(pm, ["e_xfac"])["e_xfac"]
    for c in ("k", "l"):
        assert res[(c, "o1")] == pytest.approx(-x[(c, "o1")], rel=1e-12)
    for c in ("k", "l"):
        pm.xfac[c, "o1"].value = 2 * x[(c, "o1")]
    res = block_residuals(pm, ["e_xfac"])["e_xfac"]
    assert max(abs(res[(c, "o1")]) for c in ("k", "l")) < 1e-12


def test_ces_tree_leontief_parent() -> None:
    """sigma = 0: xfac = alpha·xfac(parent)/lambda and pxfac = Σ alpha·pxfac/lambda
    (the Leontief branch, ``model.inc:1290, 1342-1355``)."""
    nodes, parents = ("v", "k", "l"), ("v",)
    alpha = {("k", "v", o): 0.3 for o in OWNERS} | {("l", "v", o): 0.6 for o in OWNERS}
    block = _tree_block(nodes, parents, alpha, {("v", o): 0.0 for o in OWNERS})
    pm = _build(block, nodes, parents)
    x = {("v", "o1"): 4.0, ("k", "o1"): 1.1, ("l", "o1"): 2.9}
    p = {("v", "o1"): 1.7, ("k", "o1"): 1.2, ("l", "o1"): 0.6}
    lam = {("v", "o1"): 1.4, ("k", "o1"): 1.25, ("l", "o1"): 0.8}
    for name, cells in (("xfac", x), ("pxfac", p), ("lambdaxfac", lam)):
        _put(pm, name, cells)
    res = block_residuals(pm, ["e_xfac", "e_pxfac"])
    for c in ("k", "l"):
        want = x[(c, "o1")] - alpha[(c, "v", "o1")] * x[("v", "o1")] / lam[(c, "o1")]
        assert res["e_xfac"][(c, "o1")] == pytest.approx(want, rel=1e-12)
    unit_cost = sum(
        alpha[(c, "v", "o1")] * p[(c, "o1")] / lam[(c, "o1")] for c in ("k", "l")
    )
    assert res["e_pxfac"][("v", "o1")] == pytest.approx(
        p[("v", "o1")] - unit_cost, rel=1e-12
    )


def test_ces_tree_multiple_parents() -> None:
    """A child under two parents has one row: the sum of both demands
    (``model.inc:1256``, ``sum(fpPar $ p_alphaXfac, ...)``)."""
    nodes, parents = ("p1", "p2", "n", "m"), ("p1", "p2")
    alpha = {("n", "p1", "o1"): 0.4, ("m", "p1", "o1"): 0.5, ("n", "p2", "o1"): 0.7}
    sigma = {("p1", "o1"): 0.5, ("p2", "o1"): 2.0}
    pm = _build(_tree_block(nodes, parents, alpha, sigma), nodes, parents)
    x = {("p1", "o1"): 3.0, ("p2", "o1"): 2.0, ("n", "o1"): 1.0, ("m", "o1"): 1.0}
    p = {("p1", "o1"): 1.1, ("p2", "o1"): 0.9, ("n", "o1"): 1.3, ("m", "o1"): 1.0}
    lam = dict.fromkeys(x, 1.0) | {("n", "o1"): 1.2}
    for name, cells in (("xfac", x), ("pxfac", p), ("lambdaxfac", lam)):
        _put(pm, name, cells)
    rows = block_residuals(pm, ["e_xfac"])["e_xfac"]
    assert set(rows) == {("n", "o1"), ("m", "o1")}  # one row for n, none in o2

    def demand(par: str) -> float:
        s = sigma[(par, "o1")]
        return (
            alpha[("n", par, "o1")] * x[(par, "o1")] * 1.2 ** (s - 1)
            * (p[(par, "o1")] / p[("n", "o1")]) ** s
        )  # fmt: skip

    assert rows[("n", "o1")] == pytest.approx(
        1.0 - demand("p1") - demand("p2"), rel=1e-12
    )


def test_ces_tree_additive() -> None:
    """An additive (volume preserving) parent has an exhaustion-price row
    ``e_pxfacInd``, its price rows use ``pxfacInd``, and the shifter enters as
    ``lambda^(-sigma)`` (``model.inc:1280-1282, 1307-1375``)."""
    nodes, parents = ("e", "c1", "c2", "v", "k"), ("e", "v")
    sigma, big_p = 0.7, 1.1  # not 0.5, where -σ = σ - 1
    x = {("c1", "o1"): 2.0, ("c2", "o1"): 3.0}
    p = {("c1", "o1"): 1.4, ("c2", "o1"): 0.7}
    lam = {("c1", "o1"): 1.3, ("c2", "o1"): 0.9, ("e", "o1"): 1.0}
    x[("e", "o1")] = x[("c1", "o1")] + x[("c2", "o1")]
    p[("e", "o1")] = (
        sum(x[(c, "o1")] * p[(c, "o1")] for c in ("c1", "c2")) / x[("e", "o1")]
    )
    alpha = {
        (c, "e", "o1"): x[(c, "o1")] / x[("e", "o1")] * lam[(c, "o1")] ** sigma
        * (p[(c, "o1")] / big_p) ** sigma
        for c in ("c1", "c2")
    }  # fmt: skip
    alpha[("k", "v", "o1")] = 1.0  # a standard parent beside it
    block = _tree_block(
        nodes, parents, alpha, {("e", "o1"): sigma, ("v", "o1"): 0.5}, additive=("e",)
    )
    pm = _build(block, nodes, parents)
    for name, cells in (("xfac", x), ("pxfac", p), ("lambdaxfac", lam)):
        _put(pm, name, cells)
    pm.pxfacInd["e", "o1"].value = big_p
    res = block_residuals(pm, ["e_xfac", "e_pxfac", "e_pxfacInd"])
    assert set(res["e_pxfacInd"]) == {("e", "o1")}  # the additive parent only
    for fam in ("e_pxfac", "e_pxfacInd"):
        assert abs(res[fam][("e", "o1")]) < 1e-12, fam
    for c in ("c1", "c2"):
        assert abs(res["e_xfac"][(c, "o1")]) < 1e-12
    # the exponent of lambda is -sigma: move lambda and the row moves by it
    pm.lambdaxfac["c1", "o1"].value = 2.0
    got = block_residuals(pm, ["e_xfac"])["e_xfac"][("c1", "o1")]
    want = (
        x[("c1", "o1")]
        - alpha[("c1", "e", "o1")]
        * x[("e", "o1")]
        * 2.0 ** (-sigma)
        * (big_p / p[("c1", "o1")]) ** sigma
    )
    assert got == pytest.approx(want, rel=1e-12)


def test_ces_tree_leaves_from_another_family_are_not_written_yet() -> None:
    with pytest.raises(NotImplementedError, match="leaf_quantity"):
        CESTree(
            quantity="q", price="p", node_set="N", owner_set="O", alpha="a",
            sigma="s", additive="ad", top="t", leaf_quantity="xa",
        )  # fmt: skip
