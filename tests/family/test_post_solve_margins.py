"""``mp_margin_params`` is computed once per benchmark and gives what it gave before.

``_reference`` is the body of ``post_solve.mp_margin_params`` on main (1bfcb8ee),
before MAR-246 rewrote it: one ``get`` per (r, i, rp, m) and 26 rebuilds per 20x41 run.
"""

from __future__ import annotations

from types import SimpleNamespace

from tests.family._data import load_params


def _reference(params, sets) -> dict:
    bm = params.benchmark
    tmarg: dict = {}
    amgm: dict = {}
    for r in sets.r:
        for i in sets.i:
            for rp in sets.rp if hasattr(sets, "rp") else sets.r:
                vtwr_total = (
                    sum(float(bm.vtwr.get((r, i, rp, mg), 0.0) or 0.0) for mg in sets.m)
                    if hasattr(bm, "vtwr")
                    else 0.0
                )
                vcif = (
                    float(bm.vcif.get((r, i, rp), 0.0) or 0.0)
                    if hasattr(bm, "vcif")
                    else 0.0
                )
                vfob = (
                    float(bm.vfob.get((r, i, rp), 0.0) or 0.0)
                    if hasattr(bm, "vfob")
                    else 0.0
                )
                xw_bench = (
                    float(bm.vxsb.get((r, i, rp), 0.0) or 0.0)
                    if hasattr(bm, "vxsb")
                    else 0.0
                )
                if xw_bench <= 0.0 and hasattr(bm, "vxmd"):
                    xw_bench = float(bm.vxmd.get((r, i, rp), 0.0) or 0.0)
                tmarg[(r, i, rp)] = (
                    max(vcif - vfob, 0.0) / max(xw_bench, 1e-12)
                    if xw_bench > 0.0
                    else 0.0
                )
                if vtwr_total > 0.0:
                    for mg in sets.m:
                        flow = (
                            float(bm.vtwr.get((r, i, rp, mg), 0.0) or 0.0)
                            if hasattr(bm, "vtwr")
                            else 0.0
                        )
                        amgm[(mg, r, i, rp)] = flow / vtwr_total
    return {"tmarg": (3, tmarg), "amgm": (4, amgm)}


def _stub():
    """A benchmark with only ``vxmd``, ``vtwr`` and ``vcif`` (no ``vfob``, no
    ``vxsb``), a ``None`` cell, a margin row that sums to 0 and a zero trade base."""
    sets = SimpleNamespace(r=["A", "B"], i=["x"], m=["t", "w"])
    bm = SimpleNamespace(
        vxmd={("A", "x", "B"): 4.0, ("B", "x", "A"): 0.0},
        vcif={("A", "x", "B"): 5.0, ("B", "x", "A"): 1.0, ("A", "x", "A"): None},
        vtwr={
            ("A", "x", "B", "t"): 1.0,
            ("A", "x", "B", "w"): 3.0,
            ("B", "x", "A", "t"): 0.0,
            ("B", "x", "A", "w"): None,
        },
    )
    return SimpleNamespace(benchmark=bm), sets


def test_same_as_reference_on_nus333() -> None:
    from equilibria.family.post_solve import mp_margin_params

    p = load_params("nus333")
    assert mp_margin_params(p, p.sets) == _reference(p, p.sets)


def test_same_as_reference_on_a_partial_benchmark() -> None:
    from equilibria.family.post_solve import mp_margin_params

    p, sets = _stub()
    got = mp_margin_params(p, sets)
    assert got == _reference(p, sets)
    assert ("t", "B", "x", "A") not in got["amgm"][1]  # vtwr_total == 0: no key


def test_second_call_is_cached() -> None:
    from equilibria.family.post_solve import mp_margin_params

    p = load_params("nus333")
    assert mp_margin_params(p, p.sets) is mp_margin_params(p, p.sets)
