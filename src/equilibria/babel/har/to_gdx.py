"""GTAP HAR → three-GDX wrapper for ``tariff_sim.gms``.

Reads ``basedata*.har`` / ``sets*.har`` / ``default*.prm`` from a directory
and writes the trio expected by ``tariff_sim.gms`` via ``getData.gms``:

* ``<base>Sets.gdx`` — ACTS, COMM, MARG, REG, ENDW, ENDWF, ENDWM, ENDWS
* ``<base>Dat.gdx`` — SAM monetary flows (VDFB, VMFB, EVFB, MAKB, …)
* ``<base>Prm.gdx`` — elasticities (ESUBT, ESUBVA, ESUBD, ESUBM, …)

100% native: uses ``equilibria.babel.har`` for input and
``equilibria.babel.gdx.writer`` for output. No GAMS API, no harpy3.
"""

from __future__ import annotations

from pathlib import Path

from equilibria.babel.gdx.symbols import Parameter, Set
from equilibria.babel.gdx.writer import write_gdx
from equilibria.babel.har.to_equilibria import load_gtap_from_har


def _set(name: str, members: list[str], desc: str = "") -> Set:
    return Set(
        name=name,
        sym_type="set",
        dimensions=1,
        description=desc,
        elements=[[m] for m in members],
    )


def _param(
    name: str,
    dim: int,
    records: list[tuple[list[str], float]],
    desc: str = "",
) -> Parameter:
    return Parameter(
        name=name,
        sym_type="parameter",
        dimensions=dim,
        description=desc,
        records=records,
    )


def _records_3d(
    data: dict,
    order: tuple[int, int, int],
) -> list[tuple[list[str], float]]:
    """Reorder a 3D dict's keys per ``order`` and emit non-zero records."""
    out: list[tuple[list[str], float]] = []
    for key, value in data.items():
        if value == 0.0:
            continue
        reordered = [key[order[0]], key[order[1]], key[order[2]]]
        out.append((reordered, float(value)))
    return out


def _records_2d(
    data: dict,
    order: tuple[int, int],
) -> list[tuple[list[str], float]]:
    out: list[tuple[list[str], float]] = []
    for key, value in data.items():
        if value == 0.0:
            continue
        out.append(([key[order[0]], key[order[1]]], float(value)))
    return out


def _records_1d(data: dict) -> list[tuple[list[str], float]]:
    out: list[tuple[list[str], float]] = []
    for key, value in data.items():
        if value == 0.0:
            continue
        out.append(([key], float(value)))
    return out


def _records_4d(
    data: dict,
    order: tuple[int, int, int, int],
) -> list[tuple[list[str], float]]:
    out: list[tuple[list[str], float]] = []
    for key, value in data.items():
        if value == 0.0:
            continue
        out.append(
            (
                [key[order[0]], key[order[1]], key[order[2]], key[order[3]]],
                float(value),
            )
        )
    return out


def convert_har_to_gdx(
    har_dir: str | Path,
    output_dir: str | Path,
    *,
    base_name: str,
    suffix: str | None = None,
) -> dict[str, Path]:
    """Convert a directory of GTAP HAR/PRM files into the three GDX files
    consumed by ``tariff_sim.gms``.

    Args:
        har_dir: Directory with ``basedata*.har``, ``sets*.har``,
            ``default*.prm``.
        output_dir: Destination directory; created if missing.
        base_name: GAMS ``%BaseName%`` token (e.g. ``"9x10"`` produces
            ``9x10Sets.gdx`` / ``9x10Dat.gdx`` / ``9x10Prm.gdx``).
        suffix: Aggregation tag passed to ``load_gtap_from_har`` (e.g. ``"-9x10"``).

    Returns:
        Mapping from logical name (``"sets"``, ``"dat"``, ``"prm"``) to the
        Path of the written GDX file.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    p = load_gtap_from_har(har_dir, suffix=suffix)

    sets_path = out_dir / f"{base_name}Sets.gdx"
    dat_path = out_dir / f"{base_name}Dat.gdx"
    prm_path = out_dir / f"{base_name}Prm.gdx"

    write_gdx(sets_path, _build_sets_symbols(p))
    write_gdx(dat_path, _build_dat_symbols(p))
    write_gdx(prm_path, _build_prm_symbols(p))

    return {"sets": sets_path, "dat": dat_path, "prm": prm_path}


# --------------------------------------------------------------------------
# Symbol builders
# --------------------------------------------------------------------------


def _build_sets_symbols(p) -> list:
    s = p.sets
    margins = sorted(getattr(s, "marg", []) or [])
    if not margins and getattr(s, "i", None):
        # Fall back to the conventional GTAP margin flag — leave empty if
        # neither attribute nor convention is available.
        margins = []
    endwf = sorted(getattr(s, "ff", []) or [])
    endwm = sorted(getattr(s, "mf", []) or [])
    endws = sorted(getattr(s, "sf", []) or [])
    return [
        _set("ACTS", sorted(s.a), "activities"),
        _set("COMM", sorted(s.i), "commodities"),
        _set("MARG", margins, "margin commodities"),
        _set("REG", sorted(s.r), "regions"),
        _set("ENDW", sorted(s.f), "endowments"),
        _set("ENDWF", endwf, "fixed endowments"),
        _set("ENDWM", endwm, "mobile endowments"),
        _set("ENDWS", endws, "sluggish endowments"),
    ]


def _build_dat_symbols(p) -> list:
    """Build the ``Dat.gdx`` symbols.

    Reorderings translate Python (r, …) keys to the GAMS ``COMM*ACTS*REG``
    style declared in ``getData.gms`` / Dat.gdx headers:

    * VDFB, VDFP, VMFB, VMFP, MAKB, MAKS: ``(i, a, r)`` from ``(r, i, a)``
    * EVFB, EVFP, EVOS:                   ``(f, a, r)`` from ``(r, f, a)``
    * VXSB, VFOB, VCIF, VMSB:             ``(i, r, rp)`` from ``(r, i, rp)``
    * VTWR:                               ``(m, i, r, rp)`` from ``(r, i, rp, m)``
    * VDPB/VDPP/VMPB/VMPP/VDGB/…:         ``(i, r)`` from ``(r, i)``
    * VST:                                ``(m, r)`` from ``(r, m)``
    """
    b = p.benchmark
    return [
        # firm flows (i, a, r)
        _param("VDFB", 3, _records_3d(b.vdfb, (1, 2, 0))),
        _param("VDFP", 3, _records_3d(b.vdfp, (1, 2, 0))),
        _param("VMFB", 3, _records_3d(b.vmfb, (1, 2, 0))),
        _param("VMFP", 3, _records_3d(b.vmfp, (1, 2, 0))),
        # household (i, r)
        _param("VDPB", 2, _records_2d(b.vdpb, (1, 0))),
        _param("VDPP", 2, _records_2d(b.vdpp, (1, 0))),
        _param("VMPB", 2, _records_2d(getattr(b, "vmpb", b.vmpp), (1, 0))),
        _param("VMPP", 2, _records_2d(b.vmpp, (1, 0))),
        # government (i, r)
        _param("VDGB", 2, _records_2d(b.vdgb, (1, 0))),
        _param("VDGP", 2, _records_2d(b.vdgp, (1, 0))),
        _param("VMGB", 2, _records_2d(getattr(b, "vmgb", b.vmgp), (1, 0))),
        _param("VMGP", 2, _records_2d(b.vmgp, (1, 0))),
        # investment (i, r)
        _param("VDIB", 2, _records_2d(b.vdib, (1, 0))),
        _param("VDIP", 2, _records_2d(b.vdip, (1, 0))),
        _param("VMIB", 2, _records_2d(getattr(b, "vmib", b.vmip), (1, 0))),
        _param("VMIP", 2, _records_2d(b.vmip, (1, 0))),
        # primary factors (f, a, r)
        _param("EVFB", 3, _records_3d(b.evfb, (1, 2, 0))),
        _param("EVFP", 3, _records_3d(getattr(b, "evfp", b.evfb), (1, 2, 0))),
        _param("EVOS", 3, _records_3d(b.evos, (1, 2, 0))),
        # trade (i, r, rp)
        _param("VXSB", 3, _records_3d(b.vxsb, (1, 0, 2))),
        _param("VFOB", 3, _records_3d(b.vfob, (1, 0, 2))),
        _param("VCIF", 3, _records_3d(b.vcif, (1, 0, 2))),
        _param("VMSB", 3, _records_3d(b.vmsb, (1, 0, 2))),
        # margins
        _param("VST", 2, _records_2d(b.vst, (1, 0))),  # (m, r) from (r, m)
        _param("VTWR", 4, _records_4d(b.vtwr, (3, 1, 0, 2))),  # (m, i, r, rp)
        # macro
        _param("SAVE", 1, _records_1d(b.save)),
        _param("VDEP", 1, _records_1d(b.vdep)),
        _param("VKB", 1, _records_1d(b.vkb)),
        _param("POP", 1, _records_1d(b.pop)),
        # make matrix (i, a, r)
        _param("MAKS", 3, _records_3d(b.maks, (1, 2, 0))),
        _param("MAKB", 3, _records_3d(b.makb, (1, 2, 0))),
    ]


def _build_prm_symbols(p) -> list:
    """Build the ``Prm.gdx`` elasticity symbols.

    GAMS ``getData.gms`` ``$loadDC`` order — names mapped from PRM file:

    * ESUBT, ESUBC, ESUBVA: per ``(a, r)`` from Python ``(r, a)``
    * ETRAQ, ESUBQ:         per ``(a, r)`` / ``(i, r)``
    * INCPAR, SUBPAR:       per ``(i, r)`` from ``(r, i)``
    * ESUBG, ESUBI:         per ``r``
    * ESUBD, ESUBM:         per ``(i, r)``
    * ESUBS:                per ``m`` (margin)
    * ETRAE:                per ``(f, r)`` — Python stores per ``f``; replicate
    * RORFLEX:              per ``r``
    """
    e = p.elasticities
    s = p.sets
    margins = sorted(getattr(s, "marg", []) or [])
    return [
        _param("ESUBT", 2, _records_2d(e.esubt, (1, 0))),
        _param("ESUBC", 2, _records_2d(e.esubc, (1, 0))),
        _param("ESUBVA", 2, _records_2d(e.esubva, (1, 0))),
        _param("ETRAQ", 2, _records_2d(e.etraq, (1, 0))),
        _param("ESUBQ", 2, _records_2d(e.esubq, (1, 0))),
        _param("INCPAR", 2, _records_2d(e.incpar, (1, 0))),
        _param("SUBPAR", 2, _records_2d(e.subpar, (1, 0))),
        _param("ESUBG", 1, _records_1d(e.esubg)),
        _param("ESUBI", 1, _records_1d(e.esubi)),
        _param("ESUBD", 2, _records_2d(e.esubd, (1, 0))),
        _param("ESUBM", 2, _records_2d(e.esubm, (1, 0))),
        _param(
            "ESUBS",
            1,
            [([m], _esubs_default(e, m)) for m in margins],
        ),
        _param(
            "ETRAE",
            2,
            [
                ([f, r], float(v))
                for f, v in (e.etrae or {}).items()
                for r in sorted(s.r)
                if v != 0.0
            ],
        ),
        _param("RORFLEX", 1, _records_1d(e.rorflex)),
    ]


def _esubs_default(elasticities, margin: str) -> float:
    """Margin elasticity ESUBS — fall back to 1.5 (GTAP default) when absent."""
    val = getattr(elasticities, "esubs", None)
    if isinstance(val, dict):
        return float(val.get(margin, 1.5))
    if isinstance(val, (int, float)):
        return float(val)
    return 1.5
