"""Gates of the ALTERTAX outputs against a GEMPACK run folder: the ONE implementation.

Every ``test_altertax_*`` test and ``scripts/family/run_altertax.py --compare`` call the
functions here. One section per output (updated, summary, volume, decomposition, % changes,
levels); each returns plain data (gaps, failing cells) and ``report_*`` turns it into the
record the command writes to ``compare.json``. Tolerances come from
:mod:`equilibria.family.altertax.tolerances`.

Waivers. The known-miss tables (``PCT_KNOWN_MISS``, ``DECOMP_KNOWN_MISS``,
``NOISE_UNMATCHED``) were measured on ONE experiment: nus333 with Burfisher's ALTAX shock.
Their keys are region/commodity names that 3x3, 3x4, 5x5 or 20x41 share, so they must never
apply by key alone. ``waivers_for(dataset_dir, sl4)`` returns them only when the input data
is the nus333 ``basedata.har`` (SHA-256) AND the GEMPACK solution's shock is
``to(AGR,AGR,USA) = 0.0012``; otherwise it returns ``NO_WAIVERS`` and every miss fails the
gate. ``compare.json`` records which was used (``waivers``).
"""

from __future__ import annotations

import gzip
import hashlib
import shutil
import tempfile
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from equilibria.babel.har.reader import read_har
from equilibria.babel.har.symbols import HeaderArray
from equilibria.compare.measure import rel_diff
from equilibria.family.altertax import compare as _levels
from equilibria.family.altertax.decomp import DECOMP_HEADERS, decomp_headers
from equilibria.family.altertax.pct import gempack_changes, pct_changes, zero_base_cells
from equilibria.family.altertax.run import AltertaxRun
from equilibria.family.altertax.summary import SUMMARY_HEADERS, summary_headers
from equilibria.family.altertax.tolerances import (
    ABS_FLOOR,
    TOL_CHANGE,
    TOL_LEVELS,
    TOL_PCT,
    TOL_SUMMARY,
    TOL_UPDATED,
)
from equilibria.family.altertax.updated import UPDATED_HEADERS, updated_headers
from equilibria.family.altertax.volume import (
    VOLUME_HEADERS,
    volume_bases,
    volume_headers,
)

# --- GEMPACK run-folder names: GEMPACK's own first, then the burfisher fixture names ----
UPDATED_FILES = _levels.UPDATED_NAMES  # same file for the levels and the updated gates
SUMMARY_FILES = ("summary.har", "altax_summary.har")
DECOMP_FILES = ("decomp.har", "altax_decomp.har")
VOLUME_FILES = ("volume.har", "altax_volume.har")
SL4_DUMP = "sl4dump.har"  # RunGTAP folders: % changes only, no levels, no .sl4
SL4_FILES = _levels.SL4_NAMES + (SL4_DUMP,)
LEVELS_FILES = _levels.LEVELS_NAMES + _levels.SL4_NAMES  # sl4levels.har, else the .sl4

THRESHOLD_ULPS = (
    8  # volume: cells above 8 float32 ulp of the level are compared relatively
)
NOISE = 1e-6  # decomposition: relative to max|EV| of the region (user 2026-11-09)
RATE_COLS = ("taxrateb", "taxrateu")  # decomp columns that are tax rates (levels)
TEXT_HEADERS = ("XXHS",)  # updated.har headers that are text
SUMMARY_LEVELS = ("GDPE", "COST", "GDPI", "SALE")
KNOWN_MISS_FACTOR = 1.5  # a known miss may grow to 1.5x its measured error


def ulp32(x: Any) -> np.ndarray:
    return np.spacing(
        np.abs(np.asarray(x, dtype=np.float64)).astype(np.float32)
    ).astype(np.float64)


def find_files(folder: Path, groups: dict[str, tuple[str, ...]]) -> dict[str, Path]:
    """The first existing name of each group in ``folder``; a clear error naming every
    missing one."""
    folder = Path(folder)
    found: dict[str, Path] = {}
    missing = []
    for key, names in groups.items():
        for n in names:
            if (folder / n).exists():
                found[key] = folder / n
                break
        else:
            missing.append(f"{key}: {' | '.join(names)}")
    if missing:
        raise FileNotFoundError(f"GEMPACK folder {folder} lacks: " + "; ".join(missing))
    return found


def read_sl4(path: Path):
    """``read_sl4_changes`` of a ``.sl4`` or ``.sl4.gz``; a ``.har`` is a ``sl4dump.har``
    (RunGTAP folders), read by :func:`equilibria.family.altertax.pct.read_sl4dump`."""
    from equilibria.babel.har.sl4 import read_sl4_changes

    path = Path(path)
    if path.suffix == ".har":
        from equilibria.family.altertax.pct import read_sl4dump

        return read_sl4dump(path)
    if path.suffix != ".gz":
        return read_sl4_changes(path)
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / path.stem
        with gzip.open(path, "rb") as src, out.open("wb") as dst:
            shutil.copyfileobj(src, dst)
        return read_sl4_changes(out)


# --- shared helpers -----------------------------------------------------------------


def worst_gap(a: Any, b: Any) -> tuple[float, tuple]:
    """Max of ``rel_diff(a, b, ref=b, ABS_FLOOR)`` over the cells, and its index."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    d = np.vectorize(lambda x, y: rel_diff(x, y, y, ABS_FLOOR))(a, b)
    i = np.unravel_index(int(np.nanargmax(d)), d.shape) if d.size else ()
    return float(d[i]) if d.size else 0.0, i


def gaps_by_header(
    ours: dict[str, HeaderArray], gem: dict[str, HeaderArray], names: Iterable[str]
) -> dict[str, tuple[float, tuple]]:
    return {n: worst_gap(ours[n].array, gem[n].array) for n in names}


def worst_cells(
    ours: dict[str, HeaderArray],
    gem: dict[str, HeaderArray],
    names: Iterable[str],
    n: int = 10,
) -> list[dict]:
    """The ``n`` cells with the largest relative gap over ``names``."""
    rows = []
    for name in names:
        a = np.asarray(ours[name].array, dtype=float)
        b = np.asarray(gem[name].array, dtype=float)
        for ix in np.ndindex(a.shape):
            rows.append(
                (rel_diff(a[ix], b[ix], b[ix], ABS_FLOOR), name, ix, a[ix], b[ix])
            )
    rows.sort(key=lambda r: (r[0] != r[0], r[0]), reverse=True)
    return [_cell(h, ix, m, g, d) for d, h, ix, m, g in rows[:n]]


def _cell(header: str, ix: Any, ours: float, gem: float, gap: float) -> dict:
    return {
        "header": header,
        "cell": [int(i) for i in ix],
        "ours": float(ours),
        "gempack": float(gem),
        "gap": float(gap),
    }


def structure_problems(
    ours: dict[str, HeaderArray],
    gem: dict[str, HeaderArray],
    names: Iterable[str],
    *,
    labels: bool = True,
) -> list[str]:
    """Headers whose sets, shape, coefficient or long name differ from GEMPACK's.
    ``labels=False`` leaves the long name out: RunGTAP's ``gtapv7.tab`` words the long
    names of the data headers differently from Burfisher's (``VDFB``: "expenditure on
    domestic c by act. a in r at basic prices" against "domestic purchases, by firms, at
    basic prices"); sets, shape and coefficient name are still compared. A text header
    (``TEXT_HEADERS``, the ``XXHS`` history) is compared on coefficient name, long name and
    sets only: its lines and their count are the run's own history, not data."""
    bad = []
    for name in names:
        a, b = ours[name], gem[name]
        text = name in TEXT_HEADERS
        if (
            a.set_names != b.set_names
            or a.set_elements != b.set_elements
            or (not text and a.array.shape != b.array.shape)
            or a.coeff_name != b.coeff_name
            or (labels and a.long_name != b.long_name)
        ):
            bad.append(name)
    return bad


def miss_is_bounded(err: float, ref: float) -> bool:
    """A known miss keeps its sign and stays within 1.5x of its measured error."""
    return err * ref > 0 and abs(err) <= KNOWN_MISS_FACTOR * abs(ref)


def _record(output, gate, max_, kind, worst, passed, **extra) -> dict:
    return {
        "output": output,
        "gate": gate,
        "max": max_,
        "max_kind": kind,
        "worst": worst,
        "pass": bool(passed),
        **extra,
    }


# --- updated.har ----------------------------------------------------------------------


def updated_failures(ours, gem) -> dict[str, tuple[float, tuple]]:
    """Numeric headers of ``updated.har`` above TOL_UPDATED."""
    names = [n for n in UPDATED_HEADERS if n not in TEXT_HEADERS]
    return {
        n: g
        for n, g in gaps_by_header(ours, gem, names).items()
        if not g[0] <= TOL_UPDATED
    }


def report_updated(ours, gem, labels: bool = True) -> dict:
    names = [n for n in UPDATED_HEADERS if n not in TEXT_HEADERS]
    gaps = gaps_by_header(ours, gem, names)
    bad_struct = structure_problems(ours, gem, UPDATED_HEADERS, labels=labels)
    mx = max((g[0] for g in gaps.values()), default=0.0)
    ok = not updated_failures(ours, gem) and not bad_struct
    return _record(
        "updated",
        {"TOL_UPDATED": TOL_UPDATED},
        mx,
        "relative",
        worst_cells(ours, gem, names),
        ok,
        known_miss=[],
        structure_problems=bad_struct,
    )


# --- waivers: known misses of the nus333 ALTAX reference ---------------------------------

# SHA-256 of ``data/gtap7/nus333/basedata.har``: the identity of the input data.
NUS333_BASEDATA_SHA256 = (
    "13e784b73f1fbf8ecdabc803c0454290aa5fb097f74bb5c59f0659511f841543"
)
NUS333_ALTAX_SHOCK = {("AGR", "AGR", "USA"): 0.0012}  # ``to`` of altax.exp
REFERENCE_ID = "burfisher-nus333-altax"


@dataclass(frozen=True)
class Waivers:
    """The measured misses that may pass: ``summary`` headers (float32 noise), ``decomp``
    regional totals and ``pct`` cells with their measured relative error. ``reference`` is
    the id of the reference they were measured on, ``None`` for none."""

    reference: str | None = None
    summary: frozenset = frozenset()
    decomp: dict = field(default_factory=dict)
    pct: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "reference": self.reference,
            "applied": self.reference is not None,
            "summary_headers": sorted(self.summary),
            "decomp_totals": len(self.decomp),
            "pct_cells": len(self.pct),
        }


NO_WAIVERS = Waivers()


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def is_nus333_altax_reference(dataset_dir: Path, sl4: Any) -> bool:
    """True iff the input data is the nus333 ``basedata.har`` and the GEMPACK solution is
    the ALTAX experiment (``to`` is zero but ``AGR,AGR,USA`` = 0.0012)."""
    base = Path(dataset_dir) / "basedata.har"
    if not base.exists() or _sha256(base) != NUS333_BASEDATA_SHA256:
        return False
    to = sl4.changes.get("to")
    if to is None:
        return False
    cells = {
        tuple(to.set_elements[d][i] for d, i in enumerate(ix)): float(to.values[ix])
        for ix in np.ndindex(to.values.shape)
        if to.values[ix] != 0.0
    }
    return set(cells) == set(NUS333_ALTAX_SHOCK) and all(
        abs(cells[k] - v) <= 1e-9 for k, v in NUS333_ALTAX_SHOCK.items()
    )


def waivers_for(dataset_dir: Path, sl4: Any) -> Waivers:
    """``NUS333_ALTAX_WAIVERS`` for the nus333 ALTAX reference, else ``NO_WAIVERS``."""
    if is_nus333_altax_reference(dataset_dir, sl4):
        return NUS333_ALTAX_WAIVERS
    return NO_WAIVERS


# --- summary.har ----------------------------------------------------------------------

# Accounting checks that are float32 residuals of GEMPACK's data and sums, so they are
# zero or a few ulp of the flow F they check (ulp32(F) is cell dependent) and a relative
# metric against them is meaningless: the headers below are NOT matched at TOL_SUMMARY
# and the test below fails if one starts to be. Measured cause per header:
#  * VMCK/VMCP/VTCK/VTCP (pre): GEMPACK's order of float32 accumulation of VCB, MAKEBCOM
#    and VTMUSE is not reproducible (a sequential float32 sum of the same basedata gives 3
#    of 6 VMCK cells, by chance); in float64 the data residual is another small number.
#  * VMCQ/TRCQ/VTCQ/MPCQ (post): GEMPACK's residuals on the updated data against the
#    exact ~0 of the float64 model. TRCQ and MPCQ are the input dataset's own float32
#    imbalance carried through GEMPACK's update (TRCK cells 3, 6, 7 equal TRCQ; float64 on
#    altax_updated.har reproduces them), not GEMPACK computing noise; VMCQ and VTCQ are
#    accumulation-order noise as VMCK and VTCK.
NOISE_UNMATCHED = frozenset(
    ["VMCK", "VMCP", "VTCK", "VTCP", "VMCQ", "TRCQ", "VTCQ", "MPCQ"]
)


def summary_failures(ours, gem, names) -> dict[str, tuple[float, tuple]]:
    return {
        n: g
        for n, g in gaps_by_header(ours, gem, names).items()
        if not g[0] <= TOL_SUMMARY
    }


def report_summary(
    ours, gem, waivers: Waivers = NO_WAIVERS, labels: bool = True
) -> dict:
    checks = [n for n in SUMMARY_HEADERS if n not in SUMMARY_LEVELS]
    bad_levels = summary_failures(ours, gem, SUMMARY_LEVELS)
    bad_checks = summary_failures(ours, gem, checks)
    unexpected = sorted(set(bad_checks) - waivers.summary)
    bad_struct = structure_problems(ours, gem, SUMMARY_HEADERS, labels=labels)
    gaps = gaps_by_header(ours, gem, SUMMARY_LEVELS)
    mx = max(g[0] for g in gaps.values())
    return _record(
        "summary",
        {"TOL_SUMMARY": TOL_SUMMARY, "gated": list(SUMMARY_LEVELS)},
        mx,
        "relative",
        worst_cells(ours, gem, SUMMARY_LEVELS),
        not bad_levels and not unexpected and not bad_struct,
        known_miss=sorted(set(bad_checks) & waivers.summary),
        failing_unexpected=unexpected,
        structure_problems=bad_struct,
    )


# --- volume.har -----------------------------------------------------------------------


def change_scan(
    name: str,
    ours: Any,
    gem: Any,
    thr: Callable[[tuple], float],
    *,
    level_cols: Callable[[tuple], bool] | None = None,
) -> tuple[list[tuple], list[tuple]]:
    """Cells of a change table above their noise threshold ``thr(ix)`` on either side are
    compared relatively (same sign, ``TOL_CHANGE``); the rest must agree in absolute terms
    within the threshold itself. ``level_cols(ix)`` marks a cell that is a level (compared
    relatively, no threshold). Returns ``(above, abs_bad)``; ``above`` holds
    ``(name, ix, ours, gem, rel, ok)``."""
    ours = np.asarray(ours, dtype=np.float64)
    gem = np.asarray(gem, dtype=np.float64)
    above, abs_bad = [], []
    for ix in np.ndindex(ours.shape):
        m, g = float(ours[ix]), float(gem[ix])
        t = thr(ix)
        d = rel_diff(m, g, g, ABS_FLOOR)
        if level_cols is not None and level_cols(ix):
            above.append((name, ix, m, g, d, d <= TOL_CHANGE))
        elif max(abs(m), abs(g)) > t:
            above.append((name, ix, m, g, d, m * g > 0 and d <= TOL_CHANGE))
        elif abs(m - g) > t:
            abs_bad.append((name, ix, m, g, t))
    return above, abs_bad


def volume_compare(name, ours, gem, level):
    """(cells above threshold, relative failures, absolute failures below it)."""
    thr = THRESHOLD_ULPS * ulp32(level)
    above, abs_bad = change_scan(name, ours, gem, lambda ix: float(thr[ix]))
    rel_bad = [(n, ix, m, g, d) for n, ix, m, g, d, ok in above if not ok]
    return len(above), rel_bad, abs_bad


def report_volume(run: AltertaxRun, ours, gem, labels: bool = True) -> dict:
    bases = volume_bases(run)
    above, abs_bad = [], []
    for name in VOLUME_HEADERS:
        thr = THRESHOLD_ULPS * ulp32(bases[name])
        a, b = change_scan(
            name, ours[name].array, gem[name].array, lambda ix: float(thr[ix])
        )
        above += a
        abs_bad += b
    bad = [c for c in above if not c[5]]
    worst = sorted(above, key=lambda c: c[4], reverse=True)[:10]
    bad_struct = structure_problems(ours, gem, VOLUME_HEADERS, labels=labels)
    return _record(
        "volume",
        {"TOL_CHANGE": TOL_CHANGE, "threshold_ulps": THRESHOLD_ULPS},
        max((c[4] for c in above), default=0.0),
        "relative",
        [_cell(n, ix, m, g, d) | {"ok": ok} for n, ix, m, g, d, ok in worst],
        not bad and not abs_bad and not bad_struct,
        known_miss=[],
        cells_compared=len(above),
        failing=[_cell(n, ix, m, g, d) for n, ix, m, g, d, _ in bad],
        failing_absolute=len(abs_bad),
        structure_problems=bad_struct,
    )


# --- decomp.har -----------------------------------------------------------------------


def decomp_scale(gem) -> np.ndarray:
    """max|EV| of each region: the largest |component| of its WELFARE row."""
    return np.abs(np.asarray(gem["A"].array, dtype=np.float64)).max(axis=1)


def region_scale(ha, ix, scale):
    """Scale of cell ``ix``: the largest scale of the regions it is indexed by."""
    pos = [i for i, n in enumerate(ha.set_names) if n == "REG"]
    return max(scale[ix[i]] for i in pos) if pos else scale.max()


def decomp_compare(name, ours_ha, gem_ha, scale):
    """(cells above the noise threshold, relative failures, absolute failures)."""
    last = gem_ha.set_elements[-1] if gem_ha.set_names else []

    def is_rate(ix):
        return (
            bool(gem_ha.set_names)
            and gem_ha.set_names[-1] == "COL"
            and last[ix[-1]] in RATE_COLS
        )

    above, abs_bad = change_scan(
        name,
        ours_ha.array,
        gem_ha.array,
        lambda ix: NOISE * region_scale(gem_ha, ix, scale),
        level_cols=is_rate,
    )
    rel_bad = [(n, ix, m, g, d) for n, ix, m, g, d, ok in above if not ok]
    return len(above), rel_bad, abs_bad


def _regional_totals(name, ha):
    """``{(header, part, region): total}``: the sum over every dimension but the region
    of a header's welfare columns. ``COL`` headers give ``welcnt`` and ``dvol`` (the tax
    rates are levels); the trade tax headers attribute a cell to the region the CNT uses
    (``xtax``/export: the source, axis 1; ``mtax``/import: the destination, axis 2)."""
    arr = np.asarray(ha.array, dtype=np.float64)
    regs = ha.set_elements[ha.set_names.index("REG")]
    parts = []  # (label, array, region axis)
    if name == "A23":
        parts = [("xtax", arr[..., 0], 1), ("mtax", arr[..., 1], 2)]
    elif name == "A231":
        for c, lab in enumerate(("welcnt", "dvol")):
            parts += [(f"{lab}/export", arr[:, :, :, c, 1], 1)]
            parts += [(f"{lab}/import", arr[:, :, :, c, 0], 2)]
    elif ha.set_names[-1] == "COL":
        ax = ha.set_names.index("REG")
        parts = [(lab, arr[..., c], ax) for c, lab in enumerate(("welcnt", "dvol"))]
    else:
        parts = [("", arr, ha.set_names.index("REG"))]
    out = {}
    for lab, a, ax in parts:
        tot = a.sum(axis=tuple(i for i in range(a.ndim) if i != ax))
        for r, t in zip(regs, tot, strict=True):
            out[(name, lab, r)] = float(t)
    return out


def total_gaps(headers, gempack, scale):
    """Relative gap of every regional total above the noise floor (``NOISE * max|EV|``)."""
    floor = NOISE * scale.max()
    gaps = {}  # two-sided: a total above the floor on EITHER side is gated
    for name in DECOMP_HEADERS:
        if name.startswith("C"):
            continue  # exogenous at zero: no cell above the floor
        ours = _regional_totals(name, headers[name])
        gem = _regional_totals(name, gempack[name])
        for k, g in gem.items():
            m = ours[k]
            if max(abs(g), abs(m)) > floor:
                gaps[k] = (m, g, rel_diff(m, g, g, ABS_FLOOR) if m * g > 0 else np.inf)
    return gaps


# Regional totals that miss TOL_CHANGE, with the signed error (ours - GEMPACK) / |GEMPACK|
# measured at the time. EMPTY since the exact sigma = 1 Cobb-Douglas (b8fe408f, b3b161ac:
# ``unit_sigma="gempack"`` keeps the unit elasticities at exactly 1) and the refresh of the
# base Fisher aggregates after the settled seed (440d9533). Every miss this table used to
# carry came from the 1.01 stand-in (A21 USA, A222 USA, A24 ROW, A23 mtax/xtax and A231
# dvol/export, all linear in sigma - 1) or from the small-cell residual that the stand-in
# fed (A231 dvol/import USA, a near-cancelling sum of three small cells); at exact CD all of
# them pass. Measured on the nus333 ALTAX reference: A231 within 2.7e-5 of GEMPACK, EV within
# 2.3e-5 (USA) and 7.9e-7 (ROW), every regional total within TOL_CHANGE. A miss that returns
# is entered here with its signed error; ``test_known_misses_are_bounded`` then keeps it
# within 1.5x of that error (same sign) and fails when it starts to pass or gets worse.
# ``load_altertax_params(unit_sigma=1.01)`` still selects the old stand-in for A/B, and
# those misses come back with it.
DECOMP_KNOWN_MISS: dict = {}


def ev_gaps(ours, gem) -> list[float]:
    """Relative gap of the EV (sum of the WELFARE row) of each region."""
    ev = np.asarray(ours["A"].array, dtype=np.float64).sum(axis=1)
    g = np.asarray(gem["A"].array, dtype=np.float64).sum(axis=1)
    return [rel_diff(m, x, x, ABS_FLOOR) for m, x in zip(ev, g, strict=True)]


def report_decomp(
    ours, gem, waivers: Waivers = NO_WAIVERS, labels: bool = True
) -> dict:
    """Gate: EV of each region and each header's regional totals above the noise floor at
    TOL_CHANGE (controller ruling 2026-11-09); cell-level components are reported, not
    gated. Totals in ``waivers.decomp`` pass while they keep their sign and stay within
    1.5x of their measured error."""
    scale = decomp_scale(gem)
    ev = ev_gaps(ours, gem)
    gaps = total_gaps(ours, gem, scale)
    failing, known = [], []
    for k, (m, g, rel) in gaps.items():
        if rel <= TOL_CHANGE:
            continue
        err = (m - g) / abs(g)
        rec = {
            "total": list(k),
            "ours": m,
            "gempack": g,
            "gap": float(rel),
            "error": err,
        }
        if k in waivers.decomp and miss_is_bounded(err, waivers.decomp[k]):
            known.append(rec | {"measured": waivers.decomp[k]})
        else:
            failing.append(rec)
    cells = ok_cells = 0
    for name in DECOMP_HEADERS:
        n, r, _ = decomp_compare(name, ours[name], gem[name], scale)
        cells += n
        ok_cells += n - len(r)
    worst = sorted(
        [
            {"total": list(k), "ours": m, "gempack": g, "gap": float(r)}
            for k, (m, g, r) in gaps.items()
        ],
        key=lambda d: d["gap"],
        reverse=True,
    )[:10]
    bad_struct = structure_problems(ours, gem, DECOMP_HEADERS, labels=labels)
    ev_ok = max(ev) <= TOL_CHANGE
    return _record(
        "decomp",
        {"TOL_CHANGE": TOL_CHANGE, "noise": NOISE, "gated": "EV + regional totals"},
        max([*ev, *(r for *_, r in gaps.values() if np.isfinite(r))]),
        "relative",
        worst,
        ev_ok and not failing and not bad_struct,
        known_miss=known,
        failing=failing,
        ev_gap=[float(x) for x in ev],
        totals_compared=len(gaps),
        cells_within_tol=[ok_cells, cells],
        structure_problems=bad_struct,
    )


def report_decomp_of(
    run: AltertaxRun, gem, waivers: Waivers = NO_WAIVERS, labels: bool = True
) -> dict:
    """:func:`report_decomp` of the run's decomposition: the path integral when the run
    has one (``solve_altertax(decomp_steps=...)``, recorded as ``steps`` and ``path``),
    else the one-step ``decomp_headers`` (``steps`` 1)."""
    path = getattr(run, "decomp_path", None)
    if path is None:
        return report_decomp(decomp_headers(run), gem, waivers, labels) | {
            "steps": 1,
            "steps_half": None,
            "richardson": False,
            "path": None,
        }
    return report_decomp(path.headers, gem, waivers, labels) | {
        "steps": path.steps,
        "steps_half": path.steps_half,
        "richardson": path.richardson,
        "path": path.to_dict(),
    }


# --- pct (solution.sl4) ---------------------------------------------------------------


def pct_passes(m: float, g: float) -> bool:
    if abs(m - g) > max(TOL_CHANGE * abs(g), TOL_PCT):
        return False
    return not (abs(g) > 10 * TOL_PCT and m * g <= 0)


def pct_rel(m: float, g: float) -> float:
    return (m - g) / max(abs(g), TOL_PCT)


# Cells whose base flow is 0 (EVFB(LAND, MFG|SER, USA|ROW) = 0 in basedata.har: 8 cells, qes
# and qfe of LAND in MFG and SER of both regions) are excluded by rule. A % change on a zero
# flow is undefined; we report 0, GEMPACK reports the value of its price formula: E_qes2 holds
# for every cell regardless of the weight, and the 0.0013 comes from -ETRAE * (pes - pe) on a
# zero flow (qe itself is 0 in the .sl4). GEMPACK's own volume change DQFE = qfe / 100 * BEVFB
# zeroes it. Four of the 8 would pass anyway through the absolute floor (ROW).
ZERO_BASE = {
    ("qes", ("LAND", "MFG", "USA")),
    ("qes", ("LAND", "SER", "USA")),
    ("qfe", ("LAND", "MFG", "USA")),
    ("qfe", ("LAND", "SER", "USA")),
    ("qes", ("LAND", "MFG", "ROW")),
    ("qes", ("LAND", "SER", "ROW")),
    ("qfe", ("LAND", "MFG", "ROW")),
    ("qfe", ("LAND", "SER", "ROW")),
}

# Measured misses of the ALTERTAX shock; the value is the relative error
# (ours - gem) / max(|gem|, TOL_PCT). EMPTY since the exact sigma = 1 Cobb-Douglas
# (b8fe408f, b3b161ac: ``unit_sigma="gempack"``) and the refresh of the base Fisher
# aggregates after the settled seed (440d9533): every cell now passes. The misses it
# carried were pfactor (USA +4.52, ROW -4.52, ~3e-5 pp: the b / (sigma - 1) term of the
# 1.01 stand-in) and the small ROW -> USA AGR import cells qpm, qms and qxs (same family),
# all of which close at exact CD. pgdp / qgdp were misses until 2026-11-09 (Fisher GDP
# valued trade at the frozen pefob / pmcif; they now expand M_PEFOB / M_PMCIF, 81bca7e4).
# A miss that returns is entered here with its measured error. The old stand-in is still
# available for A/B with ``load_altertax_params(unit_sigma=1.01)``.
PCT_KNOWN_MISS: dict = {}


def pct_compared(ours, gempack, zero=ZERO_BASE):
    """``{(var, cell): (ours, gem)}`` of the mapped variables, zero-base cells excluded."""
    return {
        (n, k): (ours[n][k], g)
        for n, cells in gempack.items()
        if n in ours
        for k, g in cells.items()
        if (n, k) not in zero
    }


def pct_failures(ours, gempack, zero=ZERO_BASE):
    return {
        key: pct_rel(m, g)
        for key, (m, g) in pct_compared(ours, gempack, zero).items()
        if not pct_passes(m, g)
    }


def report_pct(run: AltertaxRun, ours, gem, waivers: Waivers = NO_WAIVERS) -> dict:
    """Gate per cell: ``|ours - gem| <= max(TOL_CHANGE * |gem|, TOL_PCT)`` and the same
    sign where ``|gem| > 10 * TOL_PCT``. Zero-base cells of this run are excluded."""
    zero = {(n, k) for n, cells in zero_base_cells(run).items() for k in cells}
    cells = pct_compared(ours, gem, zero)
    fails = pct_failures(ours, gem, zero)
    failing, known = [], []
    for key, err in fails.items():
        m, g = cells[key]
        rec = {
            "variable": key[0],
            "cell": list(key[1]),
            "ours": m,
            "gempack": g,
            "error": err,
        }
        if key in waivers.pct and miss_is_bounded(err, waivers.pct[key]):
            known.append(rec | {"measured": waivers.pct[key]})
        else:
            failing.append(rec)
    errs = sorted(
        ((abs(m - g), key, m, g) for key, (m, g) in cells.items()),
        reverse=True,
    )[:10]
    worst = [
        {"variable": k[0], "cell": list(k[1]), "ours": m, "gempack": g, "abs_gap": d}
        for d, k, m, g in errs
    ]
    return _record(
        "pct",
        {"TOL_CHANGE": TOL_CHANGE, "TOL_PCT": TOL_PCT},
        max((abs(m - g) for m, g in cells.values()), default=0.0),
        "absolute (percentage points)",
        worst,
        not failing and bool(cells),
        known_miss=known,
        failing=failing,
        cells_compared=len(cells),
        zero_base_excluded=sorted([n, list(k)] for n, k in zero),
    )


NUS333_ALTAX_WAIVERS = Waivers(
    REFERENCE_ID, NOISE_UNMATCHED, DECOMP_KNOWN_MISS, PCT_KNOWN_MISS
)


# --- levels ---------------------------------------------------------------------------


def levels_ok(report) -> bool:
    """Levels gate: cells compared, max gap within TOL_LEVELS, no cell missing or unmatched
    on a side, pre and post both compared, ``base`` the only family-only period and no
    GEMPACK-only one, and every exclusion carries its reason."""
    return (
        report.overall["cells"] > 0
        and report.overall["max"] <= TOL_LEVELS
        and report.overall["missing_in_a"] == report.overall["missing_in_b"] == 0
        and not report.unmatched
        and {"pre", "post"} <= set(report.periods_compared)
        and report.periods_only_in_a == ["base"]
        and not report.periods_only_in_b
        and all(e.why.strip() for e in report.exclusions)
        and all(x.why.strip() for x in report.excluded_cells)
        and all(why.strip() for why in report.excluded.values())
    )


def level_gap(post: float, pre: float, pct: float) -> float:
    """Relative gap of a post level against ``pre * (1 + pct / 100)`` (``rel_diff`` with
    the ``ABS_FLOOR`` of the gtap7 map)."""
    ref = pre * (1.0 + pct / 100.0)
    return abs(post - ref) / max(abs(ref), ABS_FLOOR)


def report_levels_from_pct(run: AltertaxRun, gem_pct) -> dict:
    """Levels gate when the reference folder has no GEMPACK levels (RunGTAP's
    ``sl4dump.har`` holds % changes only): GEMPACK's post level is taken as
    ``pre * (1 + pct / 100)`` with ``pre`` the family CHECK level (which reproduces the
    base: ``CHECK_TOL`` gate of ``test_check_reproduces_base``), and compared with the
    family SHOCK level at ``TOL_LEVELS``. Every compared variable of ``gtap7.yaml`` with a
    GEMPACK % change is covered, in equilibria's own units; zero-base cells are excluded
    as in the pct gate. This is the pct gate restated in levels, not an independent
    measurement of GEMPACK's levels."""
    from equilibria.blocks.gtap._derived_params import xscale_data
    from equilibria.family.altertax import pct as _pct

    cg = _levels.load_compare_script()
    lv = _levels.family_levels(run.result.mp, cg.EQ_VARS)
    side = cg.equilibria_side(lv, dict(xscale_data(run.params, run.params.sets)))
    gem = _pct.gempack_changes(gem_pct)
    zero = zero_base_cells(run)
    rows = _pct.gtap7_rows()
    gaps: list[tuple[float, str, tuple, float, float]] = []
    for name, cells in gem.items():
        row = rows.get(name)
        if row is None or name in _pct.FIXED:
            continue
        perm = [row.a.dims.index(d) for d in row.b.dims]
        pos = {
            tuple(idx[p] for p in perm): idx
            for (period, idx) in side.levels[row.a.var]
            if period == "check"
        }
        for cell, g in cells.items():
            if cell in zero.get(name, ()) or cell not in pos:
                continue
            idx = pos[cell]
            pre = side.levels[row.a.var][("check", idx)]
            post = side.levels[row.a.var][("shock", idx)]
            gaps.append(
                (level_gap(post, pre, g), name, cell, post, pre * (1 + g / 100))
            )
    gaps.sort(key=lambda t: (t[0] != t[0], t[0]), reverse=True)
    mx = gaps[0][0] if gaps else float("nan")
    worst = [
        {"variable": n, "cell": list(c), "ours": o, "gempack": r, "gap": d}
        for d, n, c, o, r in gaps[:10]
    ]
    return _record(
        "levels",
        {"TOL_LEVELS": TOL_LEVELS},
        mx,
        "relative (post = check * (1 + GEMPACK pct / 100))",
        worst,
        bool(gaps) and mx <= TOL_LEVELS,
        known_miss=[],
        cells=len(gaps),
        cells_over_tol=sum(1 for g in gaps if not g[0] <= TOL_LEVELS),
        source="sl4dump.har percent changes (no GEMPACK levels in the folder)",
    )


def report_levels(run: AltertaxRun, gempack_dir: Path, basedata: Path | None) -> dict:
    found = find_files(gempack_dir, {"levels": LEVELS_FILES + (SL4_DUMP,)})["levels"]
    if found.name == SL4_DUMP:
        return report_levels_from_pct(run, read_sl4(found))
    report = _levels.compare_levels(run, gempack_dir, basedata)
    worst = sorted(report.cells, key=lambda c: (c.d != c.d, c.d), reverse=True)[:10]
    mx = report.overall["max"]
    return _record(
        "levels",
        {"TOL_LEVELS": TOL_LEVELS},
        mx,
        "relative",
        [
            {
                "variable": c.variable,
                "period": c.period,
                "cell": list(c.index)
                if isinstance(c.index, (tuple, list))
                else c.index,
                "ours": c.a,
                "gempack": c.b,
                "gap": c.d,
            }
            for c in worst
        ],
        levels_ok(report),
        known_miss=[],
        exclusions=[
            {"variable": e.variable, "where": e.where, "cells": e.cells, "why": e.why}
            for e in report.exclusions
        ],
        excluded=dict(report.excluded),
        excluded_cells=len(report.excluded_cells),
        periods_compared=list(report.periods_compared),
        cells=report.overall["cells"],
    )


# --- everything -----------------------------------------------------------------------


def gempack_files(gempack_dir: Path) -> dict[str, Path]:
    """The GEMPACK files ``compare_all`` reads, found by either naming (GEMPACK's own or
    the burfisher fixture's). ``sl4dump.har`` (Phase C folders) is optional and not read
    yet. Raises ``FileNotFoundError`` naming every missing file."""
    return find_files(
        gempack_dir,
        {
            "updated": UPDATED_FILES,
            "summary": SUMMARY_FILES,
            "decomp": DECOMP_FILES,
            "volume": VOLUME_FILES,
            "pct (solution.sl4)": SL4_FILES,
            "levels": LEVELS_FILES + (SL4_DUMP,),
        },
    )


def json_safe(obj: Any) -> Any:
    """``obj`` with every non-finite float replaced by ``None`` (valid JSON)."""
    if isinstance(obj, float):
        return obj if np.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    if isinstance(obj, np.floating):
        return json_safe(float(obj))
    if isinstance(obj, np.integer):
        return int(obj)
    return obj


def compare_all(
    run: AltertaxRun, gempack_dir: Path, labels: bool | None = None
) -> dict:
    """The six outputs of ``run`` against a GEMPACK run folder: ``{"outputs": [...],
    "pass": bool, "waivers": ...}``. A missing GEMPACK file is a ``FileNotFoundError``
    naming it. The known-miss waivers apply only to the nus333 ALTAX reference.
    ``labels`` (compare header long names): ``None`` = off for a RunGTAP folder (its
    solution is a ``sl4dump.har``, its tab words the data headers differently), on
    otherwise."""
    files = gempack_files(gempack_dir)
    if labels is None:
        labels = files["pct (solution.sl4)"].name != SL4_DUMP
    basedata = Path(run.dataset_dir) / "basedata.har"
    sl4 = read_sl4(files["pct (solution.sl4)"])
    waivers = waivers_for(run.dataset_dir, sl4)
    outputs = [
        report_levels(run, gempack_dir, basedata),
        report_updated(updated_headers(run), read_har(str(files["updated"])), labels),
        report_summary(
            summary_headers(run), read_har(str(files["summary"])), waivers, labels
        ),
        report_volume(run, volume_headers(run), read_har(str(files["volume"])), labels),
        report_decomp_of(run, read_har(str(files["decomp"])), waivers, labels),
        report_pct(run, pct_changes(run), gempack_changes(sl4), waivers),
    ]
    return {
        "outputs": outputs,
        "pass": all(o["pass"] for o in outputs),
        "waivers": waivers.to_dict(),
    }
