"""GTAP HAR → equilibria wrapper.

Specialized loader that takes a directory containing the four canonical
GTAP HAR/PRM artifacts (``basedata*.har``, ``sets*.har``, ``default*.prm``,
optionally ``baserate*.har``) and produces a fully populated
``GTAPParameters`` instance — sets, elasticities, benchmark monetary flows,
tax rates, and calibrated CES/CET shares — without going through GDX.

Naming convention follows the GEMPACK Standard 7 distribution
(``basedata-9x10.har`` / ``sets-9x10.har`` / ``default-9x10.prm`` …),
optionally suffixed by an aggregation tag (``-9x10``, ``-10x10``, …).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from equilibria.templates.gtap.gtap_parameters import GTAPParameters


_PREFIXES = {
    "basedata": ("basedata",),
    "sets": ("sets",),
    "default": ("default",),
    "baserate": ("baserate",),
}


def _resolve(har_dir: Path, kind: str, suffix: str | None, *, ext: str) -> Path | None:
    """Find the file ``<prefix><suffix>.<ext>`` in ``har_dir`` for the given kind.

    If ``suffix`` is provided, requires an exact match. Otherwise picks the
    single matching file (or ``None`` if missing / ambiguous when optional).
    """
    prefixes = _PREFIXES[kind]
    if suffix is not None:
        for prefix in prefixes:
            candidate = har_dir / f"{prefix}{suffix}.{ext}"
            if candidate.is_file():
                return candidate
        return None

    matches: list[Path] = []
    for prefix in prefixes:
        matches.extend(sorted(har_dir.glob(f"{prefix}*.{ext}")))
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        # Prefer the unsuffixed file if present (e.g. ``basedata.har``).
        for prefix in prefixes:
            exact = har_dir / f"{prefix}.{ext}"
            if exact in matches:
                return exact
        raise FileNotFoundError(
            f"Multiple {kind} files in {har_dir} ({[m.name for m in matches]}); "
            "pass suffix= to disambiguate."
        )
    return None


def load_gtap_from_har(
    har_dir: str | Path,
    *,
    suffix: str | None = None,
    require_baserate: bool = False,
) -> "GTAPParameters":
    """Load a GTAP Standard 7 dataset from HAR/PRM files into a ``GTAPParameters``.

    Args:
        har_dir: Directory containing ``basedata*.har``, ``sets*.har``,
            ``default*.prm`` and optionally ``baserate*.har``.
        suffix: Aggregation tag (e.g. ``"-9x10"``). When omitted, the loader
            picks the unique file per kind, or the unsuffixed one if multiple
            tags exist in the directory.
        require_baserate: If True, raise when ``baserate*.har`` is absent.
            By default the wrapper skips it and derives all tax rates from
            the benchmark SAM (matching ``derive_from_benchmark``).

    Returns:
        A ``GTAPParameters`` with sets, elasticities, benchmark, taxes and
        calibrated shares populated.

    Raises:
        FileNotFoundError: If a required file is missing or the directory
            contains multiple ambiguous candidates.
    """
    from equilibria.templates.gtap.gtap_parameters import GTAPParameters

    har_path = Path(har_dir)
    if not har_path.is_dir():
        raise FileNotFoundError(f"{har_path} is not a directory")

    basedata_path = _resolve(har_path, "basedata", suffix, ext="har")
    sets_path = _resolve(har_path, "sets", suffix, ext="har")
    default_path = _resolve(har_path, "default", suffix, ext="prm")
    baserate_path = _resolve(har_path, "baserate", suffix, ext="har")

    missing = [
        kind
        for kind, path in (
            ("basedata*.har", basedata_path),
            ("sets*.har", sets_path),
            ("default*.prm", default_path),
        )
        if path is None
    ]
    if require_baserate and baserate_path is None:
        missing.append("baserate*.har")
    if missing:
        raise FileNotFoundError(
            f"Missing required GTAP HAR artifacts in {har_path}: {missing}"
        )

    params = GTAPParameters()
    params.sets.load_from_har(sets_path, default_path=default_path)
    params.elasticities.load_from_har(default_path, params.sets)
    params.benchmark.load_from_har(basedata_path, params.sets)
    # makb arrives via basedata.har (after sets.har), so rebuild the output
    # structure now — sets.load_from_har was called with no make data.
    params.sets.rebuild_output_structure_from_makb(params.benchmark.makb)
    if baserate_path is not None:
        params.taxes.load_from_har(baserate_path, params.sets, params.benchmark)
    else:
        # No baserate.har — derive every rate from the benchmark SAM.
        params.taxes.derive_from_benchmark(params.benchmark, params.sets)
    params.shares.calibrate(params.benchmark, params.elasticities, params.sets)
    params.calibrated.calibrate_from_benchmark(
        params.benchmark, params.elasticities, params.sets, params.taxes
    )
    return params
