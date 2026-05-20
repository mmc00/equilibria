"""L3: reader<->writer semantic round-trip on real GTAP fixtures.

For each GEMPACK-emitted fixture shipped with equilibria, round-trip:
    original -> read_har -> write_har -> read_har == read_har(original)

at HeaderArray level. Repeated set names (e.g. VMSB on COMM x REG x REG)
must be preserved.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from equilibria.babel.har import read_har, write_har

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA = REPO_ROOT / "src/equilibria/templates/reference/gtap/data"

FIXTURES = [
    DATA / "nus333/sets.har",
    DATA / "nus333/basedata.har",
    DATA / "nus333/baserate.har",
    DATA / "nus333/default.prm",
    DATA / "9x10/sets.har",
    DATA / "9x10/basedata.har",
]


def _ha_equal(a, b) -> tuple[bool, str]:
    if a.name != b.name:
        return False, f"name: {a.name!r} != {b.name!r}"
    if a.long_name != b.long_name:
        return False, f"long_name for {a.name!r}: {a.long_name!r} != {b.long_name!r}"
    if a.set_names != b.set_names:
        return False, f"set_names for {a.name!r}: {a.set_names!r} != {b.set_names!r}"
    if a.set_elements != b.set_elements:
        return False, f"set_elements for {a.name!r}"
    if a.array.dtype.kind != b.array.dtype.kind:
        return False, f"dtype kind for {a.name!r}: {a.array.dtype} vs {b.array.dtype}"
    if a.array.shape != b.array.shape:
        return False, f"shape for {a.name!r}: {a.array.shape} != {b.array.shape}"
    if a.array.dtype == object:
        if [str(x).strip() for x in a.array.tolist()] != [str(x).strip() for x in b.array.tolist()]:
            return False, f"set elements differ for {a.name!r}"
    elif a.array.dtype.kind == "i":
        if not np.array_equal(a.array, b.array):
            return False, f"integer values differ for {a.name!r}"
    else:
        if not np.allclose(a.array, b.array, rtol=0, atol=0):
            return False, f"float values differ for {a.name!r}"
    return True, ""


@pytest.mark.parametrize("fixture", FIXTURES, ids=lambda p: p.name)
def test_semantic_roundtrip(fixture: Path, tmp_path: Path):
    if not fixture.exists():
        pytest.skip(f"fixture not present: {fixture}")
    original = read_har(fixture)
    out = tmp_path / f"rt_{fixture.name}"

    # GEMPACK fixtures contain some non-uppercase header names (e.g. rTGD,
    # rTGM). The writer emits them verbatim with a UserWarning — suppress
    # the noise in the round-trip suite.
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        write_har(out, original)

    roundtripped = read_har(out)
    assert set(original.keys()) == set(roundtripped.keys()), (
        f"header sets differ: missing={set(original)-set(roundtripped)}, "
        f"extra={set(roundtripped)-set(original)}"
    )
    for name in original:
        ok, msg = _ha_equal(original[name], roundtripped[name])
        assert ok, msg
