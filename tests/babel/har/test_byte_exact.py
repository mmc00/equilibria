"""L4: byte-for-byte equality between write_har(read_har(p)) and p.

Per spec section 4, the writer targets byte equality against the
GEMPACK-emitted source for the six GTAP fixtures shipped with equilibria.
A fixture that cannot reach byte equality due to GEMPACK non-determinism
is xfailed with a sibling .diff file documenting the divergence.
"""
from __future__ import annotations

import hashlib
import warnings
from pathlib import Path

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

# Populated as Task 9 confirms which fixtures pass / fail byte-exact.
# Each entry maps fixture-name -> human-readable reason from the .diff sidecar.
KNOWN_NON_BYTE_EXACT: dict[str, str] = {}


@pytest.mark.parametrize("fixture", FIXTURES, ids=lambda p: p.name)
def test_byte_exact(fixture: Path, tmp_path: Path):
    if not fixture.exists():
        pytest.skip(f"fixture not present: {fixture}")
    if fixture.name in KNOWN_NON_BYTE_EXACT:
        pytest.xfail(KNOWN_NON_BYTE_EXACT[fixture.name])

    original = read_har(fixture)
    out = tmp_path / f"be_{fixture.name}"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        write_har(out, original)

    original_sha = hashlib.sha256(fixture.read_bytes()).hexdigest()
    written_sha = hashlib.sha256(out.read_bytes()).hexdigest()

    if original_sha != written_sha:
        # Include the dataset directory (nus333/9x10) in the sidecar name so
        # files with the same stem (sets.har, basedata.har in both datasets)
        # don't clobber each other.
        diff_path = (
            REPO_ROOT
            / "tests/babel/har"
            / f"{fixture.parent.name}_{fixture.stem}.diff"
        )
        diff_path.parent.mkdir(parents=True, exist_ok=True)
        orig_bytes = fixture.read_bytes()
        new_bytes = out.read_bytes()
        lines = [
            f"# byte-exact diff for {fixture.name}",
            f"# original size={len(orig_bytes)} sha256={original_sha}",
            f"# written  size={len(new_bytes)} sha256={written_sha}",
        ]
        for i in range(min(len(orig_bytes), len(new_bytes))):
            if orig_bytes[i] != new_bytes[i]:
                lines.append(
                    f"offset {i:#x}: orig={orig_bytes[i]:02x} new={new_bytes[i]:02x}"
                )
                if len(lines) > 200:
                    lines.append("... (truncated)")
                    break
        diff_path.write_text("\n".join(lines))

    assert original_sha == written_sha, (
        f"byte-exact mismatch for {fixture.name}; "
        f"see tests/babel/har/{fixture.parent.name}_{fixture.stem}.diff"
    )
