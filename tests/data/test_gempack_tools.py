"""The scripts that generate GEMPACK results live in data/gtap7/gempack/tools.

Every GEMPACK result records the sha256 of the whole data/gtap7/gempack/ folder
(--model-file), so editing a generator makes the results it produced stale.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from equilibria.sources import DATA_ROOT, load_source, own_files, sha256_file

ROOT = Path(__file__).resolve().parents[2]
TOOLS = ("regenerate_gempack.py", "run_gempack_matrix.py")
OLD = {
    "regenerate_gempack.py": ROOT / "scripts" / "windows",
    "run_gempack_matrix.py": ROOT / "scripts" / "gtap",
}


def test_gempack_model_folder() -> None:
    from equilibria.sources import GEMPACK_MODEL_DIR

    assert GEMPACK_MODEL_DIR == DATA_ROOT / "gtap7" / "gempack"
    assert load_source(GEMPACK_MODEL_DIR).kind == "model"


@pytest.mark.parametrize("name", TOOLS)
def test_gempack_generators_live_with_their_model(name: str) -> None:
    from equilibria.sources import GEMPACK_MODEL_DIR

    assert (GEMPACK_MODEL_DIR / "tools" / name).is_file()
    assert not (OLD[name] / name).exists()


def _hashes(folder: Path) -> dict[str, str]:
    return {p.relative_to(folder).as_posix(): sha256_file(p) for p in own_files(folder)}


def test_pycache_is_not_part_of_a_record(tmp_path: Path) -> None:
    # running a tool writes __pycache__; that must not make results stale
    from equilibria.sources import GEMPACK_MODEL_DIR

    copy = tmp_path / "gempack"
    shutil.copytree(
        GEMPACK_MODEL_DIR, copy, ignore=shutil.ignore_patterns("__pycache__")
    )
    before = _hashes(copy)
    cache = copy / "tools" / "__pycache__"
    cache.mkdir()
    (cache / "run_gempack_matrix.cpython-313.pyc").write_bytes(b"\x00")
    assert _hashes(copy) == before
    (copy / "tools" / "run_gempack_matrix.py").write_text("# edited\n")
    assert _hashes(copy) != before
