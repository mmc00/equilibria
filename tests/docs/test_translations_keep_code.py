"""A translated document keeps its code and its numbers.

Phase 6 of ``specs/data/changes/one-data-folder`` translated the living Spanish
documents to English. A translation may only change prose: every fenced code
block must be byte-identical, and every number must keep its digits.

Numbers are compared as the multiset of digit runs (``\\d+``). That ignores the
separators (``0,099`` and ``0.099`` both give ``0`` and ``099``; ``1.058`` and
``1,058`` both give ``1`` and ``058``), which a translation to English rightly
changes, and still catches any digit that changed, appeared or vanished.

The Spanish text is read from ``BASE``, the last commit of ``main`` before the
phase. Code blocks are compared after applying the phase's path renames to the
Spanish side (a code block may name a moved file). Shallow clones (CI) do not
have ``BASE``; the test then skips.
"""

from __future__ import annotations

import re
import subprocess
from collections import Counter
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
BASE = "e2a34e72"

# Spanish document at BASE -> its English translation today.
TRANSLATED = {
    "ROADMAP.md": "docs/roadmap.md",
    "docs/architecture/monolito_vs_bloques.md": "docs/notes/architecture/monolith_vs_blocks.md",
    "docs/architecture/registro_de_bloques.md": "docs/notes/architecture/no_block_registry.md",
    "docs/reference/gdx/gdx_reader_guide.md": "docs/notes/gdx/reader_guide.md",
    "docs/reference/gdx/multidim_parameters.md": "docs/notes/gdx/multidim_parameters.md",
    "docs/technical/matrix_balancing_methods.md": "docs/notes/sam/matrix_balancing_methods.md",
    "docs/technical/mip_balancing_methods.md": "docs/notes/sam/mip_balancing_methods.md",
    "docs/technical/ras_variants_comparison.md": "docs/notes/sam/ras_variants_comparison.md",
    "docs/guides/bolivia_mip_to_sam_complete_guide.md": "docs/notes/guides/bolivia_mip_to_sam_complete_guide.md",
    "docs/guides/model_jacobian_harness.md": "docs/notes/guides/model_jacobian_harness.md",
    "docs/guides/simple_open_gams_parity.md": "docs/notes/guides/simple_open_gams_parity.md",
    "docs/guides/simple_open_parity_contract.md": "docs/notes/guides/simple_open_parity_contract.md",
    "docs/guides/simple_open_stdcge_mapping.md": "docs/notes/guides/simple_open_stdcge_mapping.md",
    "docs/guides/simulations_api.md": "docs/notes/guides/simulations_api.md",
}

_FENCE = re.compile(r"^(```|~~~)[^\n]*\n.*?^\1[ \t]*$", re.MULTILINE | re.DOTALL)


def _git(*args: str) -> str:
    return subprocess.check_output(["git", "-C", str(ROOT), *args]).decode("utf-8")


def _has_base() -> bool:
    probe = subprocess.run(
        ["git", "-C", str(ROOT), "cat-file", "-e", f"{BASE}^{{commit}}"],
        capture_output=True,
    )
    return probe.returncode == 0


def _renames() -> list[tuple[str, str]]:
    """(old, new) path pairs this phase renamed, longest first, plus basenames."""
    out = _git("diff", "-M", "--name-status", BASE, "HEAD")
    pairs = []
    for line in out.splitlines():
        parts = line.split("\t")
        if parts[0].startswith("R"):
            old, new = parts[1], parts[2]
            pairs.append((old, new))
            if Path(old).name != Path(new).name:
                pairs.append((Path(old).name, Path(new).name))
    return sorted(pairs, key=lambda p: -len(p[0]))


def code_blocks(text: str) -> list[str]:
    return [m.group(0) for m in _FENCE.finditer(text)]


def digit_runs(text: str) -> Counter[str]:
    return Counter(re.findall(r"\d+", text))


def _renamed(text: str, renames: list[tuple[str, str]]) -> str:
    for old, new in renames:
        text = text.replace(old, new)
    return text


needs_base = pytest.mark.skipif(not _has_base(), reason=f"{BASE} not in this clone")


@needs_base
@pytest.mark.parametrize("old,new", sorted(TRANSLATED.items()))
def test_code_blocks_are_unchanged(old: str, new: str) -> None:
    spanish = _renamed(_git("show", f"{BASE}:{old}"), _renames())
    english = (ROOT / new).read_text(encoding="utf-8")
    assert code_blocks(english) == code_blocks(spanish)


@needs_base
@pytest.mark.parametrize("old,new", sorted(TRANSLATED.items()))
def test_numbers_keep_their_digits(old: str, new: str) -> None:
    spanish = _git("show", f"{BASE}:{old}")
    english = (ROOT / new).read_text(encoding="utf-8")
    assert digit_runs(english) == digit_runs(spanish)


def test_digit_runs_ignore_separators_but_not_digits() -> None:
    assert digit_runs("0,099 % of 1.058") == digit_runs("0.099% of 1,058")
    assert digit_runs("0,099") != digit_runs("0.098")


def test_code_blocks_see_a_changed_line() -> None:
    a = "text\n```python\nx = 1\n```\nmore\n"
    b = "texto\n```python\nx = 2\n```\nmás\n"
    assert code_blocks(a) == ["```python\nx = 1\n```"]
    assert code_blocks(a) != code_blocks(b)
