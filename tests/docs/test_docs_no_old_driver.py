"""The user docs teach the core driver only (PR 3c of gtap-on-core).

The README, the guides and the API pages show ``Exercise(GTAP(...), gtap_static(...))``.
None names an old GTAP driver, nor ``build_block_model`` (the core builds the model
inside ``run()``): a reader who copies a page must not land on code PR 4 deletes.
The maintainer notes (``docs/notes``, ``docs/adr``, ``docs/archive``) record history
and may name them.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.scripts.test_no_old_driver_in_scripts import OLD

ROOT = Path(__file__).resolve().parents[2]
PAGES = [
    *sorted((ROOT / "docs" / "site" / "guide").glob("*.md")),
    *sorted((ROOT / "docs" / "site" / "api").glob("*.md")),
    ROOT / "README.md",
]


def test_there_are_pages() -> None:
    assert len(PAGES) > 10


@pytest.mark.parametrize("page", PAGES, ids=lambda p: p.name)
def test_user_docs_name_no_old_driver(page: Path) -> None:
    text = page.read_text(encoding="utf-8")
    assert not [n for n in (*OLD, "build_block_model") if n in text]
