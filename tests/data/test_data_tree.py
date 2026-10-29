"""The repository's data/ is clean: every file has a verified source (ADR-0010).

A branch whose phase is still open may set EQUILIBRIA_ALLOW_PENDING=1; CI on
main never does.
"""

from __future__ import annotations

import os

from equilibria.sources import check_tree


def test_data_tree_is_clean() -> None:
    allow = os.environ.get("EQUILIBRIA_ALLOW_PENDING") == "1"
    assert check_tree(allow_pending=allow) == []
