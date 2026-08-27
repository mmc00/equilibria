"""GTAP 6.2 CGE template (Hertel/Itakura/McDougall 2003).

Built on the symbolic `equilibria.blocks` framework (see
`equilibria.blocks.gtap6`), reusing the pattern F3 proved for GTAP7 but
with 6.2's own block units — v6.2 has no make-matrix, no ND intermediate
bundle, no output CET, and `cgds` is a producing sector, not an agent.
"""

from __future__ import annotations

__all__: list[str] = []
