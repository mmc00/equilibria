"""
equilibria.babel - Universal data I/O for economic modeling.

Supports reading and writing:
- GDX (GAMS Data Exchange)
- HAR (GEMPACK Header Array)
- Excel
- CSV
- GTAP databases
"""

from equilibria.babel.gdx import read_gdx
from equilibria.babel.sam import SAM
from equilibria.babel.sam_loader import SAM4D, SAM4DLoader, load_sam_4d

__all__ = ["read_gdx", "SAM", "SAM4D", "SAM4DLoader", "load_sam_4d"]
