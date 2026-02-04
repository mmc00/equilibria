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

__all__ = ["read_gdx", "SAM"]
