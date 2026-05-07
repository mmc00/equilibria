"""equilibria.babel.har — GEMPACK HAR file reader.

Native parser for the .har / .prm binary format produced by
GEMPACK / RunGTAP. No external GEMPACK or harpy3 dependency.
"""

from equilibria.babel.har.reader import (
    get_header_names,
    read_har,
    read_header_array,
)
from equilibria.babel.har.symbols import HeaderArray

__all__ = [
    "HeaderArray",
    "get_header_names",
    "read_har",
    "read_header_array",
]
