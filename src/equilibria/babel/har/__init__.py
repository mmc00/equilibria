"""equilibria.babel.har — GEMPACK HAR file reader.

Reads .har and .prm files produced by GEMPACK/RunGTAP without requiring
GEMPACK to be installed.

Requires: harpy3>=0.3  (pip install harpy3)
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
