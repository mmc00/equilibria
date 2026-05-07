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
from equilibria.babel.har.to_equilibria import load_gtap_from_har
from equilibria.babel.har.to_gdx import convert_har_to_gdx

__all__ = [
    "HeaderArray",
    "convert_har_to_gdx",
    "get_header_names",
    "load_gtap_from_har",
    "read_har",
    "read_header_array",
]
