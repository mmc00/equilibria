"""Helpers to run GAMS gdxdump from Python."""

from __future__ import annotations

from pathlib import Path
import re
import shutil
import subprocess
from typing import Dict, List, Optional, Tuple

_GDXDUMP_PARAM_CACHE: Dict[Tuple[str, str], Dict[Tuple[str, ...], float]] = {}
_GDXDUMP_SET_CACHE: Dict[Tuple[str, str], List[str]] = {}
_GDXDUMP_VAR_CACHE: Dict[Tuple[str, str], Dict[Tuple[str, ...], float]] = {}

_VARIABLE_LEVEL_PATTERN = re.compile(
    r"((?:'[^']*'\.)*'[^']*')\.(L|LO|UP)\s+(-?\d+(?:\.\d+)?(?:[Ee][+-]?\d+)?)"
)


def locate_gdxdump() -> Optional[str]:
    path = shutil.which("gdxdump")
    if path:
        return path
    fallback = Path("/Library/Frameworks/GAMS.framework/Versions/48/Resources/gdxdump")
    if fallback.exists():
        return str(fallback)
    return None


def read_parameter_with_gdxdump(
    gdx_path: Path,
    symbol_name: str,
) -> Dict[Tuple[str, ...], float]:
    key = (str(gdx_path), symbol_name)
    if key in _GDXDUMP_PARAM_CACHE:
        return _GDXDUMP_PARAM_CACHE[key]

    gdxdump = locate_gdxdump()
    if not gdxdump:
        _GDXDUMP_PARAM_CACHE[key] = {}
        return {}

    candidates = [symbol_name]
    upper = symbol_name.upper()
    if upper != symbol_name:
        candidates.append(upper)

    for candidate in candidates:
        parsed = _read_parameter_from_gdxdump(gdxdump, gdx_path, candidate)
        if parsed:
            _GDXDUMP_PARAM_CACHE[key] = parsed
            return parsed

    _GDXDUMP_PARAM_CACHE[key] = {}
    return {}


def read_variable_levels_with_gdxdump(
    gdx_path: Path,
    symbol_name: str,
) -> Dict[Tuple[str, ...], float]:
    key = (str(gdx_path), symbol_name)
    if key in _GDXDUMP_VAR_CACHE:
        return _GDXDUMP_VAR_CACHE[key]

    gdxdump = locate_gdxdump()
    if not gdxdump:
        _GDXDUMP_VAR_CACHE[key] = {}
        return {}

    candidates = [symbol_name]
    upper = symbol_name.upper()
    if upper != symbol_name:
        candidates.append(upper)

    for candidate in candidates:
        parsed = _read_variable_levels_from_gdxdump(gdxdump, gdx_path, candidate)
        if parsed:
            _GDXDUMP_VAR_CACHE[key] = parsed
            return parsed

    _GDXDUMP_VAR_CACHE[key] = {}
    return {}


def _read_variable_levels_from_gdxdump(
    gdxdump: str,
    gdx_path: Path,
    symbol_name: str,
) -> Dict[Tuple[str, ...], float]:
    try:
        proc = subprocess.run(
            [gdxdump, str(gdx_path), f"Symb={symbol_name}"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        return {}

    if proc.returncode != 0:
        return {}

    output = proc.stdout
    marker = f"Variable {symbol_name}"
    start = output.find(marker)
    if start == -1:
        return {}

    start_slash = output.find('/', start)
    end_slash = output.find('/;', start_slash)
    if start_slash == -1 or end_slash == -1:
        return {}

    body = output[start_slash + 1 : end_slash]
    return _parse_variable_body(body)


def _parse_variable_body(body: str) -> Dict[Tuple[str, ...], float]:
    entries: Dict[Tuple[str, ...], float] = {}

    for match in _VARIABLE_LEVEL_PATTERN.finditer(body):
        key_segment, attr, value_text = match.groups()
        if attr != 'L':
            continue
        keys = tuple(re.findall(r"'([^']*)'", key_segment))
        if not keys:
            continue
        try:
            value = float(value_text)
        except ValueError:
            continue
        entries[keys] = value

    return entries


def _read_parameter_from_gdxdump(
    gdxdump: str,
    gdx_path: Path,
    symbol_name: str,
) -> Dict[Tuple[str, ...], float]:
    # Try CSV format first (more reliable parsing)
    try:
        proc = subprocess.run(
            [gdxdump, str(gdx_path), f"Symb={symbol_name}", "format=csv"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        
        if proc.returncode == 0 and proc.stdout:
            # Parse CSV format
            entries = _parse_csv_parameter(proc.stdout)
            if entries:
                return entries
    except FileNotFoundError:
        return {}
    
    # Fallback to standard format
    try:
        proc = subprocess.run(
            [gdxdump, str(gdx_path), f"Symb={symbol_name}"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        return {}

    if proc.returncode != 0:
        return {}

    output = proc.stdout
    marker = f"Parameter {symbol_name}"
    start = output.find(marker)
    if start == -1:
        return {}

    start_slash = output.find('/', start)
    end_slash = output.find('/;', start_slash)
    if start_slash == -1 or end_slash == -1:
        return {}

    body = output[start_slash + 1 : end_slash]
    return _parse_gdxdump_parameter_body(body)


def read_set_with_gdxdump(gdx_path: Path, set_name: str) -> List[str]:
    key = (str(gdx_path), set_name)
    if key in _GDXDUMP_SET_CACHE:
        return _GDXDUMP_SET_CACHE[key]

    gdxdump = locate_gdxdump()
    if not gdxdump:
        _GDXDUMP_SET_CACHE[key] = []
        return []

    try:
        proc = subprocess.run(
            [gdxdump, str(gdx_path), f"Symb={set_name}"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        _GDXDUMP_SET_CACHE[key] = []
        return []

    if proc.returncode != 0:
        _GDXDUMP_SET_CACHE[key] = []
        return []

    output = proc.stdout
    marker = f"Set {set_name}"
    start = output.find(marker)
    if start == -1:
        _GDXDUMP_SET_CACHE[key] = []
        return []

    start_slash = output.find('/', start)
    end_slash = output.find('/;', start_slash)
    if start_slash == -1 or end_slash == -1:
        _GDXDUMP_SET_CACHE[key] = []
        return []

    body = output[start_slash + 1 : end_slash]
    entries = re.findall(r"'([^']*)'", body)
    _GDXDUMP_SET_CACHE[key] = entries
    return entries


def _parse_gdxdump_parameter_body(body: str) -> Dict[Tuple[str, ...], float]:
    entries: Dict[Tuple[str, ...], float] = {}
    fragments = []
    current = []
    in_quote = False

    for char in body:
        if char == "'":
            in_quote = not in_quote
        if char == "," and not in_quote:
            part = "".join(current).strip()
            if part:
                fragments.append(part)
            current = []
            continue
        current.append(char)

    trailing = "".join(current).strip()
    if trailing:
        fragments.append(trailing)

    for entry in fragments:
        entry = entry.strip()
        if not entry:
            continue
        match = re.search(r"(-?\d+(?:\.\d+)?(?:[Ee][+-]?\d+)?)$", entry)
        if not match:
            continue
        try:
            value = float(match.group(1))
        except ValueError:
            continue
        key_part = entry[: match.start()].strip()
        keys = tuple(re.findall(r"'([^']*)'", key_part))
        if not keys:
            continue
        entries[keys] = value

    return entries


def _parse_csv_parameter(csv_output: str) -> Dict[Tuple[str, ...], float]:
    """Parse CSV format output from gdxdump.
    
    Expected format:
    "Dim1","Dim2","Val"
    "key1","key2",1.23
    """
    import csv
    import io
    
    entries: Dict[Tuple[str, ...], float] = {}
    
    reader = csv.reader(io.StringIO(csv_output))
    header = next(reader, None)
    if not header:
        return {}
    
    # Find value column (usually last, named "Val" or "Value")
    val_idx = len(header) - 1
    for i, col in enumerate(header):
        if col.lower() in ('val', 'value'):
            val_idx = i
            break
    
    # All columns before value column are keys
    key_indices = list(range(val_idx))
    
    for row in reader:
        if len(row) <= val_idx:
            continue
        
        try:
            value = float(row[val_idx])
        except (ValueError, IndexError):
            continue
        
        keys = tuple(row[i] for i in key_indices if i < len(row))
        if keys:
            entries[keys] = value
    
    return entries
