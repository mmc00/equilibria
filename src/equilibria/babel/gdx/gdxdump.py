"""Helpers to run GAMS gdxdump from Python."""

from __future__ import annotations

import glob
import os
import re
import shutil
import subprocess
from pathlib import Path

_GDXDUMP_PARAM_CACHE: dict[tuple[str, str], dict[tuple[str, ...], float]] = {}
_GDXDUMP_SET_CACHE: dict[tuple[str, str], list[str]] = {}
_GDXDUMP_VAR_CACHE: dict[tuple[str, str], dict[tuple[str, ...], float]] = {}

_VARIABLE_LEVEL_PATTERN = re.compile(
    r"((?:'[^']*'\.)*'[^']*')\.(L|LO|UP)\s+(-?\d+(?:\.\d+)?(?:[Ee][+-]?\d+)?)"
)


def _orden_de_version(directorio: str) -> tuple[int, tuple[int, ...]]:
    """Clave de orden para los directorios de version de GAMS.

    `Current` es el symlink a la instalacion activa y va primero. El resto se
    ordena por version NUMERICA descendente: ordenarlos como texto pone "9"
    por delante de "53" y de "48", que es justo lo que se quiere evitar.
    """
    nombre = Path(directorio).parent.name
    if nombre == "Current":
        return (0, ())
    partes = tuple(-int(p) for p in re.findall(r"\d+", nombre)) or (0,)
    return (1, partes)


def locate_gdxdump() -> str | None:
    """Donde esta `gdxdump`, el binario que lee los .gdx de GAMS.

    Era una ruta absoluta a GAMS 48 dentro del paquete PUBLICADO: funcionaba en
    el Mac del autor y en ninguna otra maquina, y ademas el servidor de
    licencias dejo de aceptar la v48 (HTTP 400, "Node ID for category 2 not
    defined"), asi que tambien dejo de funcionar ahi.

    Orden: `EQUILIBRIA_GDXDUMP` (escape hatch explicito) -> PATH -> las rutas de
    instalacion habituales, `Current` primero y luego version numerica
    descendente. `None` si no aparece: quien llama decide si saltar o fallar.
    """
    if explicito := os.environ.get("EQUILIBRIA_GDXDUMP"):
        return explicito
    if en_path := shutil.which("gdxdump"):
        return en_path
    patrones = (
        "/Library/Frameworks/GAMS.framework/Versions/*/Resources",
        "/opt/gams/*",
        "/usr/local/gams/*",
        "C:/GAMS/*/*",
    )
    candidatos: list[str] = []
    for pat in patrones:
        candidatos.extend(glob.glob(pat))
    for d in sorted(candidatos, key=_orden_de_version):
        for nombre in ("gdxdump", "gdxdump.exe"):
            cand = Path(d) / nombre
            if cand.exists():
                return str(cand)
    return None


def read_parameter_with_gdxdump(
    gdx_path: Path,
    symbol_name: str,
) -> dict[tuple[str, ...], float]:
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
) -> dict[tuple[str, ...], float]:
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
) -> dict[tuple[str, ...], float]:
    try:
        proc = subprocess.run(
            [gdxdump, str(gdx_path), f"Symb={symbol_name}"],
            capture_output=True,
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

    start_slash = output.find("/", start)
    end_slash = output.find("/;", start_slash)
    if start_slash == -1 or end_slash == -1:
        return {}

    body = output[start_slash + 1 : end_slash]
    return _parse_variable_body(body)


def _parse_variable_body(body: str) -> dict[tuple[str, ...], float]:
    entries: dict[tuple[str, ...], float] = {}

    for match in _VARIABLE_LEVEL_PATTERN.finditer(body):
        key_segment, attr, value_text = match.groups()
        if attr != "L":
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
) -> dict[tuple[str, ...], float]:
    # Try CSV format first (more reliable parsing)
    try:
        proc = subprocess.run(
            [gdxdump, str(gdx_path), f"Symb={symbol_name}", "format=csv"],
            capture_output=True,
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
            capture_output=True,
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

    start_slash = output.find("/", start)
    end_slash = output.find("/;", start_slash)
    if start_slash == -1 or end_slash == -1:
        return {}

    body = output[start_slash + 1 : end_slash]
    return _parse_gdxdump_parameter_body(body)


def read_set_with_gdxdump(gdx_path: Path, set_name: str) -> list[str]:
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
            capture_output=True,
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

    start_slash = output.find("/", start)
    end_slash = output.find("/;", start_slash)
    if start_slash == -1 or end_slash == -1:
        _GDXDUMP_SET_CACHE[key] = []
        return []

    body = output[start_slash + 1 : end_slash]
    entries = re.findall(r"'([^']*)'", body)
    _GDXDUMP_SET_CACHE[key] = entries
    return entries


def _parse_gdxdump_parameter_body(body: str) -> dict[tuple[str, ...], float]:
    entries: dict[tuple[str, ...], float] = {}
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


def _parse_csv_parameter(csv_output: str) -> dict[tuple[str, ...], float]:
    """Parse CSV format output from gdxdump.

    Expected format:
    "Dim1","Dim2","Val"
    "key1","key2",1.23
    """
    import csv
    import io

    entries: dict[tuple[str, ...], float] = {}

    reader = csv.reader(io.StringIO(csv_output))
    header = next(reader, None)
    if not header:
        return {}

    # Find value column (usually last, named "Val" or "Value")
    val_idx = len(header) - 1
    for i, col in enumerate(header):
        if col.lower() in ("val", "value"):
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
