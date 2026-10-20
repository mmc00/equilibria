"""Parse specs and map repo code to architecture areas.

Pure-Python library behind the spec system (``specs/<area>/``). A CLI ``main()``
is added on top of these functions elsewhere; keep the signatures stable.
"""

# No `from __future__ import annotations`: dataclasses with string annotations break
# when the module is loaded via spec_from_file_location without sys.modules entry.
import re
from dataclasses import dataclass
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib  # type: ignore[no-redef]

SRC_ROOT = "src/equilibria/"
CODE_ROOTS = ("src/", "scripts/")


@dataclass(frozen=True)
class Areas:
    areas: dict[str, tuple[str, ...]]
    unowned: tuple[str, ...]


@dataclass
class Deltas:
    added: dict[str, str]
    modified: dict[str, str]
    removed: list[str]
    warnings: list[str]


def load_areas(path: Path) -> Areas:
    data = tomllib.loads(Path(path).read_text(encoding="utf-8"))
    areas = {k: tuple(v) for k, v in data.get("areas", {}).items()}
    return Areas(areas=areas, unowned=tuple(data.get("unowned", ())))


def _match(path: str, areas: Areas) -> tuple[bool, str | None]:
    """Longest-prefix match. Returns (matched, area); area None means unowned."""
    best_len = -1
    best: str | None = None
    for name, prefixes in areas.areas.items():
        for prefix in prefixes:
            if path.startswith(prefix) and len(prefix) > best_len:
                best_len, best = len(prefix), name
    for prefix in areas.unowned:
        # Tie goes to unowned: an explicit exclusion beats a coarser owner.
        if path.startswith(prefix) and len(prefix) >= best_len:
            best_len, best = len(prefix), None
    return best_len >= 0, best


def area_for(path: str, areas: Areas) -> str | None:
    if not path.startswith(CODE_ROOTS):
        return None
    return _match(path, areas)[1]


def unmapped(paths: list[str], areas: Areas) -> list[str]:
    return [p for p in paths if p.startswith(SRC_ROOT) and not _match(p, areas)[0]]


def _spec_area(path: str, areas: Areas) -> str | None:
    """Area whose ``specs/<area>/`` folder holds ``path`` (deepest wins)."""
    best: str | None = None
    for name in areas.areas:
        if path.startswith(f"specs/{name}/") and (
            best is None or len(name) > len(best)
        ):
            best = name
    return best


def areas_without_spec(changed: list[str], areas: Areas) -> list[str]:
    code = {a for p in changed if (a := area_for(p, areas)) is not None}
    specs = {a for p in changed if (a := _spec_area(p, areas)) is not None}
    return sorted(code - specs)


_HEADING = re.compile(r"^###\s+requirement:\s*(.*?)\s*$", re.IGNORECASE)
_SECTION = re.compile(r"^##\s+(ADDED|MODIFIED|REMOVED)\s+Requirements\s*$", re.I)


def _normalize(lines: list[str]) -> str:
    out: list[str] = []
    for line in lines:
        line = line.rstrip()
        if not line and (not out or not out[-1]):
            continue
        out.append(line)
    return "\n".join(out).strip()


def _parse_blocks(
    lines: list[str], drop_before: bool = False
) -> tuple[dict[str, str], list[str]]:
    reqs: dict[str, str] = {}
    warns: list[str] = []
    name: str | None = None
    body: list[str] = []

    def flush() -> None:
        if name is not None:
            reqs[name] = _normalize(body)

    for line in lines:
        m = _HEADING.match(line)
        if m:
            flush()
            name, body = re.sub(r"\s+", " ", m.group(1)), []
        elif line.startswith("#"):
            flush()
            name, body = None, []
            if line.startswith("###"):
                warns.append(f"malformed requirement heading: {line.strip()!r}")
        elif name is not None:
            if drop_before and line.lstrip().startswith("Before:"):
                continue
            body.append(line)
    flush()
    return reqs, warns


def parse_requirements(md: str) -> tuple[dict[str, str], list[str]]:
    return _parse_blocks(md.splitlines())


def parse_deltas(md: str) -> Deltas:
    sections: dict[str, list[str]] = {"ADDED": [], "MODIFIED": [], "REMOVED": []}
    current: str | None = None
    for line in md.splitlines():
        m = _SECTION.match(line)
        if m:
            current = m.group(1).upper()
        elif line.startswith("## "):
            current = None
        elif current is not None:
            sections[current].append(line)
    added, w1 = _parse_blocks(sections["ADDED"])
    modified, w2 = _parse_blocks(sections["MODIFIED"], drop_before=True)
    removed, w3 = _parse_blocks(sections["REMOVED"])
    return Deltas(added, modified, list(removed), w1 + w2 + w3)


def check_delta(baseline: dict[str, str], deltas: Deltas) -> list[str]:
    """Stateless: a delta already reflected in the baseline counts as synced."""
    errors: list[str] = []
    for name, body in deltas.added.items():
        if name in baseline and baseline[name] != body:
            errors.append(f"ADDED {name!r}: duplicate, baseline has different text")
    for name, body in deltas.modified.items():
        if name not in baseline:
            errors.append(f"MODIFIED {name!r}: requirement not found in baseline")
    return errors


def hand_edits(
    old: dict[str, str], new: dict[str, str], open_deltas: list[Deltas]
) -> list[str]:
    covered: set[str] = set()
    for d in open_deltas:
        covered |= set(d.added) | set(d.modified) | set(d.removed)
    changed = [n for n in {*old, *new} if old.get(n) != new.get(n)]
    return sorted(n for n in changed if n not in covered)
