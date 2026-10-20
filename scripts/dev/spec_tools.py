"""Parse specs and map repo code to architecture areas, plus the spec-flow CLI.

Library behind the spec system (``specs/<area>/``) with a ``main()`` that offers
``new-area``, ``new-change``, ``new-adr``, ``archive`` and ``check-pr``.
Run from the repository root.
"""

# No `from __future__ import annotations`: dataclasses with string annotations break
# when the module is loaded via spec_from_file_location without sys.modules entry.
import argparse
import datetime
import fnmatch
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
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
    docs: dict[str, tuple[str, ...]] = field(default_factory=dict)


@dataclass
class Deltas:
    added: dict[str, str]
    modified: dict[str, str]
    removed: list[str]
    warnings: list[str]


def load_areas(path: Path) -> Areas:
    data = tomllib.loads(Path(path).read_text(encoding="utf-8"))
    areas = {k: tuple(v) for k, v in data.get("areas", {}).items()}
    docs = {k: tuple(v) for k, v in data.get("docs", {}).items()}
    return Areas(areas=areas, unowned=tuple(data.get("unowned", ())), docs=docs)


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


def stale_docs(changed: list[str], areas: Areas) -> list[str]:
    """Areas whose baseline changed while none of their user-doc pages did."""
    stale: list[str] = []
    for area, patterns in areas.docs.items():
        prefix = f"specs/{area}/specs/"
        if not any(p.startswith(prefix) for p in changed):
            continue
        if not any(fnmatch.fnmatchcase(p, g) for p in changed for g in patterns):
            stale.append(area)
    return sorted(stale)


_HEADING = re.compile(r"^###\s+requirement:\s*(.*?)\s*$", re.IGNORECASE)
_FENCE = re.compile(r"^\s*```")
_BOUNDARY = re.compile(r"^#{1,3}\s")  # headings of level 1-3 end a block
_LEVEL3 = re.compile(r"^###\s")
_SECTION = re.compile(r"^##\s+(ADDED|MODIFIED|REMOVED)\s+Requirements\s*$", re.I)
# File-level stamp, not part of any requirement (older baselines put it last).
_LAST_SYNCED = re.compile(r"^\s*Last synced:", re.IGNORECASE)


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

    in_fence = False
    for line in lines:
        if _FENCE.match(line):
            in_fence = not in_fence
        m = None if in_fence else _HEADING.match(line)
        if m:
            flush()
            name, body = re.sub(r"\s+", " ", m.group(1)), []
        elif not in_fence and _BOUNDARY.match(line):
            flush()
            name, body = None, []
            if _LEVEL3.match(line):
                warns.append(f"malformed requirement heading: {line.strip()!r}")
        elif name is not None:
            if drop_before and line.lstrip().startswith("Before:"):
                continue
            if _LAST_SYNCED.match(line):
                continue
            body.append(line)
    flush()
    return reqs, warns


def parse_requirements(md: str) -> tuple[dict[str, str], list[str]]:
    return _parse_blocks(md.splitlines())


def parse_deltas(md: str) -> Deltas:
    sections: dict[str, list[str]] = {"ADDED": [], "MODIFIED": [], "REMOVED": []}
    current: str | None = None
    in_fence = False
    for line in md.splitlines():
        if _FENCE.match(line):
            in_fence = not in_fence
        m = None if in_fence else _SECTION.match(line)
        if m:
            current = m.group(1).upper()
        elif not in_fence and re.match(r"^#{1,2}\s", line):
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


def sync_mismatches(
    old: dict[str, str], new: dict[str, str], open_deltas: list[Deltas]
) -> list[str]:
    """Covered requirements whose new baseline text is not what a delta states.

    A changed requirement named in a delta is synced only if some delta matches
    it: ADDED/MODIFIED text equal to the new baseline text, or REMOVED with the
    requirement gone. Uncovered names are left to ``hand_edits``.
    """
    bad: list[str] = []
    for n in {*old, *new}:
        if old.get(n) == new.get(n):
            continue
        named = [d for d in open_deltas if n in {*d.added, *d.modified, *d.removed}]
        if not named:
            continue
        ok = any(
            (n in new and new[n] in (d.added.get(n), d.modified.get(n)))
            or (n not in new and n in d.removed)
            for d in named
        )
        if not ok:
            bad.append(n)
    return sorted(bad)


# --- CLI ---------------------------------------------------------------------

SPECS = Path("specs")
TEMPLATES = SPECS / "_templates"
SLUG = re.compile(r"^[a-z0-9][a-z0-9-]*$")
AREA_KEY = re.compile(r"^[a-z0-9][a-z0-9_-]*(/[a-z0-9][a-z0-9_-]*)*$")
BASELINE = re.compile(r"^specs/(.+)/specs/[^/]+\.md$")
OPEN_CHANGE = re.compile(r"^specs/(.+)/changes/([^/]+)/specs/[^/]+\.md$")
SKIP_LABELS = {"no-delta", "bootstrap"}
NO_DOCS_LABEL = "no-docs"


class _Fail(Exception):
    def __init__(self, msg: str, code: int = 1) -> None:
        super().__init__(msg)
        self.code = code


def _check_name(value: str, pattern: re.Pattern[str], what: str) -> None:
    if not pattern.match(value):
        raise _Fail(f"invalid {what}: {value!r}")


def _copy_template(name: str, dest: Path) -> None:
    src = TEMPLATES / name
    if not src.is_dir():
        raise _Fail(f"template not found: {src}")
    shutil.copytree(src, dest)


def _cmd_new_area(args: argparse.Namespace) -> int:
    _check_name(args.area, AREA_KEY, "area")
    dest = SPECS / args.area
    if dest.exists():
        raise _Fail(f"area already exists: {dest}")
    _copy_template("new-area", dest)
    print(f"created {dest}")
    return 0


def _cmd_new_change(args: argparse.Namespace) -> int:
    _check_name(args.area, AREA_KEY, "area")
    _check_name(args.slug, SLUG, "slug")
    if not (SPECS / args.area).is_dir():
        raise _Fail(f"area not found: {SPECS / args.area}")
    dest = SPECS / args.area / "changes" / args.slug
    if dest.exists():
        raise _Fail(f"change already exists: {dest}")
    _copy_template("new-change", dest)
    print(f"created {dest}")
    return 0


def _cmd_new_adr(args: argparse.Namespace) -> int:
    _check_name(args.slug, SLUG, "slug")
    adr_dir = Path("docs") / "adr"
    numbers = [
        int(m.group(1))
        for p in adr_dir.glob("*.md")
        if (m := re.match(r"^(\d{4})-", p.name))
    ]
    dest = adr_dir / f"{max(numbers, default=0) + 1:04d}-{args.slug}.md"
    adr_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(TEMPLATES / "adr.md", dest)
    print(f"created {dest}")
    return 0


def _cmd_archive(args: argparse.Namespace) -> int:
    _check_name(args.area, AREA_KEY, "area")
    _check_name(args.slug, SLUG, "slug")
    base = SPECS / args.area / "changes"
    src = base / args.slug
    if not src.is_dir():
        raise _Fail(f"change not found: {src}")
    dest = base / "archive" / f"{datetime.date.today().isoformat()}-{args.slug}"
    if dest.exists():
        raise _Fail(f"archive target already exists: {dest}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(dest))
    print(f"archived to {dest}")
    return 0


def _git(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["git", *args], capture_output=True, text=True)


def _show(rev: str, path: str) -> str:
    """File content at ``rev`` or empty string if it does not exist there."""
    out = _git("show", f"{rev}:{path}")
    return out.stdout if out.returncode == 0 else ""


def _ignored(path: str) -> bool:
    return path.startswith("specs/_templates/") or "/changes/archive/" in path


def _baseline_reqs(area: str, rev: str) -> dict[str, str]:
    reqs: dict[str, str] = {}
    listing = _git("ls-tree", "--name-only", rev, f"specs/{area}/specs/").stdout
    for path in listing.splitlines():
        if path.endswith(".md"):
            reqs.update(parse_requirements(_show(rev, path))[0])
    return reqs


def _has_baseline(area: str, rev: str) -> bool:
    out = _git("ls-tree", "--name-only", rev, f"specs/{area}/specs/")
    return out.returncode == 0 and bool(out.stdout.strip())


def _archived_changes(
    base: str, changed: list[str], areas: Areas
) -> list[tuple[str, str, str, Deltas]]:
    """Changes open at ``base`` that this PR moved into ``changes/archive/``."""
    listing = _git("ls-tree", "-r", "--name-only", base, "specs/").stdout
    found: list[tuple[str, str, str, Deltas]] = []
    for rel in listing.splitlines():
        m = OPEN_CHANGE.match(rel)
        if not m or _ignored(rel) or not _spec_area(rel, areas):
            continue
        area, slug = m.group(1), m.group(2)
        if _show("HEAD", rel):
            continue  # still open at HEAD; read from there
        moved = re.compile(
            rf"^specs/{re.escape(area)}/changes/archive/"
            rf"\d{{4}}-\d{{2}}-\d{{2}}-{re.escape(slug)}/"
        )
        if any(moved.match(p) for p in changed):
            found.append((area, slug, rel, parse_deltas(_show(base, rel))))
    return found


def _cmd_check_pr(args: argparse.Namespace) -> int:
    base = args.base
    if _git("rev-parse", "--verify", "--quiet", f"{base}^{{commit}}").returncode:
        raise _Fail(f"base ref not found: {base}", 2)
    diff = _git("diff", "--name-only", f"{base}...HEAD")
    if diff.returncode:
        raise _Fail(
            f"cannot diff {base}...HEAD (shallow clone or no merge base?): "
            f"{diff.stderr.strip()}",
            2,
        )
    all_changed = [p for p in diff.stdout.splitlines() if p]
    changed = [p for p in all_changed if not _ignored(p)]
    labels = {x.strip() for x in args.labels.split(",") if x.strip()}
    skip = bool(labels & SKIP_LABELS)
    areas = load_areas(SPECS / "areas.toml")
    errors: list[str] = []
    warnings: list[str] = []

    if not skip:
        for a in areas_without_spec(changed, areas):
            if not _has_baseline(a, "HEAD"):
                continue  # area not specified yet; nothing to keep in sync
            warnings.append(f"area '{a}' has code changes but no spec change")
    for p in unmapped(changed, areas):
        warnings.append(f"path not mapped to any area in specs/areas.toml: {p}")

    open_changes: list[tuple[str, str, str, Deltas]] = []
    for path in sorted(Path(".").glob("specs/**/changes/*/specs/*.md")):
        rel = path.as_posix()
        m = OPEN_CHANGE.match(rel)
        if m and not _ignored(rel) and _spec_area(rel, areas):
            deltas = parse_deltas(_show("HEAD", rel))
            open_changes.append((m.group(1), m.group(2), rel, deltas))

    # Changes archived by this PR still justify the syncs it makes (the last PR
    # of a change may sync and archive at once). Their deltas are read at base.
    archived_here = _archived_changes(base, all_changed, areas)

    for area, slug, rel, deltas in open_changes:
        prefix = f"specs/{area}/changes/{slug}/"
        if not any(p.startswith(prefix) for p in changed):
            continue
        for w in deltas.warnings:
            warnings.append(f"{rel}: {w}")
        for e in check_delta(_baseline_reqs(area, "HEAD"), deltas):
            errors.append(f"{rel}: {e}")

    if not skip:
        for path in changed:
            m = BASELINE.match(path)
            if not m or "/changes/" in path:
                continue
            area = m.group(1)
            old = parse_requirements(_show(base, path))[0]
            new = parse_requirements(_show("HEAD", path))[0]
            deltas_in_area = [
                d for a, _, _, d in open_changes + archived_here if a == area
            ]
            for name in hand_edits(old, new, deltas_in_area):
                errors.append(
                    f"{path}: requirement {name!r} edited by hand; "
                    f"describe it in a change (specs/{area}/changes/<slug>/) "
                    "or label the PR no-delta"
                )
            for name in sync_mismatches(old, new, deltas_in_area):
                errors.append(
                    f"{path}: requirement {name!r} does not match its delta in "
                    f"specs/{area}/changes/; sync the text exactly as the delta "
                    "states it (ADDED/MODIFIED) or delete it (REMOVED)"
                )

    if NO_DOCS_LABEL not in labels:
        for a in stale_docs(changed, areas):
            warnings.append(
                f"area '{a}' baseline changed but none of its user docs did "
                f"(see [docs] in specs/areas.toml); update them or label the PR "
                f"{NO_DOCS_LABEL}"
            )

    for w in warnings:
        print(f"::warning::{w}")
    for e in errors:
        print(f"::error::{e}")
    return 1 if errors else 0


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="spec_tools", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("new-area")
    s.add_argument("area")
    s.set_defaults(fn=_cmd_new_area)
    s = sub.add_parser("new-change")
    s.add_argument("area")
    s.add_argument("slug")
    s.set_defaults(fn=_cmd_new_change)
    s = sub.add_parser("new-adr")
    s.add_argument("slug")
    s.set_defaults(fn=_cmd_new_adr)
    s = sub.add_parser("archive")
    s.add_argument("area")
    s.add_argument("slug")
    s.set_defaults(fn=_cmd_archive)
    s = sub.add_parser("check-pr")
    s.add_argument("--base", required=True)
    s.add_argument("--labels", default="")
    s.set_defaults(fn=_cmd_check_pr)
    return p


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        return args.fn(args)
    except _Fail as exc:
        print(f"::error::{exc}" if args.cmd == "check-pr" else f"error: {exc}")
        return exc.code


if __name__ == "__main__":
    sys.exit(main())
