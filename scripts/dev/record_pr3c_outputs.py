"""Record what the Burfisher notebooks and the executed doc pages print (PR 3c oracle).

PR 3c of gtap-on-core moves the notebooks and the user guides from the old GTAP
driver to ``Exercise``. Run on ``main`` before the change, this writes every line
each notebook and each page prints; after the change, the tests compare against it
(``tests/_printed.py``). The runs are the tests' own: a notebook's code cells as one
script from its folder (``test_burfisher_notebooks._run``), a page's ``python``
blocks as one script from a temporary folder with ``data/`` and ``tests/`` linked
(``test_doc_examples``).

usage: python scripts/dev/record_pr3c_outputs.py --root <checkout> --out FILE
           [--only NAME ...]
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

PAGES = (
    "README.md",
    "docs/site/guide/gtap_quickstart.md",
    "docs/site/guide/welfare_decomposition.md",
)


def _notebook_code(nb: Path) -> str:
    cells = json.loads(nb.read_text(encoding="utf-8"))["cells"]
    return "\n\n".join("".join(c["source"]) for c in cells if c["cell_type"] == "code")


def _run(code: str, cwd: Path, env: dict) -> list[str]:
    with tempfile.TemporaryDirectory() as tmp:
        script = Path(tmp) / "run.py"
        script.write_text(code, encoding="utf-8")
        r = subprocess.run(
            [sys.executable, str(script)],
            cwd=cwd,
            env=env,
            capture_output=True,
            text=True,
            timeout=3600,
        )
    if r.returncode != 0:
        raise RuntimeError(f"{cwd}: exit {r.returncode}\n{r.stderr[-3000:]}")
    return r.stdout.splitlines()


def _page(root: Path, rel: str) -> list[str]:
    sys.path.insert(0, str(root))
    from tests.docs.test_doc_examples import _script

    with tempfile.TemporaryDirectory() as tmp:
        for folder in ("data", "tests"):
            (Path(tmp) / folder).symlink_to(root / folder, target_is_directory=True)
        env = {**os.environ, "MPLBACKEND": "Agg"}
        return _run(_script(root / rel), Path(tmp), env)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--jobs", type=int, default=4)
    a = ap.parse_args(argv)
    root = a.root.resolve()
    nb_dir = root / "examples" / "notebooks" / "gtap7" / "burfisher"
    notebooks = sorted(nb_dir.glob("*.ipynb"))
    if a.only:
        notebooks = [n for n in notebooks if n.stem.upper() in a.only]
    env = dict(os.environ)

    def one(nb: Path) -> tuple[str, list[str]]:
        print(f"notebook {nb.stem}", flush=True)
        return nb.stem.upper(), _run(_notebook_code(nb), nb_dir, env)

    with ThreadPoolExecutor(a.jobs) as pool:
        out_nb = dict(pool.map(one, notebooks))
    pages = [p for p in PAGES if not a.only or Path(p).name in a.only]
    out_docs = {}
    for rel in pages:
        print(f"page {rel}", flush=True)
        out_docs[Path(rel).name] = _page(root, rel)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(
        json.dumps({"notebooks": out_nb, "docs": out_docs}, indent=1, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
