"""Phase 3.37 — force deterministic Python startup.

The v6.2 closure logic (``_make_square.apply_v62_closure_and_square``)
calls ``nx.bipartite.maximum_matching`` (Hopcroft–Karp). NetworkX uses
``set`` for BFS/DFS layers, so when Python hash randomization is on
(default), the algorithm picks **different equivalent matchings**
between processes. Different matchings fix different variables, which
changes the MCP system PATH sees, which changes the solve outcome.

Importing this module BEFORE any other heavy import re-execs the
interpreter with ``PYTHONHASHSEED=0`` if needed. Effect is invisible
to the user; the script behaves as if launched with that env var set.

Usage at the top of an entry-point script::

    import _deterministic_startup  # noqa: F401  must be first import

The module is a no-op if ``PYTHONHASHSEED`` is already set to a fixed
value (anything other than ``random``).
"""
from __future__ import annotations

import os
import sys

_TARGET = "0"
_DONE_MARKER = "EQUILIBRIA_HASH_SEED_LOCKED"


def _ensure_locked_hash_seed() -> None:
    """Re-exec the current Python process with ``PYTHONHASHSEED=0`` if
    the seed is currently random.

    Three conditions short-circuit the re-exec:
    * ``PYTHONHASHSEED=0`` is already in the environment
    * The marker env var is already set (re-exec already happened)
    * ``sys.flags.hash_randomization == 0`` (interpreter was built with
      hash randomization disabled at compile time)
    """
    if os.environ.get(_DONE_MARKER):
        return
    if os.environ.get("PYTHONHASHSEED") == _TARGET:
        return
    if sys.flags.hash_randomization == 0:
        return

    new_env = dict(os.environ)
    new_env["PYTHONHASHSEED"] = _TARGET
    new_env[_DONE_MARKER] = "1"

    # On Windows ``os.execvpe`` would replace the current process but
    # the parent shell does not get the exit code right; ``os.execv``
    # works because we already have a full path to the interpreter.
    if sys.platform == "win32":
        # Windows: ``execv`` family returns immediately and the parent
        # process becomes a zombie reading stale output. Use ``subprocess``
        # so the parent's stdout/stderr stream uninterrupted.
        #
        # CAVEAT: ``Stop-Process`` on the PARENT does NOT propagate to
        # the spawned child. To kill the actual computation, terminate
        # the child Python process (it is the larger-WorkingSet entry
        # in ``Get-Process python``).
        import subprocess
        rc = subprocess.call([sys.executable] + sys.argv, env=new_env)
        sys.exit(rc)
    else:
        os.execve(sys.executable, [sys.executable] + sys.argv, new_env)


_ensure_locked_hash_seed()
