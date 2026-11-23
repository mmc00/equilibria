#!/usr/bin/env bash
# Pre-push hook: the small datasets on the family (JAX) route against the Pyomo
# photos, plus the photo freshness check. The photos are MUMPS runs, so the Python
# needs jax and pymumps: EQUILIBRIA_FAMILY_PYTHON (e.g. /opt/homebrew/envs/eq-mumps/bin/python)
# or, without it, the uv env. A Python without them stops the push with a message,
# never a silent skip; `git push --no-verify` skips the hook on purpose.
set -euo pipefail
root=$(git rev-parse --show-toplevel)
cd "$root"
if [ -n "${EQUILIBRIA_FAMILY_PYTHON:-}" ]; then
  py=("$EQUILIBRIA_FAMILY_PYTHON")
  export PYTHONNOUSERSITE=1 PYTHONPATH="$root/src:$root${PYTHONPATH:+:$PYTHONPATH}"
else
  py=(uv run python)
fi
if ! "${py[@]}" -c "import jax, mumps" 2>/dev/null; then
  echo "pre-push family: ${py[*]} has no jax or no pymumps." >&2
  echo "  set EQUILIBRIA_FAMILY_PYTHON to a Python with both (the MUMPS env)," >&2
  echo "  or skip this hook once with: git push --no-verify" >&2
  exit 1
fi
start=$(date +%s)
pytest=("${py[@]}" -m pytest -q -p no:cacheprovider -o addopts= -m "not slow and not gams")
# every photo fresh (all variants), then the small datasets against them
"${pytest[@]}" tests/family/test_photo_freshness.py
"${pytest[@]}" tests/family -k "gtap7_3x3 or gtap7_3x4 or gtap7_5x5" \
  --deselect tests/family/test_photo_freshness.py
# the 3x3 photo comparisons whose test ids carry no dataset (-k misses them)
"${pytest[@]}" tests/family/test_family_e2e.py tests/family/test_family_e2e_capflex.py \
  tests/family/test_family_e2e_ifsub0.py tests/family/test_squared_system.py \
  tests/family/test_mp_arrays.py tests/family/test_fscale.py tests/family/test_photos.py \
  tests/family/test_pairing_cache.py \
  -k "not (gtap7_3x3 or gtap7_3x4 or gtap7_5x5)"
echo "pre-push family: $(( ($(date +%s) - start) / 60 )) min $(( ($(date +%s) - start) % 60 )) s"
