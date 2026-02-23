"""Check GAMS file to understand set definitions."""

from pathlib import Path

FIXTURES_DIR = Path(__file__).resolve().parents[2] / "tests" / "fixtures"

gms_file = FIXTURES_DIR / "generate_gdx_fixtures.gms"

print("GAMS file content (first 60 lines):")
print("=" * 70)

with open(gms_file) as f:
    for i, line in enumerate(f, 1):
        print(f"{i:3d}: {line}", end='')
        if i >= 60:
            break
