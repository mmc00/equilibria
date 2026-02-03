"""Test reading 5D parameters from GDX."""
from pathlib import Path
from equilibria.babel.gdx.reader import read_gdx, read_parameter_values

gdx_file = Path('tests/fixtures/test_5d.gdx')
data = read_gdx(gdx_file)

# Show all symbols
print("Symbols in file:")
for sym in data['symbols']:
    print(f"  {sym['name']}: {sym['type_name']} dim={sym['dimension']} records={sym['records']}")

print("\n" + "="*60)
print("Testing 5D sparse parameter")
print("="*60)

values = read_parameter_values(data, 'p5d_sparse')
print(f"\nRead {len(values)} values")

expected = {
    ('i1','j1','k1','m1','n1'): 11111.0,
    ('i1','j2','k1','m1','n2'): 12112.0,
    ('i1','j2','k2','m2','n1'): 12221.0,
    ('i2','j3','k1','m2','n2'): 23122.0,
    ('i2','j3','k2','m1','n1'): 23211.0,
    ('i2','j3','k2','m2','n2'): 23222.0,
}

print("\nExpected values:")
for key, val in sorted(expected.items()):
    status = "✓" if key in values and abs(values[key] - val) < 0.001 else "✗"
    actual = values.get(key, "MISSING")
    print(f"  {status} {key} = {val} (actual: {actual})")

print(f"\nAll values correct: {len(values) == len(expected) and all(key in values and abs(values[key] - val) < 0.001 for key, val in expected.items())}")

print("\n" + "="*60)
print("Analyzing binary structure of p5d_sparse")
print("="*60)

from equilibria.babel.gdx.reader import read_data_sections
import struct

raw_data = gdx_file.read_bytes()
sections = read_data_sections(raw_data)

# Find p5d_sparse
idx = None
for i, sym in enumerate(data['symbols']):
    if sym['name'] == 'p5d_sparse':
        idx = i
        break

if idx is not None:
    _, section = sections[idx]
    print(f"\nSection size: {len(section)} bytes")
    print("\nFirst 400 bytes (hex):")
    for i in range(0, min(400, len(section)), 32):
        chunk = section[i:i+32]
        hex_str = ' '.join(f'{b:02x}' for b in chunk)
        ascii_str = ''.join(chr(b) if 32 <= b < 127 else '.' for b in chunk)
        print(f"  {i:04x}: {hex_str:<96} {ascii_str}")
    
    print("\n\nUEL for reference:")
    for i, elem in enumerate(data['elements']):
        print(f"  {i}: {elem}")

print("\n" + "="*60)
print("Testing 5D dense parameter (partial support expected)")
print("="*60)

values_dense = read_parameter_values(data, 'p5d_dense')
print(f"\nRead {len(values_dense)} values (expected: {2*3*2*2*2} = 48)")

if len(values_dense) > 0:
    print("\nFirst 10 values:")
    for i, (key, val) in enumerate(list(values_dense.items())[:10]):
        print(f"  {key} = {val}")
