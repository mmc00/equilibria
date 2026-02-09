"""Analyze binary structure using equilibria functions."""
from pathlib import Path

from equilibria.babel.gdx.reader import read_data_sections, read_gdx

gdx_file = Path('tests/fixtures/multidim_test.gdx')
data = read_gdx(gdx_file)

# Show all symbols
print("Symbols:")
for i, sym in enumerate(data['symbols']):
    print(f"  {i}: {sym['name']} - {sym['type_name']} dim={sym['dimension']} records={sym['records']}")

# Read raw data
raw_data = gdx_file.read_bytes()

# Get data sections
sections = read_data_sections(raw_data)
print(f"\nFound {len(sections)} data sections")

# Analyze p3d (should be around index 5)
print("\n" + "="*60)
print("Analyzing p3d (3D dense parameter)")
print("="*60)

p3d_idx = None
for i, sym in enumerate(data['symbols']):
    if sym['name'] == 'p3d':
        p3d_idx = i
        break

if p3d_idx is not None:
    _, section = sections[p3d_idx]
    print(f"Section size: {len(section)} bytes")
    print("First 300 bytes (hex):")
    for i in range(0, min(300, len(section)), 32):
        chunk = section[i:i+32]
        hex_str = ' '.join(f'{b:02x}' for b in chunk)
        ascii_str = ''.join(chr(b) if 32 <= b < 127 else '.' for b in chunk)
        print(f"  {i:04x}: {hex_str:<96} {ascii_str}")

# Analyze p4d
print("\n" + "="*60)
print("Analyzing p4d (4D dense parameter)")
print("="*60)

p4d_idx = None
for i, sym in enumerate(data['symbols']):
    if sym['name'] == 'p4d':
        p4d_idx = i
        break

if p4d_idx is not None:
    _, section = sections[p4d_idx]
    print(f"Section size: {len(section)} bytes")
    print("First 300 bytes (hex):")
    for i in range(0, min(300, len(section)), 32):
        chunk = section[i:i+32]
        hex_str = ' '.join(f'{b:02x}' for b in chunk)
        ascii_str = ''.join(chr(b) if 32 <= b < 127 else '.' for b in chunk)
        print(f"  {i:04x}: {hex_str:<96} {ascii_str}")

# Also show UEL for reference
print("\n" + "="*60)
print("UEL (Unique Element List):")
print("="*60)
for i, elem in enumerate(data['elements']):
    print(f"  {i}: {elem}")
