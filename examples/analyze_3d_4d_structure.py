"""Analyze binary structure of 3D/4D parameters."""
from pathlib import Path
import struct

gdx_file = Path('tests/fixtures/multidim_test.gdx')
data = gdx_file.read_bytes()

# Find _DATA_ sections
data_marker = b'_DATA_'
pos = 0
sections = []

while True:
    pos = data.find(data_marker, pos)
    if pos == -1:
        break
    
    # Read size
    size_pos = pos + len(data_marker)
    if size_pos + 4 > len(data):
        break
    
    size = struct.unpack_from("<I", data, size_pos)[0]
    section_start = size_pos + 4
    section_end = section_start + size
    
    if section_end <= len(data):
        sections.append(data[section_start:section_end])
    
    pos = section_end

print(f"Found {len(sections)} _DATA_ sections\n")

# Analyze p3d (should be section index 5)
print("="*60)
print("Analyzing p3d (3D dense parameter)")
print("="*60)
section = sections[5]  # p3d
print(f"Section size: {len(section)} bytes")
print(f"First 200 bytes (hex):")
hex_view = ' '.join(f'{b:02x}' for b in section[:200])
for i in range(0, min(200, len(section)), 32):
    chunk = section[i:i+32]
    hex_str = ' '.join(f'{b:02x}' for b in chunk)
    ascii_str = ''.join(chr(b) if 32 <= b < 127 else '.' for b in chunk)
    print(f"  {i:04x}: {hex_str:<96} {ascii_str}")

print("\n" + "="*60)
print("Analyzing p4d (4D dense parameter)")  
print("="*60)
section = sections[7]  # p4d
print(f"Section size: {len(section)} bytes")
print(f"First 200 bytes (hex):")
for i in range(0, min(200, len(section)), 32):
    chunk = section[i:i+32]
    hex_str = ' '.join(f'{b:02x}' for b in chunk)
    ascii_str = ''.join(chr(b) if 32 <= b < 127 else '.' for b in chunk)
    print(f"  {i:04x}: {hex_str:<96} {ascii_str}")

print("\n" + "="*60)
print("Analyzing p3d_sparse (3D sparse parameter)")
print("="*60)
section = sections[6]  # p3d_sparse
print(f"Section size: {len(section)} bytes")
print(f"First 200 bytes (hex):")
for i in range(0, min(200, len(section)), 32):
    chunk = section[i:i+32]
    hex_str = ' '.join(f'{b:02x}' for b in chunk)
    ascii_str = ''.join(chr(b) if 32 <= b < 127 else '.' for b in chunk)
    print(f"  {i:04x}: {hex_str:<96} {ascii_str}")
