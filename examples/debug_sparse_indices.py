"""Detailed analysis of p3d_sparse structure."""
from pathlib import Path
from equilibria.babel.gdx.reader import read_gdx, read_data_sections
import struct

gdx_file = Path('tests/fixtures/multidim_test.gdx')
data = read_gdx(gdx_file)
raw_data = gdx_file.read_bytes()

# Get sections
sections = read_data_sections(raw_data)

# Find p3d_sparse
idx = None
for i, sym in enumerate(data['symbols']):
    if sym['name'] == 'p3d_sparse':
        idx = i
        print(f"p3d_sparse is at index {i}")
        print(f"Symbol info: dim={sym['dimension']}, records={sym['records']}")
        break

if idx is None:
    print("p3d_sparse not found!")
    exit(1)

_, section = sections[idx]
print(f"\nSection size: {len(section)} bytes\n")

# Parse the section manually
print("="*60)
print("Manual parsing of p3d_sparse:")
print("="*60)

print("\nUEL for reference:")
for i, elem in enumerate(data['elements']):
    print(f"  {i}: {elem}")

print("\nExpected values from GAMS:")
print("  i1.j1.k1 = 111")
print("  i1.j2.k1 = 121")
print("  i2.j3.k2 = 232")
print("  i3.j4.k1 = 341")

print("\nBinary structure (first 200 bytes):")
pos = 19  # Skip header
entry = 0

while pos < min(len(section), 200):
    byte = section[pos]
    
    # Look for row markers
    if (byte == 0x01 and pos + 4 < len(section) and
        section[pos+2] == 0x00 and section[pos+3] == 0x00 and section[pos+4] == 0x00):
        
        dim1_byte = section[pos+1]
        
        # Read next 2 int32s (for 3D parameter)
        if pos + 13 < len(section):
            dim2_int32 = struct.unpack_from("<I", section, pos+5)[0]
            dim3_int32 = struct.unpack_from("<I", section, pos+9)[0]
            
            # Check if there's a double marker
            if section[pos+13] == 0x0a and pos + 22 < len(section):
                value = struct.unpack_from("<d", section, pos+14)[0]
                
                print(f"\nEntry {entry}:")
                print(f"  Position: {pos}")
                print(f"  dim1 byte: {dim1_byte} (0x{dim1_byte:02x})")
                print(f"  dim2 int32: {dim2_int32}")
                print(f"  dim3 int32: {dim3_int32}")
                print(f"  value: {value}")
                
                # Try different interpretations
                print(f"  Interpretation 1 (direct UEL): {data['elements'][dim1_byte]}, {data['elements'][dim2_int32]}, {data['elements'][dim3_int32]}")
                
                # Try as 1-based
                if dim1_byte > 0 and dim2_int32 > 0 and dim3_int32 > 0:
                    if dim1_byte-1 < len(data['elements']) and dim2_int32-1 < len(data['elements']) and dim3_int32-1 < len(data['elements']):
                        print(f"  Interpretation 2 (1-based): {data['elements'][dim1_byte-1]}, {data['elements'][dim2_int32-1]}, {data['elements'][dim3_int32-1]}")
                
                entry += 1
                pos += 22
                continue
    
    pos += 1
