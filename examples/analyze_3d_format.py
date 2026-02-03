"""Analyze 3D set binary format."""
from pathlib import Path
import struct

gdx_path = Path(__file__).parent.parent / "tests" / "fixtures" / "set_3d.gdx"

with open(gdx_path, 'rb') as f:
    content = f.read()

print("Expected 3D tuples (from gdxdump):")
print("  ('a', 'x', 'p')")
print("  ('a', 'x', 'q')")
print("  ('a', 'y', 'p')")
print("  ('b', 'x', 'p')")
print("  ('b', 'y', 'q')")
print("  ('c', 'y', 'p')")
print("  ('c', 'y', 'q')")
print()
print("UEL mapping: a=1, b=2, c=3, x=4, y=5, p=6, q=7")
print("Expected indices:")
print("  (1,4,6), (1,4,7), (1,5,6), (2,4,6), (2,5,7), (3,5,6), (3,5,7)")
print()

# Search for patterns with 01 01 00 00 00
pos = 0
found_count = 0
while pos < len(content) - 20 and found_count < 10:
    if (content[pos] == 0x01 and 
        content[pos+2:pos+5] == b'\x00\x00\x00'):
        
        row = content[pos+1]
        print(f"\nFound ROW_START at offset {pos} (0x{pos:04X})")
        print(f"  Row index: {row}")
        
        # Show next 30 bytes
        chunk = content[pos:pos+30]
        hex_str = ' '.join(f'{b:02X}' for b in chunk)
        print(f"  Bytes: {hex_str}")
        
        # Try to decode as int32 values
        print(f"  Decoded as int32s:")
        for i in range(5, min(25, len(chunk)), 5):
            if i + 4 <= len(chunk):
                val = struct.unpack_from("<I", chunk, i)[0]
                marker = chunk[i+4] if i+4 < len(chunk) else 0
                print(f"    Offset {i}: int32={val}, marker=0x{marker:02X}")
        
        found_count += 1
        pos += 30
    else:
        pos += 1
