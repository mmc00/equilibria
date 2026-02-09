"""Detailed hex analysis of 2D set format."""
import struct
from pathlib import Path

gdx_path = Path(__file__).parent.parent / "tests" / "fixtures" / "set_2d_sparse.gdx"

with open(gdx_path, 'rb') as f:
    content = f.read()

# Find _DATA_ section
data_marker = b'_DATA_\x00\x00'
data_start = content.find(data_marker)
if data_start == -1:
    print("_DATA_ section not found")
    exit(1)

data_start += len(data_marker)

# Skip to third symbol (map) - each symbol starts with a marker
# We need to find symbol headers
print("=== Symbol Headers ===")
pos = data_start
symbol_count = 0
symbol_starts = []

while pos < len(content) - 10 and symbol_count < 3:
    # Look for potential symbol start pattern
    # Usually: 0C 00 00 00 (12 bytes header) followed by dimension, type, records
    if content[pos:pos+4] == b'\x0c\x00\x00\x00':
        print(f"Possible symbol {symbol_count + 1} at offset {pos} (0x{pos:X})")
        symbol_starts.append(pos)
        symbol_count += 1
        pos += 100  # Skip ahead to find next symbol
    else:
        pos += 1

print(f"\nFound {symbol_count} symbols")
print()

if symbol_count >= 3:
    # Analyze third symbol (map - the 2D set)
    sym_start = symbol_starts[2]
    print(f"=== Third Symbol (map) at offset {sym_start} (0x{sym_start:X}) ===")

    # Read first 100 bytes of this symbol
    chunk = content[sym_start:sym_start + 150]

    print("First 150 bytes (hex):")
    for i in range(0, len(chunk), 16):
        hex_str = ' '.join(f'{b:02X}' for b in chunk[i:i+16])
        ascii_str = ''.join(chr(b) if 32 <= b < 127 else '.' for b in chunk[i:i+16])
        print(f"{sym_start+i:6d} (0x{sym_start+i:04X}): {hex_str:48s} | {ascii_str}")

    print()
    print("=== Detailed Byte-by-Byte Analysis ===")

    # Expected:
    # Row 1 (agr=1): cols 4 (food), 6 (services)
    # Row 2 (mfg=2): cols 5 (goods), 6 (services)
    # Row 3 (srv=3): col 6 (services)

    print("Expected tuples:")
    print("  (1, 4) = agr.food")
    print("  (1, 6) = agr.services")
    print("  (2, 5) = mfg.goods")
    print("  (2, 6) = mfg.services")
    print("  (3, 6) = srv.services")
    print()

    # Look for ROW_START markers (0x01)
    print("Searching for ROW_START markers (0x01):")
    for i in range(len(chunk)):
        if chunk[i] == 0x01 and i + 10 < len(chunk):
            print(f"\nOffset {sym_start+i} (0x{sym_start+i:04X}), byte {i}:")
            print(f"  Context: {' '.join(f'{b:02X}' for b in chunk[i:i+20])}")

            # Try to parse as row
            if i + 1 < len(chunk):
                row_idx = chunk[i + 1]
                print(f"  Row index (1-based): {row_idx}")

                # Look for int32 values after padding
                j = i + 2
                while j < i + 40 and j < len(chunk):
                    # Check if we have 4 bytes for int32
                    if j + 4 <= len(chunk):
                        val = struct.unpack_from("<I", chunk, j)[0]
                        if 1 <= val <= 10:  # Likely a UEL index
                            next_byte = chunk[j + 4] if j + 4 < len(chunk) else 0
                            print(f"    Offset {j}: int32={val}, next_byte=0x{next_byte:02X}")
                    j += 1
