"""Extract and analyze ALL data between ROW_START markers."""
from pathlib import Path

gdx_path = Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "set_2d_sparse.gdx"

with open(gdx_path, 'rb') as f:
    content = f.read()

# We know from hex dump that _DATA_ for symbol 3 (map) starts around offset 0x150
# And we see ROW_START at 0x15F
# Let's extract the region 0x150-0x180

start = 0x150
end = 0x180

chunk = content[start:end]

print("=== Hex dump 0x150-0x180 ===")
for i in range(0, len(chunk), 16):
    offset = start + i
    hex_str = ' '.join(f'{b:02X}' for b in chunk[i:i+16])
    ascii_str = ''.join(chr(b) if 32 <= b < 127 else '.' for b in chunk[i:i+16])
    print(f"0x{offset:04X}: {hex_str:48s} | {ascii_str}")

print("\n=== Expected data ===")
print("UEL: agr(1), mfg(2), srv(3), food(4), goods(5), services(6)")
print("Row 1 (agr=1): cols 4(food), 6(services)")
print("Row 2 (mfg=2): cols 5(goods), 6(services)")
print("Row 3 (srv=3): col 6(services)")

print("\n=== Parsing attempt ===")

# Find 01 01 00 00 00 pattern
i = 0
while i < len(chunk):
    if chunk[i] == 0x01 and i + 4 < len(chunk) and chunk[i+2:i+5] == b'\x00\x00\x00':
        row = chunk[i+1]
        print(f"\nROW_START at offset 0x{start+i:04X} (byte {i})")
        print(f"  Row number (1-based): {row}")

        # Extract next 20 bytes
        data_bytes = chunk[i+5:min(i+25, len(chunk))]
        print(f"  Following bytes: {' '.join(f'{b:02X}' for b in data_bytes)}")

        # Look for all 0x05 markers (these seem to indicate columns)
        col_markers = []
        for j, b in enumerate(data_bytes):
            if b == 0x05:
                col_markers.append(j)

        print(f"  Found {len(col_markers)} markers (0x05) at positions: {col_markers}")

        # Try to extract column values
        print("  Column parsing:")
        j = 0
        col_num = 0
        while j < len(data_bytes) - 1:
            if j + 4 <= len(data_bytes):
                # Try int32
                int_val = int.from_bytes(data_bytes[j:j+4], 'little')
                next_byte = data_bytes[j+4] if j+4 < len(data_bytes) else None

                if 1 <= int_val <= 10 and next_byte == 0x05:
                    print(f"    Col {col_num+1}: int32={int_val} (offset {j}, marker at {j+4})")
                    col_num += 1
                    j += 5
                    continue

            # Try single byte + marker
            if j + 1 < len(data_bytes) and data_bytes[j+1] == 0x05:
                print(f"    Col {col_num+1}: byte={data_bytes[j]:02X} (offset {j}, marker at {j+1})")
                col_num += 1
                j += 2
                continue

            j += 1

        # Skip past this row
        i += 5
    else:
        i += 1
