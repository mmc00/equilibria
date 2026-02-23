"""Analyze 6D dense parameter binary structure."""

import struct
from pathlib import Path


def analyze_6d_dense():
    """Analyze the binary encoding of 6D dense parameter."""
    gdx_path = Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "test_6d.gdx"
    data = gdx_path.read_bytes()

    # Find _DATA_ section for p6d_dense
    data_marker = b'_DATA_'
    pos = 0

    print("Searching for p6d_dense data sections...")

    # Find all _DATA_ markers
    data_sections = []
    while True:
        pos = data.find(data_marker, pos)
        if pos == -1:
            break
        data_sections.append(pos)
        pos += len(data_marker)

    print(f"Found {len(data_sections)} _DATA_ sections")

    # The second one should be p6d_dense (after p6d_sparse)
    if len(data_sections) < 2:
        print("Not enough data sections")
        return

    # Analyze second data section (p6d_dense)
    section_start = data_sections[1]
    pos = section_start + len(data_marker)

    # Read dimension marker (4 bytes after _DATA_)
    dimension = struct.unpack_from("<I", data, pos)[0]
    pos += 4
    print(f"\nDimension: {dimension}")

    # Read section length
    section_len = struct.unpack_from("<I", data, pos)[0]
    pos += 4
    print(f"Section length: {section_len} bytes")

    section_data = data[pos:pos+section_len]

    # Analyze patterns
    print("\nAnalyzing first 500 bytes of section:")
    print("Pos  Byte  Interpretation")
    print("-" * 60)

    i = 0
    pattern_counts = {0x01: 0, 0x02: 0, 0x03: 0, 0x04: 0, 0x05: 0, 0x0a: 0}

    while i < min(500, len(section_data)):
        byte = section_data[i]

        info = ""
        if byte == 0x01:
            pattern_counts[0x01] += 1
            # Full tuple
            info = "Pattern 1: Full tuple follows"
        elif byte == 0x02:
            pattern_counts[0x02] += 1
            info = "Pattern 2: Update dim2+"
        elif byte == 0x03:
            pattern_counts[0x03] += 1
            info = "Pattern 3: Update dim3+"
        elif byte == 0x04:
            pattern_counts[0x04] += 1
            info = "Pattern 4: Update dim4+"
        elif byte == 0x05:
            pattern_counts[0x05] += 1
            info = "Pattern 5: Update dim5+?"
        elif byte == 0x0a:
            pattern_counts[0x0a] += 1
            # Double marker
            if i + 8 < len(section_data):
                value = struct.unpack_from("<d", section_data, i + 1)[0]
                info = f"Double marker: value={value:.0f}"

        if info or byte in [0x01, 0x02, 0x03, 0x04, 0x05, 0x0a]:
            print(f"{i:4d}  0x{byte:02x}  {info}")

        i += 1

    print("\nPattern counts in first 500 bytes:")
    for pattern, count in sorted(pattern_counts.items()):
        if count > 0:
            print(f"  0x{pattern:02x}: {count} occurrences")

if __name__ == "__main__":
    analyze_6d_dense()
