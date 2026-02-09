"""Deep analysis of set data sections in GDX files."""

import struct
from pathlib import Path

from equilibria.babel.gdx.reader import (
    get_sets,
    read_data_sections,
    read_gdx,
)

FIXTURES_DIR = Path(__file__).parent.parent / "tests" / "fixtures"


def analyze_set_data_section():
    """Analyze how sets are stored in _DATA_ sections."""

    gdx_path = FIXTURES_DIR / "variables_equations_test.gdx"
    if not gdx_path.exists():
        print(f"File not found: {gdx_path}")
        return

    # Read the GDX file
    gdx_data = read_gdx(gdx_path)
    raw_data = gdx_path.read_bytes()

    # Get set information
    sets = get_sets(gdx_data)
    print("=" * 70)
    print("SET INFORMATION")
    print("=" * 70)
    for idx, s in enumerate(sets):
        print(f"\nSet #{idx}: {s['name']}")
        print(f"  Description: {s.get('description', 'N/A')}")
        print(f"  Type: {s['type']} ({s['type_name']})")
        print(f"  Dimension: {s['dimension']}")
        print(f"  Records: {s['records']}")

    # Get UEL elements
    elements = gdx_data.get("elements", [])
    print(f"\n\nUEL Elements ({len(elements)} total):")
    for idx, elem in enumerate(elements):
        print(f"  [{idx}] {elem}")

    # Get data sections
    data_sections = read_data_sections(raw_data)
    print(f"\n\n_DATA_ Sections: {len(data_sections)} found")

    # Analyze the first set's data section
    if sets and data_sections:
        # Find which data section corresponds to the first set
        symbols = gdx_data["symbols"]
        set_idx = -1
        for idx, sym in enumerate(symbols):
            if sym["name"] == sets[0]["name"]:
                set_idx = idx
                break

        if set_idx >= 0 and set_idx < len(data_sections):
            sym_idx, section = data_sections[set_idx]

            print("\n\n" + "=" * 70)
            print(f"DATA SECTION FOR SET '{sets[0]['name']}'")
            print("=" * 70)
            print(f"Symbol index: {sym_idx}")
            print(f"Section length: {len(section)} bytes")

            # Display hex dump
            print("\nHex dump (first 150 bytes):")
            for i in range(0, min(150, len(section)), 16):
                hex_str = " ".join(f"{b:02x}" for b in section[i:i+16])
                ascii_str = "".join(
                    chr(b) if 32 <= b < 127 else "." for b in section[i:i+16]
                )
                print(f"{i:04x}: {hex_str:48s} {ascii_str}")

            # Try to parse the section
            print("\n\nParsing attempt:")
            pos = 19  # Skip header
            record_count = 0

            while pos < len(section) - 1:
                byte = section[pos]
                print(f"\n  Pos {pos:4d}: 0x{byte:02x} ({byte:3d})", end="")

                if byte == 0x01:  # Row start
                    if pos + 5 <= len(section):
                        count = section[pos + 1]
                        print(f" -> ROW_START, following count={count}")
                        pos += 2  # Skip marker and count

                        # Read 'count' indices
                        if pos + count <= len(section):
                            indices = []
                            for i in range(count):
                                idx = section[pos + i]
                                indices.append(idx)
                                if idx > 0 and idx <= len(elements):
                                    print(f"      Index[{i}]: {idx} -> '{elements[idx-1]}'")
                                    record_count += 1
                            pos += count
                        continue

                if byte == 0x02:  # Record
                    if pos + 2 <= len(section):
                        idx = section[pos + 1]
                        print(f" -> RECORD, index={idx}", end="")
                        if idx > 0 and idx <= len(elements):
                            print(f" ({elements[idx-1]})")
                            record_count += 1
                        else:
                            print(" (invalid index)")
                        pos += 2
                        continue

                if byte == 0x03:  # Continue
                    print(" -> CONTINUE")
                    pos += 1
                    continue

                if byte in (0x04, 0x06, 0x08):
                    if pos + 4 <= len(section):
                        print(" -> CONTROL_BLOCK")
                        pos += 4
                        continue

                if byte == 0x0a:  # Double
                    if pos + 9 <= len(section):
                        val = struct.unpack_from("<d", section, pos + 1)[0]
                        print(f" -> DOUBLE = {val}")
                        pos += 9
                        continue

                # Check if it's ASCII text
                if 32 <= byte < 127:
                    # Try to read a string
                    text_start = pos
                    text_end = pos
                    while text_end < len(section) and 32 <= section[text_end] < 127:
                        text_end += 1
                    text = section[text_start:text_end].decode('ascii', errors='ignore')
                    print(f" -> TEXT: '{text}'")
                    pos = text_end
                    continue

                print(" -> Unknown")
                pos += 1

            print(f"\n\nTotal records found: {record_count}")
            print(f"Expected records: {sets[0]['records']}")


if __name__ == "__main__":
    analyze_set_data_section()
