"""Analyze the binary format of sets in GDX files."""

from pathlib import Path
import struct

gdx_path = Path("tests/fixtures/simple_test.gdx")
data = gdx_path.read_bytes()

# Buscar marcadores _DATA_
marker = b"_DATA_\x00\x00"
pos = 0
data_sections = []
while True:
    pos = data.find(marker, pos)
    if pos == -1:
        break
    data_sections.append(pos)
    pos += 1

print(f"Found {len(data_sections)} _DATA_ sections")

# Analizar la primera sección
if data_sections:
    start = data_sections[0]
    end = data_sections[1] if len(data_sections) > 1 else start + 200
    section = data[start:end]
    
    print(f"\nFirst _DATA_ section (length: {len(section)} bytes)")
    print("\nFirst 100 bytes:")
    for i in range(0, min(100, len(section)), 16):
        hex_str = " ".join(f"{b:02x}" for b in section[i:i+16])
        ascii_str = "".join(chr(b) if 32 <= b < 127 else "." for b in section[i:i+16])
        print(f"{i:04x}: {hex_str:48s} {ascii_str}")
