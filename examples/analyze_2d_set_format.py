"""
Reverse engineer GDX 2D set binary format.

Analyze the exact byte structure of 2D sets to understand how
GAMS encodes multidimensional set elements.
"""

from pathlib import Path
from equilibria.babel.gdx.reader import read_gdx, read_data_sections

def analyze_set_format(gdx_file: Path, set_name: str):
    """Analyze binary format of a specific set."""
    print(f"\n{'='*80}")
    print(f"Analyzing: {gdx_file.name} - Set: {set_name}")
    print(f"{'='*80}\n")
    
    # Read GDX metadata
    gdx_data = read_gdx(gdx_file)
    
    # Find the set
    set_symbol = None
    set_index = -1
    for i, sym in enumerate(gdx_data["symbols"]):
        if sym["name"] == set_name:
            set_symbol = sym
            set_index = i
            break
    
    if not set_symbol:
        print(f"ERROR: Set '{set_name}' not found!")
        return
    
    print(f"Set metadata:")
    print(f"  Name: {set_symbol['name']}")
    print(f"  Type: {set_symbol['type']} ({set_symbol['type_name']})")
    print(f"  Dimension: {set_symbol['dimension']}")
    print(f"  Records: {set_symbol['records']}")
    print(f"  Description: {set_symbol.get('description', 'N/A')}")
    print()
    
    # Show UEL
    print(f"UEL (Unique Element List) - {len(gdx_data['elements'])} elements:")
    for i, elem in enumerate(gdx_data['elements']):
        print(f"  [{i:2d}] (1-based={i+1:2d}): {elem}")
    print()
    
    # Get data section
    raw_data = gdx_file.read_bytes()
    data_sections = read_data_sections(raw_data)
    
    if set_index >= len(data_sections):
        print(f"ERROR: No data section for set index {set_index}")
        return
    
    _, section = data_sections[set_index]
    print(f"Data section: {len(section)} bytes total\n")
    
    # Show header
    print("Header (bytes 0-26):")
    print(f"  Marker: {section[0:6]}")  # _DATA_
    for i in range(6, min(27, len(section))):
        print(f"  [{i:2d}]: 0x{section[i]:02x} ({section[i]:3d})")
    print()
    
    # Analyze data bytes with context
    print("Data bytes (27 onwards) - detailed analysis:")
    print(f"{'Pos':<5} {'Hex':<6} {'Dec':<5} {'ASCII':<6} {'Context'}")
    print("-" * 80)
    
    pos = 27
    record_num = 0
    current_row = None
    
    while pos < len(section):
        byte = section[pos]
        hex_str = f"0x{byte:02x}"
        dec_str = f"{byte:3d}"
        ascii_char = chr(byte) if 32 <= byte < 127 else '.'
        
        # Try to identify context
        context = ""
        
        # Check for row start marker
        if byte == 0x01 and pos + 4 < len(section):
            next_byte = section[pos + 1]
            if 1 <= next_byte <= len(gdx_data['elements']):
                current_row = next_byte - 1  # Convert to 0-based
                row_name = gdx_data['elements'][current_row]
                context = f"ROW_START -> row {next_byte} ({row_name})"
                record_num += 1
        
        # Check if it's a potential column index
        elif byte > 0 and byte <= len(gdx_data['elements']) and current_row is not None:
            col_idx = byte - 1  # Convert to 0-based
            if col_idx < len(gdx_data['elements']):
                col_name = gdx_data['elements'][col_idx]
                row_name = gdx_data['elements'][current_row]
                context = f"COL_IDX? -> {byte} ({col_name}) => ({row_name}, {col_name})"
        
        # Check for common markers
        elif byte == 0x00:
            context = "PADDING/NULL"
        elif byte == 0xFF:
            context = "MARKER 0xFF"
        elif byte == 0x02:
            context = "RECORD_MARKER?"
        elif byte == 0x03:
            context = "DOUBLE_MARKER?"
        elif byte == 0x04:
            context = "CONTROL?"
        elif byte == 0x05:
            context = "COUNT/SIZE?"
        elif byte == 0x06:
            context = "END_MARKER?"
        elif byte == 0x7F:
            context = "MAX_VALUE"
        
        print(f"{pos:<5d} {hex_str:<6} {dec_str:<5} {ascii_char:<6} {context}")
        pos += 1
    
    print(f"\n{'='*80}\n")


def compare_multiple_sets():
    """Compare multiple 2D sets to find patterns."""
    fixtures_dir = Path(__file__).parent.parent / "tests" / "fixtures"
    
    test_cases = [
        ("set_2d_sparse.gdx", "map"),
        ("set_2d_full.gdx", "rr"),
        ("set_2d_cartesian.gdx", "cart"),
    ]
    
    for gdx_file, set_name in test_cases:
        file_path = fixtures_dir / gdx_file
        if file_path.exists():
            analyze_set_format(file_path, set_name)
        else:
            print(f"WARNING: File not found: {file_path}")


if __name__ == "__main__":
    compare_multiple_sets()
