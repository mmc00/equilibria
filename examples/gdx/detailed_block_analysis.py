"""
Detailed block-by-block analysis with ALL bytes shown.
"""

from pathlib import Path

from equilibria.babel.gdx.reader import read_data_sections, read_gdx

gdx = read_gdx('tests/fixtures/set_2d_sparse.gdx')
data = Path('tests/fixtures/set_2d_sparse.gdx').read_bytes()
sections = read_data_sections(data)
_, section = sections[2]

print('Expected tuples (from GAMS):')
print('  (1,4) agr.food')
print('  (1,6) agr.services')
print('  (2,5) mfg.goods')
print('  (2,6) mfg.services')
print('  (3,6) srv.services')
print()

print('UEL: [0]=agr, [1]=mfg, [2]=srv, [3]=food, [4]=goods, [5]=services')
print('     1-based: 1=agr, 2=mfg, 3=srv, 4=food, 5=goods, 6=services')
print()

# Parse row by row
pos = 27
blocks = []

while pos < len(section):
    if section[pos] == 0x01 and pos + 1 < len(section):
        row_idx = section[pos + 1]

        # Find next ROW_START or end
        start = pos
        pos += 2
        while pos < len(section) and section[pos] != 0x01:
            pos += 1
        end = pos

        block_bytes = list(section[start:end])
        blocks.append((row_idx, block_bytes))
    else:
        pos += 1

for row_idx, block_bytes in blocks:
    row_name = gdx['elements'][row_idx - 1]
    print(f'\n{"="*70}')
    print(f'Row {row_idx} ({row_name}):')
    print(f'{"="*70}')
    print('Raw bytes:', ' '.join(f'{b:02x}' for b in block_bytes))
    print()

    # Skip first 5 bytes (01 <row> 00 00 00)
    data_bytes = block_bytes[5:]
    print(f'Data part (after header): {" ".join(f"{b:02x}" for b in data_bytes)}')
    print()

    # Try to identify pattern
    print('Attempting to parse:')
    i = 0
    entry_num = 0
    while i < len(data_bytes):
        # Look for pattern: <col> 00 00 00 <marker>
        if i + 4 < len(data_bytes):
            col_candidate = data_bytes[i]
            zeros = data_bytes[i+1:i+3]
            marker = data_bytes[i+3] if i+3 < len(data_bytes) else None

            if zeros == [0, 0] and 4 <= col_candidate <= 6:
                entry_num += 1
                col_name = gdx['elements'][col_candidate - 1]
                marker_str = f'0x{marker:02x}' if marker is not None else 'None'
                print(f'  Entry {entry_num}: col={col_candidate} ({col_name}), marker={marker_str}')
                i += 4

                # Skip marker byte
                if i < len(data_bytes) and data_bytes[i] in (0x04, 0x05, 0x06):
                    # This might be another column!
                    next_col = data_bytes[i]
                    if 4 <= next_col <= 6:
                        col2_name = gdx['elements'][next_col - 1]
                        print(f'    -> Possible col2={next_col} ({col2_name})?')
                    i += 1
                elif i < len(data_bytes):
                    print(f'    -> Next byte: 0x{data_bytes[i]:02x}')
                    i += 1
            else:
                i += 1
        else:
            i += 1
