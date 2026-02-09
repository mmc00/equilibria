"""
Manual parsing attempt for 2D set format.
"""

from pathlib import Path

from equilibria.babel.gdx.reader import read_data_sections, read_gdx

# Analyze set_2d_sparse specifically
gdx = read_gdx('tests/fixtures/set_2d_sparse.gdx')
data = Path('tests/fixtures/set_2d_sparse.gdx').read_bytes()
sections = read_data_sections(data)
_, section = sections[2]  # map

# Expected tuples from GAMS:
expected = [
    (1, 4),  # agr.food
    (1, 6),  # agr.services
    (2, 5),  # mfg.goods
    (2, 6),  # mfg.services
    (3, 6),  # srv.services
]

print('Expected (1-based indices):')
for row, col in expected:
    row_name = gdx['elements'][row-1]
    col_name = gdx['elements'][col-1]
    print(f'  ({row}, {col}) = ({row_name}, {col_name})')

print('\nHex dump with interpretation:')
pos = 27
while pos < len(section):
    print(f'{pos:3d}: 0x{section[pos]:02x}  ', end='')

    # Print next 10 bytes for context
    for i in range(pos, min(pos + 10, len(section))):
        print(f'{section[i]:02x} ', end='')
    print()

    if pos >= 62:
        break
    pos += 1

print('\n' + '='*80)
print('PATTERN HYPOTHESIS:')
print('='*80)
print('Row block format: 01 <row_idx> 00 00 00 <col1> 00 00 00 XX <col2> XX ...')
print('where XX appears to be some kind of counter or marker (often 05)')
print()

# Let's try hypothesis: look at blocks after 01 <row> 00 00 00
print('Extracting column indices after each ROW_START:')
print()

pos = 27
row_blocks = []

while pos < len(section) - 5:
    if section[pos] == 0x01:
        row_idx = section[pos + 1]
        print(f'Row {row_idx} ({gdx["elements"][row_idx-1]}) at pos {pos}:')

        # Skip to after "01 <row> 00 00 00"
        pos += 5

        # Collect bytes until next 0x01 or end
        block_bytes = []
        while pos < len(section) and section[pos] != 0x01:
            block_bytes.append(section[pos])
            pos += 1

        print(f'  Block bytes: {" ".join(f"{b:02x}" for b in block_bytes)}')

        # Try to extract column indices
        # Hypothesis: pattern is <col> 00 00 00 XX or <col> XX
        cols = []
        i = 0
        while i < len(block_bytes):
            b = block_bytes[i]
            # Check if it's a valid UEL index (1-based)
            if 1 <= b <= len(gdx['elements']):
                # Check if followed by 00 00 00
                if i + 3 < len(block_bytes) and block_bytes[i+1:i+3] == [0, 0]:
                    cols.append(b)
                    print(f'    Found col {b} ({gdx["elements"][b-1]}) at offset {i}')
                    i += 4  # Skip col + 00 00 00
                    if i < len(block_bytes):
                        print(f'      Followed by: 0x{block_bytes[i]:02x}')
                        i += 1  # Skip the marker/counter byte
                else:
                    i += 1
            else:
                i += 1

        row_blocks.append((row_idx, cols))
        print()
    else:
        pos += 1

print('\n' + '='*80)
print('EXTRACTED TUPLES:')
print('='*80)
for row_idx, cols in row_blocks:
    row_name = gdx['elements'][row_idx-1]
    for col_idx in cols:
        col_name = gdx['elements'][col_idx-1]
        print(f'  ({row_idx}, {col_idx}) = ({row_name}, {col_name})')

print('\n' + '='*80)
print('COMPARISON WITH EXPECTED:')
print('='*80)
extracted_set = {(r, c) for r, cols in row_blocks for c in cols}
expected_set = set(expected)

print(f'Expected: {len(expected_set)} tuples')
print(f'Extracted: {len(extracted_set)} tuples')
print(f'Match: {extracted_set == expected_set}')

if extracted_set != expected_set:
    print(f'\nMissing: {expected_set - extracted_set}')
    print(f'Extra: {extracted_set - expected_set}')
