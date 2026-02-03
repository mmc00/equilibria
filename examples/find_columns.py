"""
New hypothesis: columns might be encoded differently.
Let's look for ALL occurrences of valid UEL indices in the data blocks.
"""

from pathlib import Path
from equilibria.babel.gdx.reader import read_gdx, read_data_sections

gdx = read_gdx('tests/fixtures/set_2d_sparse.gdx')
data = Path('tests/fixtures/set_2d_sparse.gdx').read_bytes()
sections = read_data_sections(data)
_, section = sections[2]

expected = [
    (1, 4),  # agr.food
    (1, 6),  # agr.services
    (2, 5),  # mfg.goods
    (2, 6),  # mfg.services
    (3, 6),  # srv.services
]

print('UEL: 1=agr, 2=mfg, 3=srv, 4=food, 5=goods, 6=services')
print('Expected:', expected)
print()

# New approach: look at EVERY byte in data section
# and mark which are valid column indices (>= 4 and <= 6)
print('Scanning for column index candidates (4, 5, 6):')
print('Pos  Byte  Context')
print('-' * 50)

pos = 27
current_row = None

while pos < len(section):
    b = section[pos]
    context = ''
    
    if b == 0x01:
        if pos + 1 < len(section):
            next_b = section[pos + 1]
            if 1 <= next_b <= 3:  # Row indices
                current_row = next_b
                context = f'ROW_START row={next_b}'
    
    elif b in (4, 5, 6) and current_row:  # Potential column indices
        context = f'COL_CANDIDATE row={current_row} col={b} => ({current_row},{b})'
        # Check if this matches expected
        if (current_row, b) in expected:
            context += ' ✓ EXPECTED'
        else:
            context += ' ✗ NOT EXPECTED'
    
    if context:
        print(f'{pos:3d}  0x{b:02x}  {context}')
    
    pos += 1

print()
print('='*60)
print('OBSERVATION:')
print('='*60)
print('Looking at the output above, identify which column candidates')
print('are the correct ones vs. which are noise/markers.')
