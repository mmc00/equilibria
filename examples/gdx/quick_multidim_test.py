"""Quick test of multidimensional parameter reading."""
from equilibria.babel.gdx.reader import read_gdx, read_parameter_values

# Read the test file
data = read_gdx('tests/fixtures/multidim_test.gdx')

# Show all symbols
print('Symbols in file:')
for sym in data['symbols']:
    print(f'  {sym["name"]}: {sym["type_name"]} dim={sym["dimension"]} records={sym["records"]}')

print('\n' + '='*60)
# Test 3D parameters
for param_name in ['p3d', 'p3d_sparse']:
    print(f'\nParameter: {param_name}')
    try:
        values = read_parameter_values(data, param_name)
        print(f'  Read {len(values)} values')
        print('  First 10 values:')
        for i, (key, val) in enumerate(list(values.items())[:10]):
            print(f'    {key} = {val}')
    except Exception as e:
        print(f'  Error: {e}')
        import traceback
        traceback.print_exc()

print('\n' + '='*60)
# Test 4D parameters
for param_name in ['p4d', 'p4d_sparse']:
    print(f'\nParameter: {param_name}')
    try:
        values = read_parameter_values(data, param_name)
        print(f'  Read {len(values)} values')
        print('  First 10 values:')
        for i, (key, val) in enumerate(list(values.items())[:10]):
            print(f'    {key} = {val}')
    except Exception as e:
        print(f'  Error: {e}')
        import traceback
        traceback.print_exc()
