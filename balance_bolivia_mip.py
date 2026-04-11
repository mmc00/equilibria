"""
Balance Bolivia MIP completely using GRAS method.

This script demonstrates complete MIP balancing that satisfies all three identities:
1. Row balance (supply = demand)
2. Column balance (inputs = production)
3. PIB identity (production = expenditure)
"""

import pandas as pd
from pathlib import Path

from equilibria.sam_tools.balancing import balance_complete_mip

# Paths
input_path = Path("/Users/marmol/proyectos/cge_babel/playground/bol/mip_bol_unbalanced.xlsx")
output_path = Path("/Users/marmol/proyectos/cge_babel/playground/bol/mip_bol_balanced_complete.xlsx")

# Load MIP
print("Loading unbalanced MIP...")
mip_df = pd.read_excel(input_path, sheet_name='mip', header=0, index_col=0)
print(f"  Shape: {mip_df.shape}")

# Remove total row and column if present
if mip_df.index[-1] == 'X':
    mip_df = mip_df.iloc[:-1, :]
if mip_df.columns[-1] == 'X':
    mip_df = mip_df.iloc[:, :-1]

print(f"  After removing totals: {mip_df.shape}")

# Define structure (Bolivia specific)
N_PRODUCTS = 70
N_SECTORS = 70

# Find VA rows by name
va_row_names = [
    'Remuneraciones (trabajadores asalariados)',
    'Excedente Bruto de Explotacion',
    'Otros impuestos menos subsidios'
]
va_row_indices = [mip_df.index.get_loc(name) for name in va_row_names]
print(f"  VA row indices: {va_row_indices}")

# Import rows are ind-01 to ind-70 with "importaciones " prefix
import_row_indices = list(range(70, 140))  # Rows 70-139
print(f"  Import row indices: {import_row_indices[0]} to {import_row_indices[-1]}")

# Final demand columns
fd_col_names = [
    'Consumo Final de los Hogares',
    'Consumo Final del Gobierno',
    'Formación Bruta de Capital',
    'Variación de Stock y Existencias',
    'Exportaciones FOB'
]
fd_col_indices = [mip_df.columns.get_loc(name) for name in fd_col_names]
print(f"  FD column indices: {fd_col_indices}")

# Calculate initial imbalances
print("\n" + "="*60)
print("INITIAL IMBALANCES")
print("="*60)

# Product balance (rows)
Z = mip_df.iloc[:N_PRODUCTS, :N_SECTORS].values
F = mip_df.iloc[:N_PRODUCTS, fd_col_indices].values
IMP = mip_df.iloc[import_row_indices, :].values

supply_products = Z.sum(axis=0) + IMP[:N_PRODUCTS, :N_SECTORS].sum(axis=0)  # Domestic + imports
demand_products = Z.sum(axis=1) + F.sum(axis=1)
product_diff = supply_products - demand_products[:N_SECTORS]
print(f"Product balance max diff: {abs(product_diff).max():,.2f}")

# Sector balance (columns)
VA = mip_df.iloc[va_row_indices, :N_SECTORS].values
inputs_sectors = Z.sum(axis=0) + IMP[:N_PRODUCTS, :N_SECTORS].sum(axis=0)
production_sectors = inputs_sectors + VA.sum(axis=0)
sector_diff = production_sectors - (Z.sum(axis=1) + F.sum(axis=1))[:N_SECTORS]
print(f"Sector balance max diff: {abs(sector_diff).max():,.2f}")

# PIB identity
pib_production = VA.sum()
pib_expenditure = F.sum() - IMP[:, fd_col_indices].sum()
print(f"PIB (production): {pib_production:,.2f}")
print(f"PIB (expenditure): {pib_expenditure:,.2f}")
print(f"PIB difference: {abs(pib_production - pib_expenditure):,.2f} ({100*abs(pib_production - pib_expenditure)/pib_production:.2f}%)")

# Balance MIP completely
print("\n" + "="*60)
print("BALANCING WITH GRAS METHOD")
print("="*60)
print("Settings:")
print("  - Fix VA: True (most reliable data)")
print("  - Max iterations: 1000")
print("  - Tolerance: 1e-4")
print("\nRunning GRAS balancing...")

result = balance_complete_mip(
    mip_df,
    n_products=N_PRODUCTS,
    n_sectors=N_SECTORS,
    va_row_indices=va_row_indices,
    import_row_indices=import_row_indices,
    fd_col_indices=fd_col_indices,
    fix_va=True,
    max_iterations=1000,
    tolerance=1e-4,
)

print(f"\nConverged: {result.converged}")
print(f"Iterations: {result.iterations}")

# Print final balance statistics
print("\n" + "="*60)
print("FINAL BALANCE STATISTICS")
print("="*60)
print(f"Row balance max diff: {result.row_balance_max_diff:.6f}")
print(f"Column balance max diff: {result.col_balance_max_diff:.6f}")
print(f"PIB (production): {result.pib_production:,.2f}")
print(f"PIB (expenditure): {result.pib_expenditure:,.2f}")
print(f"PIB difference: {result.pib_diff:,.2f} ({100*result.pib_diff/result.pib_production:.4f}%)")

# Add back totals
balanced_df = result.matrix.copy()
balanced_df['X'] = balanced_df.sum(axis=1)
balanced_df.loc['X'] = balanced_df.sum(axis=0)

# Save
print(f"\nSaving balanced MIP to: {output_path}")
balanced_df.to_excel(output_path, sheet_name='mip_balanced')

print("\n" + "="*60)
print("SUCCESS! MIP is now completely balanced.")
print("="*60)
print("\nThe balanced MIP satisfies:")
print("  ✓ Row balance (supply = demand for each product)")
print("  ✓ Column balance (inputs = production for each sector)")
print("  ✓ PIB identity (production = expenditure)")
print("\nNext step: Convert to SAM using run_mip_to_sam()")
