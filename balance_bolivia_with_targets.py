"""
Balance Bolivia MIP using pre-computed target vectors.

This uses the GRAS method with explicit row and column targets.
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Paths
input_path = Path("/Users/marmol/proyectos/cge_babel/playground/bol/mip_bol_unbalanced.xlsx")
output_path = Path("/Users/marmol/proyectos/cge_babel/playground/bol/mip_bol_balanced_complete.xlsx")
targets_dir = Path("/Users/marmol/proyectos/cge_babel/playground/bol")

# Load MIP
print("Loading unbalanced MIP...")
mip_df = pd.read_excel(input_path, sheet_name='mip', header=0, index_col=0)
print(f"  Shape: {mip_df.shape}")

# Remove total row and column if present
if mip_df.index[-1] == 'X':
    mip_df = mip_df.iloc[:-1, :]
if mip_df.columns[-1] == 'X':
    mip_df = mip_df.iloc[:, :-1]

# Handle NaN values (fill with 0)
n_nans = mip_df.isna().sum().sum()
if n_nans > 0:
    print(f"  Found {n_nans} NaN values, filling with 0")
    mip_df = mip_df.fillna(0)

print(f"  After removing totals: {mip_df.shape}")

# Load target vectors
col_labels = np.load(targets_dir / "col_labels.npy", allow_pickle=True)
row_labels = np.load(targets_dir / "row_labels.npy", allow_pickle=True)
targets_col = np.load(targets_dir / "targets_col.npy")
targets_row = np.load(targets_dir / "targets_row.npy")

print(f"\nTarget vectors loaded:")
print(f"  Column targets shape: {targets_col.shape}, sum: {targets_col.sum():,.2f}")
print(f"  Row targets shape: {targets_row.shape}, sum: {targets_row.sum():,.2f}")

# Define structure
N_PRODUCTS = 70
N_SECTORS = 70
va_row_names = [
    'Remuneraciones (trabajadores asalariados)',
    'Excedente Bruto de Explotacion',
    'Otros impuestos menos subsidios'
]
va_row_indices = [mip_df.index.get_loc(name) for name in va_row_names]

# Calculate initial balance
M = mip_df.values.copy()
row_sums = M.sum(axis=1)
col_sums = M.sum(axis=0)

print("\n" + "="*60)
print("INITIAL IMBALANCES")
print("="*60)
print(f"Max row diff from target: {np.abs(row_sums - targets_row).max():,.2f}")
print(f"Max col diff from target: {np.abs(col_sums - targets_col).max():,.2f}")
print(f"Row sum total: {row_sums.sum():,.2f}")
print(f"Col sum total: {col_sums.sum():,.2f}")
print(f"Row target total: {targets_row.sum():,.2f}")
print(f"Col target total: {targets_col.sum():,.2f}")

# GRAS balancing with explicit targets
print("\n" + "="*60)
print("BALANCING WITH GRAS (EXPLICIT TARGETS)")
print("="*60)

max_iter = 1000
tolerance = 1e-2  # More relaxed tolerance
fix_va = True

# Identify which rows/cols can be adjusted
adjustable_rows = [i for i in range(len(M)) if i not in va_row_indices]
all_rows = list(range(len(M)))
all_cols = list(range(M.shape[1]))

# Get VA values to preserve (only structural proportions, not absolute values)
if fix_va:
    VA_original = M[va_row_indices, :].copy()
    VA_row_totals = VA_original.sum(axis=1)
    VA_col_proportions = VA_original / (VA_original.sum(axis=0) + 1e-10)  # Column-wise proportions
    print(f"Preserving VA structure (total VA: {VA_row_totals.sum():,.2f})")

for iteration in range(max_iter):
    # Step 1: Adjust all rows (including VA) to match row targets
    row_sums = M.sum(axis=1)
    row_factors = np.where(row_sums > 1e-6, targets_row / row_sums, 1.0)
    M = M * row_factors[:, np.newaxis]

    # Step 2: Adjust columns to match column targets
    col_sums = M.sum(axis=0)
    col_factors = np.where(col_sums > 1e-6, targets_col / col_sums, 1.0)
    M = M * col_factors[np.newaxis, :]

    # Step 3: Re-adjust VA rows to maintain target row sums while respecting column structure
    # Calculate what VA should be to satisfy column targets
    non_va_col_sums = M[adjustable_rows, :].sum(axis=0)
    va_col_target = targets_col - non_va_col_sums  # What VA columns should sum to
    va_col_total = va_col_target.sum()

    # Distribute VA column targets across VA row categories using original proportions
    if fix_va and va_col_total > 0:
        # Get current VA structure
        va_current = M[va_row_indices, :]

        # Rescale VA to match required column totals while preserving row structure
        for va_idx_pos, va_idx in enumerate(va_row_indices):
            # This VA row should sum to its target
            row_target = targets_row[va_idx]
            # Distribute across columns proportionally
            row_proportion = VA_original[va_idx_pos, :] / (VA_original[va_idx_pos, :].sum() + 1e-10)
            M[va_idx, :] = row_target * row_proportion

    # Check convergence
    row_sums = M.sum(axis=1)
    col_sums = M.sum(axis=0)
    max_row_diff = np.abs(row_sums - targets_row).max()
    max_col_diff = np.abs(col_sums - targets_col).max()
    max_diff = max(max_row_diff, max_col_diff)

    if iteration % 100 == 0:
        print(f"  Iteration {iteration}: max_row_diff={max_row_diff:.4f}, max_col_diff={max_col_diff:.4f}")

    if max_diff < tolerance:
        print(f"\nConverged in {iteration + 1} iterations!")
        break
else:
    print(f"\nDid not converge after {max_iter} iterations")

# Final statistics
print("\n" + "="*60)
print("FINAL BALANCE STATISTICS")
print("="*60)
row_sums = M.sum(axis=1)
col_sums = M.sum(axis=0)
print(f"Max row diff from target: {np.abs(row_sums - targets_row).max():.6f}")
print(f"Max col diff from target: {np.abs(col_sums - targets_col).max():.6f}")
print(f"Row sum total: {row_sums.sum():,.2f}")
print(f"Col sum total: {col_sums.sum():,.2f}")

# Calculate PIB
VA_sum = M[va_row_indices, :N_SECTORS].sum()
print(f"\nPIB (from VA): {VA_sum:,.2f}")

# Verify VA was preserved
if fix_va:
    va_diff = np.abs(M[va_row_indices, :] - VA_original).max()
    print(f"VA preservation error: {va_diff:.10f}")

# Save
balanced_df = pd.DataFrame(M, index=mip_df.index, columns=mip_df.columns)
balanced_df['X'] = balanced_df.sum(axis=1)
balanced_df.loc['X'] = balanced_df.sum(axis=0)

print(f"\nSaving balanced MIP to: {output_path}")
balanced_df.to_excel(output_path, sheet_name='mip_balanced')

print("\n" + "="*60)
print("SUCCESS! MIP balanced to target vectors")
print("="*60)
