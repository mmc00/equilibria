#!/usr/bin/env python3
"""
GTAP Parity Variables Summary

This script displays all variables compared in the GTAP parity system.
"""

from equilibria.templates.gtap.gtap_parity_pipeline import GTAPVariableSnapshot
from dataclasses import fields

print("=" * 70)
print("GTAP PARITY - VARIABLES COMPARED")
print("=" * 70)
print()

# Get all fields from the dataclass
snapshot_fields = fields(GTAPVariableSnapshot)

print(f"Total variable groups: {len(snapshot_fields)}")
print()

categories = {
    'Production': [],
    'Supply': [],
    'Prices - Production': [],
    'Prices - Supply': [],
    'Prices - Import': [],
    'Prices - Export': [],
    'Trade Flows': [],
    'Factors': [],
    'Final Demand': [],
    'Income': [],
    'Price Indices': [],
}

# Categorize variables
for field in snapshot_fields:
    name = field.name
    if name in ['xp', 'x']:
        categories['Production'].append(name)
    elif name in ['xs', 'xds']:
        categories['Supply'].append(name)
    elif name in ['px', 'pp']:
        categories['Prices - Production'].append(name)
    elif name in ['ps', 'pd', 'pa']:
        categories['Prices - Supply'].append(name)
    elif name in ['pmt', 'pmcif']:
        categories['Prices - Import'].append(name)
    elif name in ['pet', 'pe', 'pefob']:
        categories['Prices - Export'].append(name)
    elif name in ['xe', 'xet', 'xw', 'xmt']:
        categories['Trade Flows'].append(name)
    elif name in ['xf', 'xft', 'pf', 'pft']:
        categories['Factors'].append(name)
    elif name in ['xc', 'xg', 'xi']:
        categories['Final Demand'].append(name)
    elif name in ['regy', 'yc', 'yg', 'yi']:
        categories['Income'].append(name)
    elif name in ['pnum', 'pabs', 'walras']:
        categories['Price Indices'].append(name)

# Print by category
total_count = 0
for category, vars in categories.items():
    if vars:
        print(f"{category}:")
        for var in vars:
            field = next(f for f in snapshot_fields if f.name == var)
            type_str = str(field.type).replace('typing.', '').replace('<class ', '').replace('>', '')
            print(f"  - {var:12s} {type_str}")
            total_count += 1
        print()

print("=" * 70)
print(f"TOTAL: {total_count} variable groups compared")
print("=" * 70)
print()
print("Each group may contain multiple indexed variables:")
print("  - (r, a)      = region × activity")
print("  - (r, i)      = region × commodity")
print("  - (r, f, a)   = region × factor × activity")
print("  - (r, i, rp)  = region × commodity × partner region")
print("  - (r)         = region only")
print("  - scalar      = single value")
print()
print("For a typical 5-region, 5-commodity, 3-factor model:")
print("  ~28 groups × ~20 indices each = ~560 individual variables")
print("=" * 70)
