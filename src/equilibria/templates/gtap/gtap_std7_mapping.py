"""GTAP Standard 7 Parameter Name Mapping

This module defines the mapping between GTAP Standard 7 native parameter names
(as they appear in the 9x10 database) and the internal template names.

GTAP Standard 7 uses uppercase parameter names with specific conventions:
- V*** = Value flows (monetary SAM entries)
- E*** = Endowment/factor related
- T*** = Tax/tariff related
- ***B = At Basic prices
- ***P = At Purchaser prices

Reference: /Users/marmol/proyectos2/cge_babel/standard_gtap_7/basedata-9x10.gdx
"""

from typing import Dict, Tuple

# =============================================================================
# GTAP Standard 7 Native Parameter Names (from GDX)
# =============================================================================

GTAP_STD7_BENCHMARK_PARAMS = {
    # Value flows (SAM entries)
    'VOSB': 'Sales of domestic output supply, at basic prices',  # (COMM, REG)
    'EVFP': 'Primary factor purchases, at purchasers prices',    # (ENDW, ACTS, REG)
    'EVFB': 'Primary factor basic flows',                        # (ENDW, ACTS, REG)
    'EVOS': 'Value of endowment supply net of direct tax',       # (ENDW, ACTS, REG)
    
    # Firm intermediate purchases
    'VDFP': 'Domestic purchases by firms, at purchasers prices', # (COMM, ACTS, REG)
    'VDFB': 'Domestic purchases by firms, at basic prices',      # (COMM, ACTS, REG)
    'VMFP': 'Imported purchases by firms, at purchasers prices', # (COMM, ACTS, REG)
    'VMFB': 'Imported purchases by firms, at basic prices',      # (COMM, ACTS, REG)
    
    # Government purchases
    'VDGP': 'Domestic purchases by government, at purchasers prices',  # (COMM, REG)
    'VDGB': 'Domestic purchases by government, at basic prices',       # (COMM, REG)
    'VMGP': 'Imported purchases by government, at purchasers prices',  # (COMM, REG)
    'VMGB': 'Imported purchases by government, at basic prices',       # (COMM, REG)
    
    # Private (household) purchases
    'VDPP': 'Domestic purchases by households, at purchasers prices',  # (COMM, REG)
    'VDPB': 'Domestic purchases by households, at basic prices',       # (COMM, REG)
    'VMPP': 'Imported purchases by households, at purchasers prices',  # (COMM, REG)
    'VMPB': 'Imported purchases by households, at basic prices',       # (COMM, REG)
    
    # Investment purchases
    'VDIP': 'Domestic purchases for investment, at purchasers prices', # (COMM, REG)
    'VDIB': 'Domestic purchases for investment, at basic prices',      # (COMM, REG)
    'VMIP': 'Imported purchases for investment, at purchasers prices', # (COMM, REG)
    'VMIB': 'Imported purchases for investment, at basic prices',      # (COMM, REG)
    
    # Trade flows
    'VXSB': 'Exports at basic prices',                           # (COMM, REG, REG)
    'VFOB': 'Exports FOB',                                       # (COMM, REG, REG)
    'VCIF': 'Imports CIF',                                       # (COMM, REG, REG)
    
    # Transport margins
    'VMRT': 'Bilateral trade transport margins',                # (MARG, COMM, REG, REG)
    'VTWR': 'Transport margins',                                # (MARG, COMM, REG, REG)
    'VMSB': 'Margins supply at basic prices',                   # (MARG, REG, REG)
    'VST': 'Stocks',                                             # (COMM, REG)
    
    # Make matrix (multi-output)
    'MAKB': 'Make matrix at basic prices',                      # (COMM, ACTS, REG)
    'MAKS': 'Make shares',                                      # (COMM, ACTS, REG)
    
    # Other
    'VKB': 'Capital stock',                                      # (REG)
    'VDEP': 'Capital depreciation',                             # (REG)
    'SAVE': 'Net saving by region',                             # (REG)
    'POP': 'Population',                                         # (REG)
    'DPSM': 'Sum of distribution parameters',                   # (REG)
}

GTAP_STD7_TAX_PARAMS = {
    # Tax parameters
    'TFRV': 'Ordinary import duty',                             # (COMM, REG, REG)
    'FTRV': 'Gross factor employment tax revenue',              # (ENDW, ACTS, REG)
    'FBEP': 'Gross factor-based tax/subsidy',                   # (ENDW, ACTS, REG)
    'OSEP': 'Net ordinary output tax/subsidy',                  # (COMM, REG)
    'CSEP': 'Net intermediate input tax/subsidy',               # (COMM, ACTS, REG, DIR)
    'ISEP': 'Net investment input tax/subsidy',                 # (COMM, REG, DIR)
    
    # Other tariff/tax revenues
    'ADRV': 'Anti-dumping duty',                                # (COMM, REG, REG)
    'MFRV': 'Export tax equivalent of MFA quota premia',        # (COMM, REG, REG)
    'XTRV': 'Export tax',                                       # (COMM, REG, REG)
    'VRRV': 'Revenue from regional tariffs',                    # (COMM, REG, REG)
}

GTAP_STD7_ELASTICITY_PARAMS = {
    # These are typically in default-9x10.gdx (parameters file)
    'ESUBD': 'CES elasticity domestic vs imported (top Armington)',
    'ESUBM': 'CES elasticity across import sources (bottom Armington)',
    'ESUBVA': 'CES elasticity between VA and intermediate demand',
    'ETRAE': 'CET elasticity for factor mobility',
    
    # Output transformation
    'OMEGAX': 'CET elasticity domestic sales vs exports',
    'OMEGAW': 'CET elasticity across export destinations',
    
    # Final demand
    'ESUBG': 'CES elasticity for government demand',
    'ESUBI': 'CES elasticity for investment demand',
    'ESUBC': 'CES elasticity for private consumption',
    
    # Transport margins
    'SIGMAM': 'Elasticity for transport margins',
}


# =============================================================================
# Mapping Functions
# =============================================================================

def get_benchmark_parameter_name(internal_name: str) -> str:
    """Map internal template name to GTAP Std 7 native name.
    
    Args:
        internal_name: Internal name like 'vom', 'vfm', 'vdfm'
        
    Returns:
        GTAP Std 7 name like 'VOSB', 'EVFP', 'VDFP'
    """
    # Define the mapping
    mapping = {
        # Output
        'vom': 'MAKB',  # Activity output (calculated from make matrix)
        'vosb': 'VOSB',  # Commodity output
        
        # Factors
        'vfm': 'EVFP',  # Factor payments → endowment purchases
        'evfp': 'EVFP',
        'evfb': 'EVFB',
        'evos': 'EVOS',
        
        # Firm intermediates
        'vdfm': 'VDFP',  # Domestic intermediate firm
        'vifm': 'VMFP',  # Imported intermediate firm (M=imported)
        'vdfb': 'VDFB',
        'vmfb': 'VMFB',
        
        # Government
        'vdgm': 'VDGP',
        'vigm': 'VMGP',
        'vdgb': 'VDGB',
        'vmgb': 'VMGB',
        
        # Private (households)
        'vpm': 'VDPP',  # Total private consumption (domestic)
        'vdpm': 'VDPP',
        'vipm': 'VMPP',
        'vdpb': 'VDPB',
        'vmpb': 'VMPB',
        
        # Investment
        'vim': 'VDIP',  # Total investment demand (domestic)
        'vdim': 'VDIP',
        'viim': 'VMIP',
        'vdib': 'VDIB',
        'vmib': 'VMIB',
        
        # Government consumption
        'vgm': 'VDGP',  # Total government consumption (domestic)
        
        # Trade
        'vxmd': 'VXSB',  # Exports
        'vfob': 'VFOB',
        'vcif': 'VCIF',
        'viws': 'VCIF',  # Imports CIF
        'vims': 'VCIF',
        
        # Make matrix
        'makb': 'MAKB',
        'make': 'MAKB',
        
        # Other
        'vkb': 'VKB',
        'save': 'SAVE',
        'pop': 'POP',
    }
    
    return mapping.get(internal_name.lower(), internal_name.upper())


def get_tax_parameter_name(internal_name: str) -> str:
    """Map internal tax name to GTAP Std 7 native name.
    
    Args:
        internal_name: Internal name like 'rto', 'rtf', 'rtms'
        
    Returns:
        GTAP Std 7 name like 'OSEP', 'FBEP', 'TFRV'
    """
    mapping = {
        'rto': 'OSEP',    # Output tax
        'rtf': 'FBEP',    # Factor tax
        'rtms': 'TFRV',   # Import tariff
        'rtxs': 'XTRV',   # Export tax
        'rtfd': 'FTRV',   # Factor employment tax
        'rtid': 'CSEP',   # Intermediate input tax
        'rtii': 'ISEP',   # Investment tax
    }
    
    return mapping.get(internal_name.lower(), internal_name.upper())


def get_elasticity_parameter_name(internal_name: str) -> str:
    """Map internal elasticity name to GTAP Std 7 native name.
    
    Args:
        internal_name: Internal name like 'esubva', 'esubd', 'esubm'
        
    Returns:
        GTAP Std 7 name like 'ESUBVA', 'ESUBD', 'ESUBM'
    """
    # GTAP Std 7 elasticity names are already uppercase
    return internal_name.upper()


# =============================================================================
# Parameter Index Order Mapping
# =============================================================================

# GTAP Std 7 index orders differ from template expectations
# Template uses (r, f, a) but GTAP Std 7 uses (f, a, r) for EVFP
GTAP_STD7_INDEX_REORDER = {
    'EVFP': (2, 0, 1),   # (f, a, r) → (r, f, a)
    'EVFB': (2, 0, 1),   # (f, a, r) → (r, f, a)
    'EVOS': (2, 0, 1),   # (f, a, r) → (r, f, a)
    'VDFP': (2, 0, 1),   # (i, a, r) → (r, i, a)
    'VDFB': (2, 0, 1),   # (i, a, r) → (r, i, a)
    'VMFP': (2, 0, 1),   # (i, a, r) → (r, i, a)
    'VMFB': (2, 0, 1),   # (i, a, r) → (r, i, a)
    'MAKB': (2, 1, 0),   # (i, a, r) → (r, a, i)
    'VXSB': (2, 0, 1),   # (i, r, rp) → (r, i, rp)
    'VFOB': (2, 0, 1),   # (i, r, rp) → (r, i, rp)
    'VCIF': (2, 0, 1),   # (i, rp, r) → (r, i, rp)
    'VTWR': (3, 1, 2, 0),  # (m, i, r, rp) → (r, i, rp, m) (4D reorder)
    # 2D parameters typically are (var, r) → (r, var) or already correct
    'VOSB': (1, 0),      # (i, r) → (r, i)
    'VDGP': (1, 0),      # (i, r) → (r, i)
    'VDGB': (1, 0),      # (i, r) → (r, i)
    'VMGP': (1, 0),      # (i, r) → (r, i)
    'VMGB': (1, 0),      # (i, r) → (r, i)
    'VDPP': (1, 0),      # (i, r) → (r, i)
    'VDPB': (1, 0),      # (i, r) → (r, i)
    'VMPP': (1, 0),      # (i, r) → (r, i)
    'VMPB': (1, 0),      # (i, r) → (r, i)
    'VDIP': (1, 0),      # (i, r) → (r, i)
    'VDIB': (1, 0),      # (i, r) → (r, i)
    'VMIP': (1, 0),      # (i, r) → (r, i)
    'VMIB': (1, 0),      # (i, r) → (r, i)
    # Tax parameters
    'FBEP': (2, 0, 1),   # (f, a, r) → (r, f, a)
    'FTRV': (2, 0, 1),   # (f, a, r) → (r, f, a)
    'OSEP': (1, 0),      # (i, r) → (r, i)
    'CSEP': (3, 1, 2, 0),  # (i, a, r, DIR) → (r, i, a, DIR) - need to handle DIR
    'ISEP': (2, 1, 0),   # (i, r, DIR) → (r, i, DIR)
    'TFRV': (2, 0, 1),   # (i, r, rp) → (r, i, rp)
    'XTRV': (2, 0, 1),   # (i, r, rp) → (r, i, rp)
    # Elasticity parameters
    'ESUBVA': (1, 0),    # (a, r) → (r, a)
    'ESUBD': (1, 0),     # (i, r) → (r, i)
    'ESUBM': (1, 0),     # (i, r) → (r, i)
    'ETRAE': (1, 0),     # (f, r) → (r, f)
    'ETRAQ': (1, 0),     # (a, r) → (r, a)
}


def reorder_parameter_keys(param_name: str, data: Dict) -> Dict:
    """Reorder parameter keys from GTAP Std 7 order to template order.
    
    Args:
        param_name: GTAP Std 7 parameter name (e.g., 'EVFP', 'VDFP')
        data: Dictionary with GTAP Std 7 key order
        
    Returns:
        Dictionary with template key order
    """
    if param_name not in GTAP_STD7_INDEX_REORDER:
        return data
    
    reorder = GTAP_STD7_INDEX_REORDER[param_name]
    reordered = {}
    
    for key, value in data.items():
        if isinstance(key, tuple):
            new_key = tuple(key[i] for i in reorder)
            reordered[new_key] = value
        else:
            reordered[key] = value
    
    return reordered


# =============================================================================
# Set Name Mapping
# =============================================================================

GTAP_STD7_SET_NAMES = {
    'r': 'REG',    # Regions
    'i': 'COMM',   # Commodities
    'a': 'ACTS',   # Activities/sectors
    'f': 'ENDW',   # Endowments/factors
    'm': 'MARG',   # Margins (transport)
}


def get_set_name(internal_name: str) -> str:
    """Map internal set name to GTAP Std 7 native name.
    
    Args:
        internal_name: Internal set name like 'r', 'i', 'a', 'f'
        
    Returns:
        GTAP Std 7 set name like 'REG', 'COMM', 'ACTS', 'ENDW'
    """
    return GTAP_STD7_SET_NAMES.get(internal_name.lower(), internal_name.upper())
    
    return mapping.get(internal_name.lower(), internal_name.upper())


def get_elasticity_parameter_name(internal_name: str) -> str:
    """Map internal elasticity name to GTAP Std 7 native name.
    
    Args:
        internal_name: Internal name like 'esubva', 'esubd'
        
    Returns:
        GTAP Std 7 name like 'ESUBVA', 'ESUBD'
    """
    # GTAP Std 7 elasticities are already uppercase versions
    return internal_name.upper()


# =============================================================================
# Index Mapping
# =============================================================================

GTAP_STD7_SET_NAMES = {
    'r': 'REG',      # Regions
    'i': 'COMM',     # Commodities
    'a': 'ACTS',     # Activities
    'f': 'ENDW',     # Endowments (factors)
    'm': 'MARG',     # Margins
}


def map_index_names(internal_sets: Tuple[str, ...]) -> Tuple[str, ...]:
    """Map internal set names to GTAP Std 7 set names.
    
    Args:
        internal_sets: Tuple like ('r', 'a', 'i')
        
    Returns:
        GTAP Std 7 names like ('REG', 'ACTS', 'COMM')
    """
    return tuple(GTAP_STD7_SET_NAMES.get(s, s.upper()) for s in internal_sets)
