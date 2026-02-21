"""
PEP Income and Savings Calibration (GAMS Section 4.1)

This module implements the calibration of income and savings variables
following the exact order and formulas from GAMS PEP-1-1_v2_1_modular.gms
lines 427-469.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from pydantic import BaseModel, Field


class IncomeCalibrationResult(BaseModel):
    """Container for all calibrated income and savings variables.
    
    Matches GAMS parameters from Section 4.1 (lines 427-469).
    """
    
    # Household incomes (lines 429-434)
    YHKO: dict[str, float] = Field(default_factory=dict, description="Capital income by household")
    YHLO: dict[str, float] = Field(default_factory=dict, description="Labor income by household")
    YHTRO: dict[str, float] = Field(default_factory=dict, description="Transfer income by household")
    YHO: dict[str, float] = Field(default_factory=dict, description="Total income by household")
    YDHO: dict[str, float] = Field(default_factory=dict, description="Disposable income by household")
    CTHO: dict[str, float] = Field(default_factory=dict, description="Total consumption by household")
    TDHO: dict[str, float] = Field(default_factory=dict, description="Direct taxes by household")
    SHO: dict[str, float] = Field(default_factory=dict, description="Savings by household")
    
    # Firm incomes (lines 436-439)
    YFKO: dict[str, float] = Field(default_factory=dict, description="Capital income by firm")
    YFTRO: dict[str, float] = Field(default_factory=dict, description="Transfer income by firm")
    YFO: dict[str, float] = Field(default_factory=dict, description="Total income by firm")
    YDFO: dict[str, float] = Field(default_factory=dict, description="Disposable income by firm")
    TDFO: dict[str, float] = Field(default_factory=dict, description="Direct taxes by firm")
    SFO: dict[str, float] = Field(default_factory=dict, description="Savings by firm")
    
    # Government income (lines 441-453)
    YGKO: float = Field(default=0.0, description="Government capital income")
    TDHTO: float = Field(default=0.0, description="Total household direct taxes")
    TDFTO: float = Field(default=0.0, description="Total firm direct taxes")
    TICTO: float = Field(default=0.0, description="Total indirect commodity taxes")
    TIMTO: float = Field(default=0.0, description="Total import taxes")
    TIXTO: float = Field(default=0.0, description="Total export taxes")
    TIWTO: float = Field(default=0.0, description="Total labor taxes")
    TIKTO: float = Field(default=0.0, description="Total capital taxes")
    TIWO: dict[tuple[str, str], float] = Field(default_factory=dict, description="Labor tax by category and sector")
    TIKO: dict[tuple[str, str], float] = Field(default_factory=dict, description="Capital tax by category and sector")
    TIPTO: float = Field(default=0.0, description="Total production taxes")
    TPRODNO: float = Field(default=0.0, description="Total production taxes")
    TPRCTSO: float = Field(default=0.0, description="Total commodity taxes")
    YGTRO: float = Field(default=0.0, description="Government transfer income")
    YGO: float = Field(default=0.0, description="Total government income")
    
    # Rest of world (lines 455-457)
    YROWO: float = Field(default=0.0, description="Rest of world income")
    CABO: float = Field(default=0.0, description="Current account balance")
    SROWO: float = Field(default=0.0, description="Rest of world savings")

    # Investment (line 459)
    ITO: float = Field(default=0.0, description="Total investment")
    SGO: float = Field(default=0.0, description="Government savings")

    # Transfer matrix from SAM
    TRO: dict[tuple[str, str], float] = Field(
        default_factory=dict, description="Transfers matrix TRO(ag,agj)"
    )
    
    # Shares and parameters (lines 461-468)
    lambda_RK: dict[tuple[str, str], float] = Field(default_factory=dict, description="Capital income shares")
    lambda_WL: dict[tuple[str, str], float] = Field(default_factory=dict, description="Labor income shares")
    lambda_TR_households: dict[tuple[str, str], float] = Field(default_factory=dict, description="Transfer shares to households")
    lambda_TR_firms: dict[tuple[str, str], float] = Field(default_factory=dict, description="Transfer shares to firms")
    sh1O: dict[str, float] = Field(default_factory=dict, description="Marginal savings propensity")
    tr1O: dict[str, float] = Field(default_factory=dict, description="Marginal transfer rate")


class IncomeCalibrator:
    """Calibrates income and savings variables following GAMS Section 4.1.
    
    Implements lines 427-469 from PEP-1-1_v2_1_modular.gms in exact order.
    """
    
    def __init__(
        self,
        sam_data: dict[str, Any],
        val_par_data: dict[str, Any] | None = None,
        sets: dict[str, list[str]] | None = None,
    ):
        """Initialize the income calibrator.
        
        Args:
            sam_data: Dictionary with SAM data from read_gdx()
            val_par_data: Optional dictionary with VAL_PAR data
            sets: Dictionary with set definitions (H, F, K, L, J, I, AG, AGNG, AGD)
        """
        self.sam = sam_data
        self.val_par = val_par_data or {}
        self.sets = sets or self._detect_sets()
        self.result = IncomeCalibrationResult()
        
    def _detect_sets(self) -> dict[str, list[str]]:
        """Auto-detect sets from SAM data if not provided."""
        # Default PEP sets based on actual SAM structure
        # These match the actual elements found in SAM-V2_0.gdx
        return {
            'H': ['hrp', 'hup', 'hrr', 'hur'],
            'F': ['firm'],
            'K': ['cap', 'land'],
            'L': ['usk', 'sk'],
            'J': ['agr', 'ind', 'ser', 'adm'],  # Actual sectors in SAM
            'I': ['agr', 'food', 'othind', 'ser', 'adm'],  # Commodities from SAM
            'AG': ['hrp', 'hup', 'hrr', 'hur', 'firm', 'gvt', 'row'],  # All agents
            'AGNG': ['hrp', 'hup', 'hrr', 'hur', 'firm', 'row'],  # Non-government agents
            'AGD': ['hrp', 'hup', 'hrr', 'hur', 'firm', 'gvt'],  # Domestic agents
        }
    
    def _get_sam_value(self, param_name: str, *indices) -> float:
        """Get value from SAM data.
        
        For the PEP model, most values come from the SAM parameter directly.
        The SAM is a 4D parameter: SAM(row_dim1, row_dim2, col_dim1, col_dim2)
        
        Args:
            param_name: Name of the parameter (e.g., 'SAM', or specific like 'lambda_RK')
            *indices: Indices for the parameter
            
        Returns:
            float: The value from SAM, or 0.0 if not found
        """
        try:
            # If it's the SAM parameter, read directly from decoded values
            if param_name == 'SAM':
                return self._get_sam_matrix_value(*indices)
            
            # Otherwise, try to find a specific parameter
            for sym in self.sam.get('symbols', []):
                if sym.get('name') == param_name:
                    records = sym.get('records', [])
                    
                    # If no indices requested, return sum of all values
                    if len(indices) == 0:
                        return sum(float(r.get('value', 0)) for r in records)
                    
                    # Find matching record by indices
                    for record in records:
                        record_indices = tuple(record.get('indices', []))
                        if record_indices == indices:
                            return float(record.get('value', 0))
                    
                    return 0.0
            
            # If parameter not found, try to get from SAM matrix
            return self._get_sam_matrix_value(*indices)
            
        except Exception as e:
            print(f"Warning: Error getting {param_name}{indices}: {e}")
            return 0.0
    
    def _get_sam_matrix_value(self, *indices) -> float:
        """Get value from SAM matrix using 4D indices.
        
        The SAM matrix is indexed as: SAM(row_cat, row_elem, col_cat, col_elem)
        
        Args:
            *indices: 2 to 4 indices depending on the query
            
        Returns:
            float: Value from SAM or 0.0 if not found
        """
        try:
            indices_upper = tuple(str(idx).upper() for idx in indices)

            # Fast path for Excel-loaded SAM representation.
            sam_matrix = self.sam.get("sam_matrix")
            if isinstance(sam_matrix, dict):
                if len(indices_upper) == 4:
                    return float(sam_matrix.get(indices_upper, 0.0))
                total = 0.0
                for key, value in sam_matrix.items():
                    if all(i >= len(indices_upper) or key[i] == indices_upper[i] for i in range(len(indices_upper))):
                        total += float(value)
                return total

            # Fallback path for GDX-loaded SAM.
            from equilibria.babel.gdx.decoder import decode_parameter_delta
            from equilibria.babel.gdx.reader import read_data_sections

            sam_path = Path(self.sam.get("filepath", ""))
            raw_data = sam_path.read_bytes()
            data_sections = read_data_sections(raw_data)
            _, section = data_sections[0]
            elements = self.sam.get("elements", [])
            values = decode_parameter_delta(section, elements, 4)

            if len(indices_upper) == 4:
                return values.get(indices_upper, 0.0)

            total = 0.0
            for key, value in values.items():
                match = True
                for i, idx in enumerate(indices_upper):
                    if i < len(key) and key[i] != idx:
                        match = False
                        break
                if match:
                    total += value
            return total

        except Exception as e:
            print(f"Warning: Error reading SAM matrix: {e}")
            return 0.0
    
    def calibrate(self) -> IncomeCalibrationResult:
        """Run full income calibration (GAMS Section 4.1).
        
        Returns:
            IncomeCalibrationResult with all calibrated variables
        """
        # Preload transfer matrix TRO(ag,agj)
        self._calibrate_transfer_matrix()

        # Step 1: Household incomes (lines 429-434)
        self._calibrate_household_incomes()
        
        # Step 2: Firm incomes (lines 436-439)
        self._calibrate_firm_incomes()
        
        # Step 3: Government income (lines 441-453)
        self._calibrate_government_income()
        
        # Step 4: Rest of world (lines 455-457)
        self._calibrate_rest_of_world()
        
        # Step 5: Investment (line 459)
        self._calibrate_investment()
        
        # Step 6: Shares and parameters (lines 461-468)
        self._calibrate_shares()
        
        return self.result

    def _calibrate_transfer_matrix(self) -> None:
        """Read TRO(ag,agj)=SAM('AG',ag,'AG',agj) from SAM."""
        AG = self.sets["AG"]
        for ag in AG:
            for agj in AG:
                self.result.TRO[(ag, agj)] = self._get_sam_value(
                    "SAM", "AG", ag.upper(), "AG", agj.upper()
                )
    
    def _calibrate_household_incomes(self) -> None:
        """Calibrate household income variables (lines 429-434).
        
        GAMS formulas:
            YHKO(h) = SUM[k, lambda_RK(h,k)]
            YHLO(h) = SUM[l, lambda_WL(h,l)]
            YHTRO(h) = SUM[ag, TRO(h,ag)]
            YHO(h) = YHLO(h) + YHKO(h) + YHTRO(h)
            YDHO(h) = YHO(h) - TDHO(h) - TRO('gvt',h)
            CTHO(h) = YDHO(h) - SHO(h) - SUM[agng, TRO(agng,h)]
        
        Note: From data_loading.inc lines 88-89:
            lambda_RK(ag,k) = SAM('Agents',ag,'K',k)
            lambda_WL(h,l)  = SAM('Agents',h,'L',l)
            
        In our GDX structure:
            lambda_RK(ag, k) = SAM('AG', ag, 'K', k)
            lambda_WL(h, l) = SAM('AG', h, 'L', l)
            TRO(h, ag) = SAM('AG', h, 'AG', ag)  # Transfers from ag to h
        """
        H = self.sets['H']
        K = self.sets['K']
        L = self.sets['L']
        AG = self.sets['AG']
        AGNG = self.sets['AGNG']
        
        for h in H:
            # YHKO(h) = sum of capital income to household h
            # lambda_RK(h, k) = SAM('AG', h, 'K', k)
            yhko = sum(
                self._get_sam_value('SAM', 'AG', h, 'K', k)
                for k in K
            )
            self.result.YHKO[h] = yhko
            
            # YHLO(h) = sum of labor income to household h
            # lambda_WL(h, l) = SAM('AG', h, 'L', l)
            yhlo = sum(
                self._get_sam_value('SAM', 'AG', h, 'L', l)
                for l in L
            )
            self.result.YHLO[h] = yhlo
            
            # YHTRO(h) = sum of transfers to household h
            # TRO(h, ag) = transfers from ag to h = SAM('AG', h, 'AG', ag)
            yhtro = sum(
                self._get_sam_value('SAM', 'AG', h, 'AG', ag)
                for ag in AG 
            )
            self.result.YHTRO[h] = yhtro
            
            # YHO(h) = total income
            yho = yhlo + yhko + yhtro
            self.result.YHO[h] = yho
            
            # YDHO(h) = disposable income = total income - taxes
            # TDHO(h) = direct taxes paid by h = SAM('Agents','TAX','Agents',h)
            # In our GDX: SAM('AG', 'TD', 'AG', h)
            tdho = self._get_sam_value('SAM', 'AG', 'TD', 'AG', h)
            self.result.TDHO[h] = tdho
            # TRO('gvt',h): transfer from household h to government
            tr_h_to_gvt = self._get_sam_value('SAM', 'AG', 'GVT', 'AG', h)
            ydho = yho - tdho - tr_h_to_gvt
            self.result.YDHO[h] = ydho
            
            # CTHO(h) = total consumption = disposable income - savings - transfers to others
            # SHO(h) = savings of h = SAM('OTH','INV','Agents',h)
            # In our GDX: SAM('OTH', 'INV', 'AG', h)
            sho = self._get_sam_value('SAM', 'OTH', 'INV', 'AG', h)
            self.result.SHO[h] = sho
            # Transfers from h to other agents (outflows)
            tro_h_out = sum(
                self._get_sam_value('SAM', 'AG', ag, 'AG', h)
                for ag in AGNG
            )
            ctho = ydho - sho - tro_h_out
            self.result.CTHO[h] = ctho
    
    def _calibrate_firm_incomes(self) -> None:
        """Calibrate firm income variables (lines 436-439).
        
        GAMS formulas:
            YFKO(f) = SUM[k, lambda_RK(f,k)]
            YFTRO(f) = SUM[ag, TRO(f,ag)]
            YFO(f) = YFKO(f) + YFTRO(f)
            YDFO(f) = YFO(f) - TDFO(f)
            
        Note: lambda_RK(f,k) = SAM('AG', f, 'K', k)
              TRO(f, ag) = SAM('AG', f, 'AG', ag)  # Transfers from ag to f
              TDFO(f) = SAM('AG', 'TD', 'AG', f)  # Direct taxes paid by f
        """
        F = self.sets['F']
        K = self.sets['K']
        AG = self.sets['AG']
        
        for f in F:
            f_upper = f.upper()
            
            # YFKO(f) = sum of capital payments to firm f
            # lambda_RK(f, k) = SAM('AG', f, 'K', k)
            yfko = sum(
                self._get_sam_value('SAM', 'AG', f_upper, 'K', k.upper())
                for k in K
            )
            self.result.YFKO[f] = yfko
            
            # YFTRO(f) = sum of transfers to firm f
            # TRO(f, ag) = transfers from ag to f = SAM('AG', f, 'AG', ag)
            yftro = sum(
                self._get_sam_value('SAM', 'AG', f_upper, 'AG', ag.upper())
                for ag in AG
            )
            self.result.YFTRO[f] = yftro
            
            # YFO(f) = total income
            yfo = yfko + yftro
            self.result.YFO[f] = yfo
            
            # YDFO(f) = disposable income = total income - taxes
            # TDFO(f) = direct taxes paid by f = SAM('AG', 'TD', 'AG', f)
            tdfo = self._get_sam_value('SAM', 'AG', 'TD', 'AG', f_upper)
            self.result.TDFO[f] = tdfo
            ydfo = yfo - tdfo
            self.result.YDFO[f] = ydfo
    
    def _calibrate_government_income(self) -> None:
        """Calibrate government income variables (lines 441-453).
        
        GAMS formulas:
            YGKO = SUM[k, lambda_RK('gvt',k)]
            TDHTO = SUM[h, TDHO(h)]
            TDFTO = SUM[f, TDFO(f)]
            TICTO = SUM[i, TICO(i)]
            TIMTO = SUM[i, TIMO(i)]
            TIXTO = SUM[i, TIXO(i)]
            TIWTO = SUM[(l,j), TIWO(l,j)]
            TIKTO = SUM[(k,j), TIKO(k,j)]
            TIPTO = SUM[j, TIPO(j)]
            TPRODNO = TIKTO + TIWTO + TIPTO
            TPRCTSO = TICTO + TIMTO + TIXTO
            YGTRO = SUM[ag, TRO('gvt',ag)]
            YGO = YGKO + TDHTO + TDFTO + TPRODNO + TPRCTSO + YGTRO
            
        Note: lambda_RK('gvt', k) = SAM('AG', 'GVT', 'K', k)
              TDHO(h) = SAM('AG', 'TD', 'AG', h)
              TDFO(f) = SAM('AG', 'TD', 'AG', f)
              TICO(i) = SAM('TI', i, 'I', i)  # Indirect tax on commodity i
              TIMO(i) = SAM('TM', i, 'I', i)  # Import tax on commodity i
              TIXO(i) = SAM('X', 'TAX', 'I', i)  # Export tax on commodity i
              TIWO(l,j) = SAM('L', l, 'J', j) * tax_rate  # Labor tax
              TIKO(k,j) = SAM('K', k, 'J', j) * tax_rate  # Capital tax
              TIPO(j) = SAM('J', j, 'J', j) - sum of inputs  # Production tax
              TRO('gvt', ag) = SAM('AG', 'GVT', 'AG', ag)  # Transfers from gvt
        """
        K = self.sets['K']
        H = self.sets['H']
        F = self.sets['F']
        I = self.sets['I']
        J = self.sets['J']
        L = self.sets['L']
        AG = self.sets['AG']
        
        # YGKO = government capital income
        # lambda_RK('gvt', k) = SAM('AG', 'GVT', 'K', k)
        ygko = sum(
            self._get_sam_value('SAM', 'AG', 'GVT', 'K', k.upper())
            for k in K
        )
        self.result.YGKO = ygko
        
        # TDHTO = total household direct taxes
        # TDHO(h) = SAM('AG', 'TD', 'AG', h)
        tdhto = sum(
            self._get_sam_value('SAM', 'AG', 'TD', 'AG', h.upper())
            for h in H
        )
        self.result.TDHTO = tdhto
        
        # TDFTO = total firm direct taxes
        # TDFO(f) = SAM('AG', 'TD', 'AG', f)
        tdfto = sum(
            self._get_sam_value('SAM', 'AG', 'TD', 'AG', f.upper())
            for f in F
        )
        self.result.TDFTO = tdfto
        
        # TICTO = total indirect commodity taxes
        # TICO(i) = SAM('Agents','ti','I',i) = SAM('AG', 'TI', 'I', i)
        ticto = sum(
            self._get_sam_value('SAM', 'AG', 'TI', 'I', i.upper())
            for i in I
        )
        self.result.TICTO = ticto
        
        # TIMTO = total import taxes
        # TIMO(i) = SAM('Agents','tm','I',i) = SAM('AG', 'TM', 'I', i)
        timto = sum(
            self._get_sam_value('SAM', 'AG', 'TM', 'I', i.upper())
            for i in I
        )
        self.result.TIMTO = timto
        
        # TIXTO = total export taxes
        # TIXO(i) = SAM('AG', 'GVT', 'X', i)
        tixto = sum(
            self._get_sam_value('SAM', 'AG', 'GVT', 'X', i.upper())
            for i in I
        )
        self.result.TIXTO = tixto
        
        # TIWTO = total labor taxes
        # TIWO(l,j) = labor tax payments
        # In SAM: payments from sectors J to labor L
        tiwto = 0
        for l in L:
            for j in J:
                tiwo = self._get_sam_value('L', l.upper(), 'J', j.upper())
                if tiwo != 0:
                    self.result.TIWO[(l, j)] = tiwo
                    tiwto += tiwo
        self.result.TIWTO = tiwto
        
        # TIKTO = total capital taxes
        # TIKO(k,j) = capital tax payments
        # In SAM: payments from sectors J to capital K
        tikto = 0
        for k in K:
            for j in J:
                tiko = self._get_sam_value('K', k.upper(), 'J', j.upper())
                if tiko != 0:
                    self.result.TIKO[(k, j)] = tiko
                    tikto += tiko
        self.result.TIKTO = tikto
        
        # TIPTO = total production taxes
        # TIPO(j) = SAM('Agents','gvt','J',j) = SAM('AG', 'GVT', 'J', j)
        tipto = sum(
            self._get_sam_value('SAM', 'AG', 'GVT', 'J', j.upper())
            for j in J
        )
        self.result.TIPTO = tipto
        
        # TPRODNO = total production taxes
        tprodno = tikto + tiwto + tipto
        self.result.TPRODNO = tprodno
        
        # TPRCTSO = total commodity taxes
        tprctso = ticto + timto + tixto
        self.result.TPRCTSO = tprctso
        
        # YGTRO = government transfer income
        # GAMS pep2 dynamic scripts calibrate this over AG:
        # YGTRO = SUM[ag, TRO('gvt', ag)].
        ygtro = sum(
            self._get_sam_value('SAM', 'AG', 'GVT', 'AG', ag.upper())
            for ag in AG
        )
        self.result.YGTRO = ygtro
        
        # YGO = total government income
        ygo = ygko + tdhto + tdfto + tprodno + tprctso + ygtro
        self.result.YGO = ygo
    
    def _calibrate_rest_of_world(self) -> None:
        """Calibrate rest of world variables (lines 455-457).
        
        GAMS formulas:
            YROWO = SUM[i, IMO(i)] + SUM[k, lambda_RK('row',k)] + SUM[ag, TRO('row',ag)]
            CABO = -SROWO
            
        Note: IMO(i) = SAM('Agents','row','I',i) = SAM('AG', 'ROW', 'I', i)
              lambda_RK('row', k) = SAM('AG', 'ROW', 'K', k)
              SROWO = SAM('OTH','INV','Agents','row') = SAM('OTH', 'INV', 'AG', 'ROW')
        """
        I = self.sets['I']
        K = self.sets['K']
        AG = self.sets['AG']
        
        # IMO(i) = imports of product i
        imo_values = {
            i: self._get_sam_value('SAM', 'AG', 'ROW', 'I', i.upper())
            for i in I
        }
        
        # YROWO = rest of world income
        # = sum of imports + capital income from row + transfers from row
        yrowo = (
            sum(imo_values.values())
            + sum(self._get_sam_value('SAM', 'AG', 'ROW', 'K', k.upper()) for k in K)
            + sum(self._get_sam_value('SAM', 'AG', 'ROW', 'AG', ag.upper()) for ag in AG)
        )
        self.result.YROWO = yrowo
        
        # CABO = current account balance = -SROWO
        # SROWO = SAM('OTH','INV','Agents','row') = SAM('OTH', 'INV', 'AG', 'ROW')
        srowo = self._get_sam_value('SAM', 'OTH', 'INV', 'AG', 'ROW')
        self.result.SROWO = srowo
        cabo = -srowo
        self.result.CABO = cabo
    
    def _calibrate_investment(self) -> None:
        """Calibrate investment variable (line 459).
        
        GAMS formula:
            ITO = SUM[h, SHO(h)] + SUM[f, SFO(f)] + SGO + SROWO
            
        Note: SHO(h) = SAM('OTH','INV','Agents',h) = SAM('OTH', 'INV', 'AG', h)
              SFO(f) = SAM('OTH','INV','Agents',f) = SAM('OTH', 'INV', 'AG', f)
              SGO = SAM('OTH','INV','Agents','gvt') = SAM('OTH', 'INV', 'AG', 'GVT')
              SROWO = SAM('OTH','INV','Agents','row') = SAM('OTH', 'INV', 'AG', 'ROW')
        """
        H = self.sets['H']
        F = self.sets['F']
        
        # SHO(h) = household savings
        sho_sum = 0.0
        for h in H:
            sho_h = self._get_sam_value('SAM', 'OTH', 'INV', 'AG', h.upper())
            self.result.SHO[h] = sho_h
            sho_sum += sho_h

        # SFO(f) = firm savings
        sfo_sum = 0.0
        for f in F:
            sfo_f = self._get_sam_value('SAM', 'OTH', 'INV', 'AG', f.upper())
            self.result.SFO[f] = sfo_f
            sfo_sum += sfo_f

        # SGO = government savings
        sgo = self._get_sam_value('SAM', 'OTH', 'INV', 'AG', 'GVT')
        self.result.SGO = sgo

        # SROWO = rest of world savings
        srowo = self._get_sam_value('SAM', 'OTH', 'INV', 'AG', 'ROW')
        self.result.SROWO = srowo
        
        ito = sho_sum + sfo_sum + sgo + srowo
        self.result.ITO = ito
    
    def _calibrate_shares(self) -> None:
        """Calibrate shares and parameters (lines 461-468).
        
        GAMS formulas:
            lambda_RK(ag,k) = lambda_RK(ag,k) / SUM[j, KDO(k,j)]
            lambda_WL(h,l) = lambda_WL(h,l) / SUM[j, LDO(l,j)]
            lambda_TR(agng,h) = TRO(agng,h) / YDHO(h)
            lambda_TR(ag,f) = TRO(ag,f) / YDFO(f)
            sh1O(h) = [SHO(h) - sh0O(h)] / YDHO(h)
            tr1O(h) = [TRO('gvt',h) - tr0O(h)] / YHO(h)
            
        Note: lambda_RK(ag,k) raw = SAM('AG', ag, 'K', k)
              lambda_WL(h,l) raw = SAM('AG', h, 'L', l)
              KDO(k,j) = SAM('K', k, 'J', j)
              LDO(l,j) = SAM('L', l, 'J', j)
              TRO(ag, h) = SAM('AG', ag, 'AG', h)
              SHO(h) = SAM('OTH', 'INV', 'AG', h)
        """
        H = self.sets['H']
        F = self.sets['F']
        K = self.sets['K']
        L = self.sets['L']
        J = self.sets['J']
        AG = self.sets['AG']
        AGNG = self.sets['AGNG']
        
        # lambda_RK(ag,k) - capital income shares
        # Raw value from SAM: SAM('AG', ag, 'K', k)
        # Normalized by: SUM[j, KDO(k,j)] where KDO(k,j) = SAM('K', k, 'J', j)
        for ag in AG:
            ag_upper = ag.upper()
            for k in K:
                k_upper = k.upper()
                # Raw capital income
                lambda_rk_raw = self._get_sam_value('SAM', 'AG', ag_upper, 'K', k_upper)
                # Total capital demand of type k
                kdo_sum = sum(
                    self._get_sam_value('SAM', 'K', k_upper, 'J', j.upper())
                    for j in J
                )
                if kdo_sum != 0:
                    self.result.lambda_RK[(ag, k)] = lambda_rk_raw / kdo_sum
        
        # lambda_WL(h,l) - labor income shares
        # Raw value from SAM: SAM('AG', h, 'L', l)
        # Normalized by: SUM[j, LDO(l,j)] where LDO(l,j) = SAM('L', l, 'J', j)
        for h in H:
            h_upper = h.upper()
            for l in L:
                l_upper = l.upper()
                # Raw labor income
                lambda_wl_raw = self._get_sam_value('SAM', 'AG', h_upper, 'L', l_upper)
                # Total labor demand of type l
                ldo_sum = sum(
                    self._get_sam_value('SAM', 'L', l_upper, 'J', j.upper())
                    for j in J
                )
                if ldo_sum != 0:
                    self.result.lambda_WL[(h, l)] = lambda_wl_raw / ldo_sum
        
        # lambda_TR(agng,h) - transfer shares to households
        # TRO(agng, h) = transfers from agng to h = SAM('AG', agng, 'AG', h)
        for agng in AGNG:
            agng_upper = agng.upper()
            for h in H:
                h_upper = h.upper()
                tro = self._get_sam_value('SAM', 'AG', agng_upper, 'AG', h_upper)
                ydho = self.result.YDHO.get(h, 0)
                if ydho != 0:
                    self.result.lambda_TR_households[(agng, h)] = tro / ydho
        
        # lambda_TR(ag,f) - transfer shares to firms
        # TRO(ag, f) = transfers from ag to f = SAM('AG', ag, 'AG', f)
        for ag in AG:
            ag_upper = ag.upper()
            for f in F:
                f_upper = f.upper()
                tro = self._get_sam_value('SAM', 'AG', ag_upper, 'AG', f_upper)
                ydfo = self.result.YDFO.get(f, 0)
                if ydfo != 0:
                    self.result.lambda_TR_firms[(ag, f)] = tro / ydfo
        
        # sh1O(h) - marginal savings propensity
        # sh0O(h) comes from VAL_PAR parameter file
        for h in H:
            h_upper = h.upper()
            # SHO(h) = household savings = SAM('OTH', 'INV', 'AG', h)
            sho = self._get_sam_value('SAM', 'OTH', 'INV', 'AG', h_upper)
            sh0o = self.val_par.get('sh0O', {}).get(h, 0)
            ydho = self.result.YDHO.get(h, 0)
            if ydho != 0:
                self.result.sh1O[h] = (sho - sh0o) / ydho
        
        # tr1O(h) - marginal transfer rate
        # TRO('gvt', h) = transfers from government to h = SAM('AG', 'GVT', 'AG', h)
        for h in H:
            h_upper = h.upper()
            tro_gvt = self._get_sam_value('SAM', 'AG', 'GVT', 'AG', h_upper)
            tr0o = self.val_par.get('tr0O', {}).get(h, 0)
            yho = self.result.YHO.get(h, 0)
            if yho != 0:
                self.result.tr1O[h] = (tro_gvt - tr0o) / yho
