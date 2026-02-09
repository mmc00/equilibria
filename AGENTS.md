

## Appendix: GDX Comparison Tool

### Overview
A standalone comparison tool has been implemented to validate data consistency between original GAMS GDX files and equilibria's data handling.

### Location
- **Script:** `scripts/compare_gdx.py`
- **Report Output:** `reports/gdx_comparison_report.md`

### Features
- Compares Excel data (via equilibria loaders) with original GDX files
- Generates detailed Markdown reports
- Configurable tolerance (default: 0.01%)
- Identifies:
  - Record count differences
  - Missing records in either source
  - Value differences exceeding tolerance
  - Metadata mismatches

### Usage
```bash
python scripts/compare_gdx.py
```

### Output Format
The tool generates a comprehensive Markdown report including:
1. **Summary Table** - Overview of all comparisons
2. **Detailed Sections** - Per-file analysis with:
   - Record counts (Excel vs GDX)
   - Common records
   - Value differences
   - Records only in Excel
   - Records only in GDX

### Current Status
- **SAM-V2_0.gdx**: ✅ **FULLY WORKING** - 196/196 records decoded correctly (100% accuracy)
  - Implementation based on official GDX source code from `/Users/marmol/proyectos/gdx/src/gxfile.cpp`
  - Correctly reads MinElem/MaxElem from data header for each dimension
  - Implements proper DeltaForRead logic (DeltaForRead = dimension for version > 6)
  - Handles all delta codes: 0x01, 0x02, 0x03, 0x05, 0x06, 0xFF (EOF)
  - Supports dynamic element types (byte/word/integer) based on range
  - Applies MinElem offset to decode correct indices
  - **Update Feb 2026**: Complete rewrite based on official GDX implementation
    - Fixed header parsing (position 11-42 for MinElem/MaxElem)
    - Fixed delta code interpretation (B > DeltaForRead = relative change)
    - Fixed value reading (0x0A followed directly by 8-byte LE double)
    - Achieved 100% match rate with CSV ground truth
- **VAL_PAR.gdx**: Comparison implemented (simplified)

### Critical Finding: Decoder Fixed (Feb 2026)

**Issue Resolved**: All 196 records now decode correctly with 100% accuracy.

**Root Cause of Previous Issues**:
- Incorrect header parsing (didn't read MinElem/MaxElem)
- Wrong delta code interpretation (assumed bit-mask instead of DeltaForRead logic)
- Missing MinElem offset application to indices
- Incorrect value structure (assumed type byte before double)

**Solution Implemented**:
Based on official GDX source code analysis from `/Users/marmol/proyectos/gdx/src/gxfile.cpp`:

1. **Correct Header Parsing**:
   - Read dimension at position 6
   - Read record count at positions 7-10
   - Read MinElem/MaxElem for each dimension at positions 11-42

2. **Proper DeltaForRead Logic**:
   - DeltaForRead = dimension (for version > 6)
   - If B > DeltaForRead: relative change in last dimension (LastElem[last] += B - DeltaForRead)
   - If B <= DeltaForRead: B indicates first dimension that changes (1-based)

3. **Dynamic Element Types**:
   - Determine type based on range (MaxElem - MinElem + 1)
   - Range <= 255: 1 byte
   - Range <= 65535: 2 bytes
   - Otherwise: 4 bytes

4. **MinElem Offset**:
   - All indices read from file must have MinElem added: LastElem[D] = ReadIndex() + MinElem[D]

5. **Value Structure**:
   - 0x0A marker followed directly by 8-byte little-endian double
   - No type byte between marker and value

**Result**: 100% match rate with CSV ground truth (196/196 records).

### Delta Code Decoding
The GDX reader now correctly interprets GAMS delta compression codes based on official GDX source:

**DeltaForRead Logic** (for version > 6):
- DeltaForRead = dimension (4 for SAM-V2_0.gdx)

**Code Interpretation**:
- **B > DeltaForRead**: Relative change in last dimension
  - `LastElem[last] += B - DeltaForRead`
  - Code 5: +1 to last dimension
  - Code 6: +2 to last dimension
  - Code 255 (0xFF): EOF marker
- **B <= DeltaForRead**: Indicates first dimension that changes (1-based)
  - Code 1: Replace all indices (read dimensions 0-3)
  - Code 2: Replace indices from dimension 1 onwards
  - Code 3: Replace indices from dimension 2 onwards

**Index Reading**:
- Element type determined by range (MaxElem - MinElem + 1):
  - Range <= 255: 1 byte per index
  - Range <= 65535: 2 bytes per index  
  - Otherwise: 4 bytes per index
- All read indices have MinElem[D] added: `LastElem[D] = ReadValue() + MinElem[D]`

**Value Reading**:
- 0x0A marker followed directly by 8-byte little-endian double

### Validation Results

**SAM-V2_0.gdx**: ✅ **100% Accuracy Achieved**
- All 196 records decoded correctly
- 100% match rate with CSV ground truth
- Keys and values both verified correct

**Implementation verified against**:
- Official GDX source code from GAMS (C++)
- CSV ground truth file (gdx_values.csv)
- Manual inspection of decoded records

