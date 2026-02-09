# GDX Parameter Delta Encoding Issue

## Problem Summary

The GDX reader fails to decode parameters with delta compression (e.g., SAM-V2_0.gdx), returning 0 values instead of the expected 196 records.

## Error Details

### What Works
- **Sets (2D, 3D, 4D)**: Fully functional with delta encoding
- **Simple parameters**: Working when not compressed
- **Test suite**: 150 tests passing, 4 skipped

### What Doesn't Work
- **Parameters with delta compression**: Returns empty dictionary
- **Example file**: `/Users/marmol/proyectos/cge_babel/pep_static_clean/data/original/SAM-V2_0.gdx`
- **Expected**: 196 records with tuples like `('AG', 'USK', 'J', 'AGR') = 1500.0`
- **Actual**: 0 records decoded

## Root Cause Analysis

### Expected Format (Current Decoder Assumption)
```
Position 11: 01 (record start marker)
Position 12: XX (dim1 index, 1-based)
Position 13-16: 00 00 00 XX (dim2 index, int32 little-endian)
Position 17-20: 00 00 00 XX (dim3 index, int32 little-endian)
...
Position N: 0A (double marker)
Position N+1: <8 bytes> (double value)
```

### Actual Format (SAM-V2_0.gdx)
```
Position 11: 01 (record start marker) ✓
Position 12: 00 (dim1 = 0, NOT 7 as expected for 'AG') ✗
Position 13-16: 00 00 00 1B (dim2 = 27, NOT 2 as expected for 'USK') ✗
Position 17-20: 00 00 00 02 (dim3 = 2, NOT 18 as expected for 'J') ✗
...
Position 48: 0A (double marker)
Position 49-56: 10002.0 (first value)
```

### Key Finding
The indices are **NOT** stored as absolute values. They use a **different delta compression scheme** than sets.

## Binary Analysis

### Header Structure (11 bytes)
```
Bytes 0-5:   5F 44 41 54 41 5F  (_DATA_ marker)
Byte 6:      04                  (dimension = 4)
Bytes 7-10:  C4 00 00 00         (record count = 196, little-endian)
```

### First Record Data (bytes 11-48)
```
11: 01 - Record start marker
12: 00 - Compressed dim1 (should be 7 for 'AG', but is 0)
13-16: 00 00 00 1B - Compressed dim2 (should be 2 for 'USK', but is 27)
17-20: 00 00 00 02 - Compressed dim3 (should be 18 for 'J', but is 2)
21-24: 00 00 00 1E - Compressed dim4 (should be 19 for 'AGR', but is 30)
...
48: 0A - Double marker
49-56: 00 00 00 00 00 70 97 40 - Value 10002.0 (double, little-endian)
```

### Pattern Observation
The indices appear to use a **delta encoding scheme** where:
- Initial record stores base indices
- Subsequent records store deltas (differences from previous)
- The pattern is NOT the same as set delta encoding

## Investigation Required

### 1. Delta Encoding Algorithm
**Question**: What is the exact delta compression algorithm for parameters?

**Investigation Steps**:
1. Compare multiple consecutive records in binary
2. Identify how indices change from record to record
3. Determine delta byte interpretation (similar to sets: delta_byte + 2 = new_index?)
4. Check for special markers (0x05, 0x06, 0x0A) and their meanings

### 2. Index Mapping
**Question**: How are the compressed indices mapped to actual UEL indices?

**Investigation Steps**:
1. Map each byte pattern to expected indices using CSV reference
2. Identify if indices are stored as:
   - Absolute values (unlikely based on evidence)
   - Delta from previous (likely)
   - Bit-packed or compressed (possible)

### 3. Record Structure
**Question**: What is the exact byte layout for parameter records?

**Investigation Steps**:
1. Measure distance between consecutive 0x0A markers
2. Identify variable-length vs fixed-length records
3. Check for continuation patterns (like sets: 0x05/0x06 markers)

## Recommended Solution

### Option 1: Reverse-Engineer Parameter Format (Recommended)
**Effort**: 4-8 hours
**Approach**:
1. Create a detailed mapping of 10-20 records from CSV to binary
2. Identify the delta encoding pattern
3. Implement a separate `_decode_parameter_delta()` function
4. Test against full CSV (196 records)

### Option 2: Use GAMS API
**Effort**: 2-4 hours
**Approach**:
1. Use GAMS Python API to read GDX files
2. Extract parameter values using official library
3. Cache results for performance

### Option 3: Document Limitation
**Effort**: 30 minutes
**Approach**:
1. Add clear documentation about parameter limitation
2. Raise NotImplementedError for compressed parameters
3. Recommend using GAMS API for complex parameters

## Files for Investigation

- **GDX file**: `/Users/marmol/proyectos/cge_babel/pep_static_clean/data/original/SAM-V2_0.gdx`
- **Reference CSV**: `/Users/marmol/proyectos/cge_babel/pep_static_clean/data/original/gdx_values.csv`
- **Decoder**: `/Users/marmol/proyectos/equilibria/src/equilibria/babel/gdx/reader.py`
  - Function: `_decode_simple_parameter()` (line ~1135)
  - Related: `_decode_set_section()` for comparison

## Current Status

- **Sets**: ✅ Fully working (2D, 3D, 4D)
- **Simple parameters**: ✅ Working
- **Compressed parameters**: ❌ Not working (returns 0 values)
- **Test suite**: 150/154 tests passing (96% success rate)

## Next Steps

1. **Immediate**: Document limitation in code and README
2. **Short-term**: Create GitHub issue with this analysis
3. **Medium-term**: Implement parameter delta decoder (Option 1)
4. **Long-term**: Consider GAMS API integration (Option 2)

## References

- GDX format documentation (limited)
- GAMS API documentation
- SAM-V2_0.gdx binary analysis (this document)
- CSV reference: gdx_values.csv (196 records)
