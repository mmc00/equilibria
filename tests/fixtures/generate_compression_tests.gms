$title Test GDX Compression Patterns - Arithmetic vs Geometric

* This file generates GDX files with known compression patterns
* to help reverse-engineer the GDX binary format

* ============================================================================
* Test 1: Pure Arithmetic Sequence
* ============================================================================
Set t1 "time periods arithmetic" / t1*t10 /;

Parameter arith(t1) "arithmetic sequence: 10, 20, 30, 40, ...";
arith("t1") = 10;
arith("t2") = 20;
arith("t3") = 30;
arith("t4") = 40;
arith("t5") = 50;
arith("t6") = 60;
arith("t7") = 70;
arith("t8") = 80;
arith("t9") = 90;
arith("t10") = 100;

execute_unload 'test_arithmetic.gdx', t1, arith;

* ============================================================================
* Test 2: Pure Geometric Sequence
* ============================================================================
Set t2 "time periods geometric" / t1*t10 /;

Parameter geom(t2) "geometric sequence: 1, 2, 4, 8, 16, ...";
geom("t1") = 1;
geom("t2") = 2;
geom("t3") = 4;
geom("t4") = 8;
geom("t5") = 16;
geom("t6") = 32;
geom("t7") = 64;
geom("t8") = 128;
geom("t9") = 256;
geom("t10") = 512;

execute_unload 'test_geometric.gdx', t2, geom;

* ============================================================================
* Test 3: Arithmetic with only 2 values (compressed)
* ============================================================================
Set t3 "time periods" / t1*t5 /;

Parameter arith2(t3) "arithmetic - only endpoints";
arith2("t1") = 100;
arith2("t5") = 500;

execute_unload 'test_arithmetic_compressed.gdx', t3, arith2;

* ============================================================================
* Test 4: Geometric with only 2 values (compressed)
* ============================================================================
Set t4 "time periods" / t1*t5 /;

Parameter geom2(t4) "geometric - only endpoints";
geom2("t1") = 10;
* 10 * 2^4 = 160
geom2("t5") = 160;

execute_unload 'test_geometric_compressed.gdx', t4, geom2;

* ============================================================================
* Test 5: Mixed pattern (neither arithmetic nor geometric)
* ============================================================================
Set t5 "time periods" / t1*t10 /;

Parameter mixed(t5) "mixed pattern";
mixed("t1") = 1;
mixed("t2") = 3;
mixed("t3") = 7;
mixed("t4") = 15;
mixed("t5") = 31;
mixed("t6") = 63;
mixed("t7") = 127;
mixed("t8") = 255;
mixed("t9") = 511;
mixed("t10") = 1023;

execute_unload 'test_mixed.gdx', t5, mixed;

* ============================================================================
* Test 6: Sparse data (many zeros)
* ============================================================================
Set t6 "time periods" / t1*t10 /;

Parameter sparse(t6) "sparse with gaps";
sparse("t1") = 100;
sparse("t3") = 300;
sparse("t7") = 700;
sparse("t10") = 1000;

execute_unload 'test_sparse.gdx', t6, sparse;

* ============================================================================
* Test 7: Growth rate (realistic geometric)
* ============================================================================
Set t7 "time periods" / t1*t10 /;

Parameter growth(t7) "growth at 5% per period";
growth("t1") = 1.0;
growth("t2") = 1.05;
growth("t3") = 1.1025;
growth("t4") = 1.157625;
growth("t5") = 1.21550625;
growth("t6") = 1.2762815625;
growth("t7") = 1.340095640625;
growth("t8") = 1.40710042265625;
growth("t9") = 1.4774554437890625;
growth("t10") = 1.551328215978515625;

execute_unload 'test_growth.gdx', t7, growth;

* ============================================================================
* Test 8: Small arithmetic (integers)
* ============================================================================
Set t8 "time periods" / t1*t5 /;

Parameter small_arith(t8) "small integers";
small_arith("t1") = 1;
small_arith("t2") = 2;
small_arith("t3") = 3;
small_arith("t4") = 4;
small_arith("t5") = 5;

execute_unload 'test_small_arithmetic.gdx', t8, small_arith;

* ============================================================================
* Test 9: Small geometric (powers of 2)
* ============================================================================
Set t9 "time periods" / t1*t5 /;

Parameter small_geom(t9) "powers of 2";
small_geom("t1") = 2;
small_geom("t2") = 4;
small_geom("t3") = 8;
small_geom("t4") = 16;
small_geom("t5") = 32;

execute_unload 'test_small_geometric.gdx', t9, small_geom;

* ============================================================================
* Display summary
* ============================================================================
