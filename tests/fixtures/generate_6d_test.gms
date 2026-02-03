$title Test 6D Parameters for GDX Reader

* Define sets for 6D parameters
Sets
    i       "First dimension"     / i1*i2 /
    j       "Second dimension"    / j1*j2 /
    k       "Third dimension"     / k1*k2 /
    l       "Fourth dimension"    / l1*l2 /
    m       "Fifth dimension"     / m1*m2 /
    n       "Sixth dimension"     / n1*n2 /
;

* 6D Sparse parameter (8 values - one per corner of 6D hypercube)
Parameter p6d_sparse(i,j,k,l,m,n) "Sparse 6D parameter";
p6d_sparse('i1','j1','k1','l1','m1','n1') = 111.111;
p6d_sparse('i2','j1','k1','l1','m1','n1') = 211.111;
p6d_sparse('i1','j2','k1','l1','m1','n1') = 121.111;
p6d_sparse('i1','j1','k2','l1','m1','n1') = 112.111;
p6d_sparse('i1','j1','k1','l2','m1','n1') = 111.211;
p6d_sparse('i1','j1','k1','l1','m2','n1') = 111.121;
p6d_sparse('i1','j1','k1','l1','m1','n2') = 111.112;
p6d_sparse('i2','j2','k2','l2','m2','n2') = 222.222;

* 6D Dense parameter (64 values - full 2^6 hypercube)
Parameter p6d_dense(i,j,k,l,m,n) "Dense 6D parameter";
p6d_dense(i,j,k,l,m,n) = ord(i)*100000 + ord(j)*10000 + ord(k)*1000 + ord(l)*100 + ord(m)*10 + ord(n);

* Smaller 3D and 4D for comparison
Parameter p3d_small(i,j,k) "Small 3D for testing";
p3d_small(i,j,k) = ord(i)*100 + ord(j)*10 + ord(k);

Parameter p4d_small(i,j,k,l) "Small 4D for testing";
p4d_small(i,j,k,l) = ord(i)*1000 + ord(j)*100 + ord(k)*10 + ord(l);

* Display for verification
display p6d_sparse, p6d_dense, p3d_small, p4d_small;

* Export to GDX
execute_unload 'test_6d.gdx', p6d_sparse, p6d_dense, p3d_small, p4d_small;
