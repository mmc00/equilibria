$title Test 5D Parameters for GDX Reader

* Create sets for testing
Set
    i "dimension 1" / i1*i2 /
    j "dimension 2" / j1*j3 /
    k "dimension 3" / k1*k2 /
    m "dimension 4" / m1*m2 /
    n "dimension 5" / n1*n2 /
;

* 5-dimensional parameter (sparse)
Parameter p5d_sparse(i,j,k,m,n) "5D parameter - sparse";
p5d_sparse('i1','j1','k1','m1','n1') = 11111;
p5d_sparse('i1','j2','k1','m1','n2') = 12112;
p5d_sparse('i1','j2','k2','m2','n1') = 12221;
p5d_sparse('i2','j3','k1','m2','n2') = 23122;
p5d_sparse('i2','j3','k2','m1','n1') = 23211;
p5d_sparse('i2','j3','k2','m2','n2') = 23222;

* 5-dimensional parameter (denser)
Parameter p5d_dense(i,j,k,m,n) "5D parameter - denser";
p5d_dense(i,j,k,m,n) = ord(i)*10000 + ord(j)*1000 + ord(k)*100 + ord(m)*10 + ord(n);

* Also create smaller dimensions for comparison
Parameter p4d_small(i,j,k,m) "4D parameter - small";
p4d_small('i1','j1','k1','m1') = 1111;
p4d_small('i2','j2','k2','m2') = 2222;

Parameter p3d_small(i,j,k) "3D parameter - small";
p3d_small('i1','j1','k1') = 111;
p3d_small('i2','j2','k2') = 222;

* Display the parameters
display p3d_small, p4d_small, p5d_sparse, p5d_dense;

* Export to GDX
execute_unload 'test_5d.gdx', i, j, k, m, n, p3d_small, p4d_small, p5d_sparse, p5d_dense;
