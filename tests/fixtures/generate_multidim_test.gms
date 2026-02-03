$title Test Multi-Dimensional Parameters for GDX Reader

* Create sets for testing
Set
    i "dimension 1" / i1*i3 /
    j "dimension 2" / j1*j4 /
    k "dimension 3" / k1*k2 /
    m "dimension 4" / m1*m3 /
;

* 3-dimensional parameter (dense)
Parameter p3d(i,j,k) "3D parameter - dense";
p3d(i,j,k) = ord(i) * 100 + ord(j) * 10 + ord(k);

* 3-dimensional parameter (sparse)
Parameter p3d_sparse(i,j,k) "3D parameter - sparse";
p3d_sparse('i1','j1','k1') = 111;
p3d_sparse('i1','j2','k1') = 121;
p3d_sparse('i2','j3','k2') = 232;
p3d_sparse('i3','j4','k1') = 341;

* 4-dimensional parameter (dense)
Parameter p4d(i,j,k,m) "4D parameter - dense";
p4d(i,j,k,m) = ord(i) * 1000 + ord(j) * 100 + ord(k) * 10 + ord(m);

* 4-dimensional parameter (sparse)
Parameter p4d_sparse(i,j,k,m) "4D parameter - sparse";
p4d_sparse('i1','j1','k1','m1') = 1111;
p4d_sparse('i1','j2','k1','m2') = 1212;
p4d_sparse('i2','j3','k2','m3') = 2323;
p4d_sparse('i3','j4','k1','m1') = 3411;
p4d_sparse('i3','j4','k2','m2') = 3422;

* Also create some 2D parameters for comparison
Parameter p2d(i,j) "2D parameter";
p2d(i,j) = ord(i) * 10 + ord(j);

* Display the parameters
display p2d, p3d, p3d_sparse, p4d, p4d_sparse;

* Export to GDX
execute_unload 'multidim_test.gdx', i, j, k, m, p2d, p3d, p3d_sparse, p4d, p4d_sparse;
