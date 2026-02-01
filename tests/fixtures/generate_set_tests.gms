$title Generate GDX test files for multidimensional sets

* Test 1: 2D set (sparse)
Sets
    i "Industries" / agr, mfg, srv /
    j "Products" / food, goods, services /
    map(i,j) "Mapping of industries to products" /
        agr.food
        agr.services
        mfg.goods
        mfg.services
        srv.services
    /
;

execute_unload 'set_2d_sparse.gdx', i, j, map;

* Test 2: 2D set (full)
Sets
    r "Regions" / north, south, east, west /
    rr "Region pairs" / north.south, north.east, south.east, east.west /
;

execute_unload 'set_2d_full.gdx', r, rr;

* Test 3: 3D set
Sets
    i3 "Dim 1" / a, b, c /
    j3 "Dim 2" / x, y /
    k3 "Dim 3" / p, q /
    cube(i3,j3,k3) "3D mapping" /
        a.x.p
        a.x.q
        a.y.p
        b.x.p
        b.y.q
        c.y.p
        c.y.q
    /
;

execute_unload 'set_3d.gdx', i3, j3, k3, cube;

* Test 4: 2D set with all combinations (Cartesian product)
Sets
    ii "Set 1" / a1, a2, a3 /
    jj "Set 2" / b1, b2 /
    cart(ii,jj) "Cartesian product" /
        a1.b1, a1.b2
        a2.b1, a2.b2
        a3.b1, a3.b2
    /
;

execute_unload 'set_2d_cartesian.gdx', ii, jj, cart;

* Test 5: Mixed - sets and parameters referencing 2D set
Sets
    src "Sources" / s1, s2, s3 /
    dst "Destinations" / d1, d2 /
    route(src,dst) "Valid routes" /
        s1.d1, s1.d2
        s2.d1
        s3.d2
    /
;

Parameter cost(src,dst) "Transport cost on valid routes";
cost('s1','d1') = 10;
cost('s1','d2') = 15;
cost('s2','d1') = 8;
cost('s3','d2') = 12;

execute_unload 'set_2d_with_param.gdx', src, dst, route, cost;

* Test 6: 4D set (very sparse)
Sets
    i4 / x1, x2 /
    j4 / y1, y2 /
    k4 / z1, z2 /
    l4 / w1, w2 /
    hypercube(i4,j4,k4,l4) "4D sparse set" /
        x1.y1.z1.w1
        x1.y2.z1.w2
        x2.y1.z2.w1
        x2.y2.z2.w2
    /
;

execute_unload 'set_4d.gdx', i4, j4, k4, l4, hypercube;

* Test 7: 2D set with text/descriptions
Sets
    region "Regions" / r1 "North Region", r2 "South Region", r3 "East Region" /
    product "Products" / p1 "Food", p2 "Textiles", p3 "Electronics" /
    supply(region,product) "Regional supply" /
        r1.p1, r1.p3
        r2.p1, r2.p2
        r3.p2, r3.p3
    /
;

execute_unload 'set_2d_with_text.gdx', region, product, supply;

* Test 8: Empty 2D set
Sets
    e1 "Empty dim 1" / e_a, e_b /
    e2 "Empty dim 2" / e_x, e_y /
    empty(e1,e2) "Empty 2D set" / /
;

execute_unload 'set_2d_empty.gdx', e1, e2, empty;

$exit
