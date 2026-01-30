$title GDX Test Fixtures Generator

* Fixture 1: Variables and Equations test
Sets
    i "industries" / agr, mfg, srv /
    j "commodities" / food, goods, services /
;

Alias(i, ii);

Parameters
    price(i) "prices by industry"
    sam(i,j) "simple SAM"
;

price("agr") = 1.5;
price("mfg") = 2.0;
price("srv") = 2.5;

sam("agr","food") = 100;
sam("agr","goods") = 50;
sam("agr","services") = 25;
sam("mfg","food") = 30;
sam("mfg","goods") = 200;
sam("mfg","services") = 40;
sam("srv","food") = 20;
sam("srv","goods") = 60;
sam("srv","services") = 150;

Variables
    X(i) "output by industry"
    Y "total output"
    obj "objective"
;

X.l("agr") = 100;
X.l("mfg") = 200;
X.l("srv") = 300;
X.lo("agr") = 0;
X.up("agr") = 500;
Y.l = 600;
obj.l = 1000;

Equations
    eq_output(i) "output equation"
    eq_total "total equation"
    eq_obj "objective equation"
;

eq_output(i).. X(i) =g= sum(j, sam(i,j));
eq_total.. Y =e= sum(i, X(i));
eq_obj.. obj =e= Y;

Model testmodel /all/;
Solve testmodel using lp minimizing obj;

execute_unload 'variables_equations_test.gdx', i, j, price, sam, X, Y, obj, eq_output, eq_total;


* Fixture 2: Multi-dimensional test
Sets
    r "regions" / north, south, east, west /
    t "time periods" / t1*t5 /
    s "sectors" / s1*s3 /
;

Parameter
    data3d(r,t,s) "3-dimensional data"
;

data3d(r,t,s) = uniform(1,100);

execute_unload 'multidim_test.gdx', r, t, s, data3d;


* Fixture 3: Sparse data
Sets
    sparse_set / a, c, e, g /
;

Parameter
    sparse_param(sparse_set) "sparse parameter"
;

sparse_param("a") = 1;
sparse_param("e") = 5;

execute_unload 'sparse_test.gdx', sparse_set, sparse_param;
