$title Simple GDX test file

sets
    i "industries" / agr, mfg, srv /
    j "commodities" / food, goods, services /
;

parameters
    price(i) "prices by industry" / agr 1.0, mfg 1.5, srv 2.0 /
    sam(i,j) "simple SAM"
;

sam(i,j) = uniform(0, 100);

execute_unload "simple_test.gdx", i, j, price, sam;
