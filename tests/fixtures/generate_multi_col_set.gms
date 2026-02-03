$title Test with multiple columns per row

Sets
    i / a, b /
    j / x, y, z /
    map(i,j) / a.x, a.y, b.z /
;

execute_unload 'set_2d_multi_col.gdx', i, j, map;

$exit
