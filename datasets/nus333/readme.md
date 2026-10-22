# Paquete de experimentos de Burfisher 3e (nus333)

Los archivos que acompañan *Introduction to Computable General Equilibrium Models* (Burfisher, 3.ª ed.) para el dataset nus333: los `.exp` de cada tabla del libro, sus `.prm`, los `.shk` y los datos (`sets.har`, `basedata.har`, ...). Están copiados tal cual del paquete del libro, con los nombres en minúsculas. Windows no distingue mayúsculas, así que los `.exp` siguen encontrando sus `.prm`.

`me8-dir.exp` y `tbl813-dir.exp` los genera `scripts/gtap/mk_exp_directos.py` desde `me8.exp` y `tbl813.exp`. Esos dos usan `rate% N from file X.shk`, una sintaxis que sólo lee la GUI de RunGTAP. El script la traduce a un shock directo por celda. `tests/scripts/test_mk_exp_directos.py` comprueba que los shocks traducidos son los que corrieron en Windows.

`scripts/windows/corridas_paper.ps1 -Pasos 5` corre los experimentos con GEMPACK.
