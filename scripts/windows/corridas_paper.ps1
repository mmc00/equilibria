<#
Corridas del paper que solo se pueden hacer en Windows (GEMPACK y sltoht).

Corre todo de una vez y deja los resultados en una carpeta con un manifiesto
(SHA-256 de cada archivo) y un resumen OK/FALLA por paso. Un paso que falla no
corta los demas. Nada se borra ni se cambia en el repo: cada paso trabaja sobre
una copia dentro de la carpeta de salida.

  1. sltoht real (#148, oraculo 3): export_sl4_levels.py sobre los .sl4 de
     Burfisher de tests\fixtures\gtap7_gempack\burfisher (se descomprimen en la
     carpeta de salida). Deja el texto de
     `sltoht -SHL -SIC` (<stem>_levels.txt) y el .har de niveles validado.
  2. sltoht real (#133): lo mismo sobre gtap7_3x3 tm10 y gtap7_20x41 capFix de
     runs\gempack_matrix. Copia ademas el .sl4 del 20x41 (#135).
  3. SIMPLE de Horridge (#134): GEMPACK (tablo + gemsim fixcap/mobcap) y GAMS
     (simpleNLP, simpleMCP, mpsgevh) sobre una copia de runs\horridge_simple.
  4. Niveles de y, qinv, rore, pinv, pva y pe: make_levels_tab.py parcha el
     gtapv7.tab de -GtapDir (orig_level en esas seis), tablo lo compila para
     GEMSIM, y se re-corren con gemsim el 3x3 tm10 y los 6 datasets capFix
     (3x3, 3x4, 5x5, 10x7, 15x10, 20x41: las filas de la Tabla 2) sobre una
     copia de su carpeta de runs\gempack_matrix. El .prm capFix se rehace desde
     data\gtap7\<ds>\default.prm con _force_rdlt (cambia solo el int32 de RDLT: el
     .prm que escribia write_har no lo lee GEMSIM 11.3). export_sl4_levels.py
     exporta los niveles contra el sl4dump de la corrida SIN parche: una variable
     cuyo % de cambio no coincide se descarta y queda nombrada en
     export_console.txt ("xcheck-mismatch=N [...]").
  5. Los ejercicios de Burfisher con los niveles de esas seis variables:
     make_levels_tab.py --uncondensed arma gtapuv7_levels (el gtapv7.tab
     parchado SIN condensar, como el GTAPUV7 de RunGTAP: los .exp fijan y
     shockean variables que la condensacion omite), tablo lo compila para
     GEMSIM y gemsim corre cada .exp de data\gtap7\nus333 (me8 y tbl813
     con su -dir.exp). Los .exp de Johansen de 1 paso (TBL45A/B/C, TBL64,
     ME4A/B) pasan a Gragg 2-4-6. Deja .sl4, .upd y log por ejercicio en
     5_burfisher\out; los niveles se leen directo del .sl4 en la Mac.
  6. Los cambios (% y ordinarios, con EV y u) de cada ejercicio de Burfisher:
     sltoht vuelca cada .sl4 a sl4dump_<exp>.har (el lector de la Mac solo lee
     niveles, y EV no tiene nivel). Dos juegos: (a) las soluciones del libro de
     tests\fixtures\gtap7_gempack\burfisher (Johansen donde el libro); (b) las
     del paso 5 de esta misma corrida (Gragg en todos), si estan. Es el
     oraculo GEMPACK del bienestar.

-Pasos elige que pasos correr (por defecto todos): -Pasos 4

Uso, desde la raiz del repo:
  powershell -ExecutionPolicy Bypass -File scripts\windows\corridas_paper.ps1
  powershell -ExecutionPolicy Bypass -File scripts\windows\corridas_paper.ps1 -GP C:\GP -Gams "C:\GAMS\53\gams.exe"
  powershell -ExecutionPolicy Bypass -File scripts\windows\corridas_paper.ps1 -Pasos 4
  powershell -ExecutionPolicy Bypass -File scripts\windows\corridas_paper.ps1 -Pasos 5
  powershell -ExecutionPolicy Bypass -File scripts\windows\corridas_paper.ps1 -Pasos 5,6
  powershell -ExecutionPolicy Bypass -File scripts\windows\corridas_paper.ps1 -Pasos 4 -Cierre capflex
  powershell -ExecutionPolicy Bypass -File scripts\windows\corridas_paper.ps1 -Pasos 4,5,6,7 -Cierre capflex

7. niveles de las corridas del paso 5 (Gragg, TAB de niveles): export_sl4_levels.py
   con sltoht sobre cada .sl4, como el paso 4. El lector de la Mac necesita esos
   niveles (el .sl4 solo no los trae). Usa los volcados del paso 6: 5, 6 y 7 van en
   la misma corrida.

-Cierre (paso 4): capfix = RDLT 0; capflex = el .prm por defecto (RDLT 1, el cierre
por defecto de GEMPACK); todos = los dos.

Con -File, "5,6" llega como UN texto; el script lo parte en la coma (antes, con
[int[]], se volvia el entero 56 y no corria nada).
#>
param(
    [string]$GP = "C:\GP",
    [string]$Gams = "",
    [string]$Out = "",
    [string]$GtapDir = "C:\runGTAP375",
    [string[]]$Pasos = @("1", "2", "3", "4", "5", "6", "7"),
    [ValidateSet("todos", "capfix", "capflex")]
    [string]$Cierre = "todos"
)

$ErrorActionPreference = "Continue"
# "-Pasos 5,6" llega como "5,6" con -File y como @("5","6") con -Command: se parte en
# la coma y se pasa a enteros, y un paso fuera de 1-6 corta antes de empezar.
$Pasos = @($Pasos | ForEach-Object { $_ -split "," } | Where-Object { $_.Trim() } | ForEach-Object { [int]$_.Trim() })
$malos = @($Pasos | Where-Object { $_ -lt 1 -or $_ -gt 6 })
if ($malos.Count -gt 0) { Write-Error "pasos fuera de 1-6: $($malos -join ',')"; exit 2 }
$Repo = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
Set-Location $Repo
if (-not $Out) { $Out = Join-Path $Repo ("windows_paper_out\" + (Get-Date -Format "yyyy-MM-dd_HHmm")) }
New-Item -ItemType Directory -Force -Path $Out | Out-Null
$Log = Join-Path $Out "log.txt"
$Summary = [System.Collections.Generic.List[string]]::new()

function Note([string]$msg) {
    $line = "[{0}] {1}" -f (Get-Date -Format "HH:mm:ss"), $msg
    Write-Host $line
    Add-Content -Path $Log -Value $line
}

function Result([string]$step, [bool]$ok, [string]$detail) {
    $tag = if ($ok) { "OK   " } else { "FALLA" }
    $Summary.Add("$tag $step  $detail")
    Note "$tag $step  $detail"
}

# --- entorno ---------------------------------------------------------------
$Sltoht = Join-Path $GP "sltoht.exe"
if (-not $Gams) {
    $cmd = Get-Command gams.exe -ErrorAction SilentlyContinue
    if ($cmd) { $Gams = $cmd.Source }
}
Note "repo      $Repo"
Note "commit    $(git rev-parse HEAD)"
Note "rama      $(git branch --show-current)"
Note "maquina   $env:COMPUTERNAME  $([System.Environment]::OSVersion.VersionString)"
Note "GEMPACK   $GP  (sltoht: $(Test-Path $Sltoht))"
Note "GAMS      $Gams"
Note "GTAP tab  $GtapDir"
Note "pasos     $($Pasos -join ',')"
Note "salida    $Out"
$env:PATH = "$GP;$env:PATH"

# --- export_sl4_levels.py sobre una copia del .sl4 ---------------------------
function Export-Levels([string]$step, [string]$sl4, [string]$sl4dump, [string]$dest) {
    if (-not (Test-Path $sl4)) { Result $step $false "no esta $sl4"; return }
    if (-not (Test-Path $sl4dump)) { Result $step $false "no esta $sl4dump"; return }
    New-Item -ItemType Directory -Force -Path $dest | Out-Null
    $copy = Join-Path $dest (Split-Path $sl4 -Leaf)
    Copy-Item $sl4 $copy -Force
    $stem = [System.IO.Path]::GetFileNameWithoutExtension($sl4)
    $har = Join-Path $dest "sl4levels_$stem.har"
    $console = Join-Path $dest "export_console.txt"
    $t0 = Get-Date
    & uv run python scripts\gtap\export_sl4_levels.py $copy --sl4dump $sl4dump --sltoht $Sltoht --out $har *> $console
    $rc = $LASTEXITCODE
    $secs = [math]::Round(((Get-Date) - $t0).TotalSeconds, 1)
    $text = Join-Path $dest "${stem}_levels.txt"
    $hasText = Test-Path $text
    Result $step ($rc -eq 0 -and $hasText) "rc=$rc texto_sltoht=$hasText ${secs}s ($dest)"
}

# --- 1. Burfisher (#148, oraculo 3) ----------------------------------------
$Fix = "tests\fixtures\gtap7_gempack"
$Matrix = "runs\gempack_matrix"
if ($Pasos -contains 1) {
foreach ($e in @("tbl45a", "me9c", "gtapv7")) {
    # se descomprime aparte: Export-Levels copia el .sl4 a $dest
    $gzDir = Join-Path $Out "1_burfisher\_sl4"
    New-Item -ItemType Directory -Force -Path $gzDir | Out-Null
    $sl4 = Join-Path $gzDir "$e.sl4"
    $in = [System.IO.File]::OpenRead((Resolve-Path "$Fix\burfisher\$e.sl4.gz").Path)
    $gz = New-Object System.IO.Compression.GZipStream($in, [System.IO.Compression.CompressionMode]::Decompress)
    $outFile = [System.IO.File]::Create($sl4)
    try { $gz.CopyTo($outFile) } finally { $outFile.Close(); $gz.Close(); $in.Close() }
    Export-Levels "1 sltoht $e" $sl4 "$Fix\burfisher\sl4dump_$e.har" (Join-Path $Out "1_burfisher\$e")
}

}

# --- 2. gtap7_3x3 tm10 y gtap7_20x41 capFix (#133, #135) --------------------
if ($Pasos -contains 2) {
Export-Levels "2 sltoht gtap7_3x3 tm10" "$Matrix\gtap7_3x3\tm10.sl4" "$Fix\sl4dump_gtap7_3x3_tm10.har" (Join-Path $Out "2_matriz\gtap7_3x3_tm10")
Export-Levels "2 sltoht gtap7_20x41 capfix" "$Matrix\gtap7_20x41\tm10_s8-16-32_capfix.sl4" "$Fix\sl4dump_gtap7_20x41_tm10_s8-16-32_capfix.har" (Join-Path $Out "2_matriz\gtap7_20x41_capfix")
if (-not (Test-Path "$Matrix\gtap7_3x3\tm10.sl4")) {
    Note "  .sl4 que hay en $Matrix\gtap7_3x3: $((Get-ChildItem "$Matrix\gtap7_3x3" -Filter *.sl4 -ErrorAction SilentlyContinue | ForEach-Object Name) -join ', ')"
}

}

# --- 3. SIMPLE de Horridge (#134) --------------------------------------------
if ($Pasos -contains 3) {
$Simple = Join-Path $Out "3_simple"
Copy-Item "runs\horridge_simple" $Simple -Recurse -Force
Push-Location $Simple
try {
    & tablo -pgs simple *> tablo_console.txt
    Result "3 SIMPLE tablo" ($LASTEXITCODE -eq 0) "rc=$LASTEXITCODE"
    foreach ($cmf in @("fixcap", "mobcap")) {
        & gemsim -cmf "$cmf.cmf" *> "gemsim_${cmf}_console.txt"
        Result "3 SIMPLE gemsim $cmf" ((Test-Path "$cmf.sl4")) "rc=$LASTEXITCODE sl4=$(Test-Path "$cmf.sl4")"
    }
    if ($Gams -and (Test-Path $Gams)) {
        foreach ($pair in @(@("simpleNLP", "ResultsNLP"), @("simpleMCP", "ResultsMCP"), @("mpsgevh", "ResultsMPSGE"))) {
            Remove-Item results.gdx -ErrorAction SilentlyContinue
            & $Gams "$($pair[0]).gms" lo=2 "lf=$($pair[0]).log" *> "gams_$($pair[0])_console.txt"
            $ok = Test-Path results.gdx
            if ($ok) { Move-Item results.gdx "$($pair[1]).gdx" -Force }
            Result "3 SIMPLE gams $($pair[0])" $ok "rc=$LASTEXITCODE gdx=$ok"
        }
    } else {
        Result "3 SIMPLE gams" $false "no se encontro gams.exe (pasar -Gams)"
    }
} finally {
    Pop-Location
}
}

# --- 4. niveles de y, qinv, rore, pinv, pva, pe (gtapv7_levels.tab) ----------
if ($Pasos -contains 4) {
$Lv = Join-Path $Out "4_niveles"
$Tab = Join-Path $Lv "tab"
New-Item -ItemType Directory -Force -Path $Tab | Out-Null
& uv run python scripts\gtap\make_levels_tab.py (Join-Path $GtapDir "gtapv7.tab") --sti (Join-Path $GtapDir "gtapv7.sti") --out-dir $Tab *> (Join-Path $Tab "make_levels_tab_console.txt")
$okTab = ($LASTEXITCODE -eq 0) -and (Test-Path (Join-Path $Tab "gtapv7_levels.tab"))
Result "4 make_levels_tab" $okTab "rc=$LASTEXITCODE ($Tab)"
if ($okTab) {
    Push-Location $Tab
    try {
        & tablo -sti gtapv7_levels.sti *> tablo_console.txt
        $okGss = (Test-Path gtapv7_levels.gss) -and (Test-Path gtapv7_levels.gst)
        Result "4 tablo gtapv7_levels" $okGss "rc=$LASTEXITCODE gss/gst=$okGss"
    } finally {
        Pop-Location
    }
    if ($okGss) {
        $Aux = Join-Path $Tab "gtapv7_levels"
        # 3x3: tm10.cmf ya no existe (era el cierre viejo con swap dpsave);
        # sl4dump_gtap7_3x3_tm10.har es identico (252/252) a la corrida tm10_s8-16-32.
        # capFlex (rdlt -1 = .prm por defecto, RDLT 1): el cierre de la Tabla 2 del paper.
        # capFix (RDLT 0): la Tabla 2 anterior (#135); equilibria capFix fija el ahorro
        # externo y GEMPACK RDLT 0 la composicion del capital: no son el mismo cierre.
        $jobs = @(@("gtap7_3x3", "tm10_s8-16-32", "sl4dump_gtap7_3x3_tm10.har", -1),
                  @("gtap7_3x4", "tm10_s8-16-32", "sl4dump_gtap7_3x4_tm10_s8-16-32.har", -1),
                  @("gtap7_5x5", "tm10_s8-16-32", "sl4dump_gtap7_5x5_tm10_s8-16-32.har", -1),
                  @("gtap7_10x7", "tm10_s8-16-32", "sl4dump_gtap7_10x7_tm10_s8-16-32.har", -1),
                  @("gtap7_15x10", "tm10_s8-16-32", "sl4dump_gtap7_15x10_tm10_s8-16-32.har", -1),
                  @("gtap7_20x41", "tm10_s8-16-32", "sl4dump_gtap7_20x41_tm10_s8-16-32.har", -1),
                  @("gtap7_3x3", "tm10_s8-16-32_capfix", "sl4dump_gtap7_3x3_tm10_s8-16-32_capfix.har", 0),
                  @("gtap7_3x4", "tm10_s8-16-32_capfix", "sl4dump_gtap7_3x4_tm10_s8-16-32_capfix.har", 0),
                  @("gtap7_5x5", "tm10_s8-16-32_capfix", "sl4dump_gtap7_5x5_tm10_s8-16-32_capfix.har", 0),
                  @("gtap7_10x7", "tm10_s8-16-32_capfix", "sl4dump_gtap7_10x7_tm10_s8-16-32_capfix.har", 0),
                  @("gtap7_15x10", "tm10_s8-16-32_capfix", "sl4dump_gtap7_15x10_tm10_s8-16-32_capfix.har", 0),
                  @("gtap7_20x41", "tm10_s8-16-32_capfix", "sl4dump_gtap7_20x41_tm10_s8-16-32_capfix.har", 0))
        foreach ($job in $jobs) {
            $ds, $tag, $dump, $rdlt = $job
            if (($Cierre -eq "capfix" -and $rdlt -lt 0) -or ($Cierre -eq "capflex" -and $rdlt -ge 0)) { continue }
            $step = "4 gemsim $ds $tag"
            $src = Join-Path $Matrix $ds
            if (-not (Test-Path (Join-Path $src "$tag.cmf"))) { Result $step $false "no esta $src\$tag.cmf"; continue }
            $dest = Join-Path $Lv "${ds}_$tag"
            New-Item -ItemType Directory -Force -Path $dest | Out-Null
            # la carpeta de la corrida sin sus salidas viejas: .cmf y entradas
            Get-ChildItem $src -File | Where-Object { $_.Extension -in ".cmf", ".har", ".prm" -and $_.Name -notmatch "^(updated|sl4dump|sl4levels|summary|decomp|volume)" } |
                Copy-Item -Destination $dest -Force
            if ($rdlt -ge 0) {
                $prm = Join-Path $dest "default_$tag.prm"
                Copy-Item "data\gtap7\$ds\default.prm" $prm -Force
                & uv run python -c "import sys; sys.path.insert(0, 'data/gtap7/gempack/tools'); from pathlib import Path; from run_gempack_matrix import _force_rdlt; _force_rdlt(Path(sys.argv[1]), int(sys.argv[2]))" $prm $rdlt *> (Join-Path $dest "force_rdlt_console.txt")
                if ($LASTEXITCODE -ne 0) { Result $step $false "fallo _force_rdlt (ver force_rdlt_console.txt)"; continue }
            }
            $cmf = Join-Path $dest "$tag.cmf"
            $text = Get-Content $cmf -Raw
            $new = [regex]::Replace($text, "(?im)^\s*Auxiliary files\s*=.*$", "Auxiliary files = $Aux ;")
            if ($new -eq $text) { Result $step $false "el .cmf no tiene linea 'Auxiliary files'"; continue }
            Set-Content -Path $cmf -Value $new -NoNewline
            Push-Location $dest
            $t0 = Get-Date
            try {
                & gemsim -cmf "$tag.cmf" *> "gemsim_console.txt"
            } finally {
                Pop-Location
            }
            $secs = [math]::Round(((Get-Date) - $t0).TotalSeconds, 1)
            $sl4 = Join-Path $dest "$tag.sl4"
            Result $step (Test-Path $sl4) "rc=$LASTEXITCODE sl4=$(Test-Path $sl4) ${secs}s"
            if (Test-Path $sl4) {
                # el sl4dump es el de la corrida SIN parche: export_sl4_levels compara sus
                # % de cambio con los de esta corrida; las que no coinciden se descartan
                # y se nombran en export_console.txt.
                Export-Levels "4 niveles $ds $tag" $sl4 "$Fix\$dump" (Join-Path $dest "niveles")
            }
        }
    }
}
}

# --- 5. Burfisher con los niveles de las 6 y Gragg (gtapuv7_levels) ----------
if ($Pasos -contains 5) {
$Bf = Join-Path $Out "5_burfisher"
$Tab5 = Join-Path $Bf "tab"
New-Item -ItemType Directory -Force -Path $Tab5 | Out-Null
& uv run python scripts\gtap\make_levels_tab.py (Join-Path $GtapDir "gtapv7.tab") --sti (Join-Path $GtapDir "gtapv7.sti") --out-dir $Tab5 --uncondensed *> (Join-Path $Tab5 "make_levels_tab_console.txt")
$okTab5 = ($LASTEXITCODE -eq 0) -and (Test-Path (Join-Path $Tab5 "gtapuv7_levels.tab"))
Result "5 make_levels_tab --uncondensed" $okTab5 "rc=$LASTEXITCODE ($Tab5)"
$okGss5 = $false
if ($okTab5) {
    Push-Location $Tab5
    try {
        & tablo -sti gtapuv7_levels.sti *> tablo_console.txt
        $okGss5 = (Test-Path gtapuv7_levels.gss) -and (Test-Path gtapuv7_levels.gst)
        Result "5 tablo gtapuv7_levels" $okGss5 "rc=$LASTEXITCODE gss/gst=$okGss5"
    } finally {
        Pop-Location
    }
}
if ($okGss5) {
    $Aux5 = Join-Path $Tab5 "gtapuv7_levels"
    $Pkg = Join-Path $Bf "nus333"
    Copy-Item "data\gtap7\nus333" $Pkg -Recurse -Force
    # me8/tbl813: corren sus -dir.exp (rate% traducido); tbl53 usa target%, que solo
    # resuelve la GUI; gtapv7.exp declara el cierre del modelo CONDENSADO.
    $skip = @("me8", "tbl813", "tbl53", "gtapv7")
    foreach ($f in (Get-ChildItem $Pkg -Filter *.exp | Sort-Object Name)) {
        $name = $f.BaseName
        if ($skip -contains $name) { continue }
        $step = "5 gemsim $name"
        $d = Join-Path $Bf "out\$name"
        New-Item -ItemType Directory -Force -Path $d | Out-Null
        # El .exp va tal cual (trae GTAPPARM, cierre y shocks), salvo el metodo: los
        # Johansen de 1 paso del libro pasan a Gragg 2-4-6, que es la solucion en
        # niveles contra la que se compara equilibria.
        $exp = Get-Content $f.FullName -Raw
        $johansen = $exp -match "(?im)^\s*Method\s*=\s*Johansen\s*;"
        if ($johansen) {
            $exp = [regex]::Replace($exp, "(?im)^\s*Method\s*=\s*Johansen\s*;", "Method = Gragg;")
            $exp = [regex]::Replace($exp, "(?im)^\s*Steps\s*=[^;]*;", "Steps = 2 4 6;")
        }
        $nMethod = ([regex]::Matches($exp, "(?im)^\s*Method\s*=")).Count
        if ($nMethod -ne 1) { Result $step $false "el .exp queda con $nMethod lineas Method"; continue }
        $head = @(
            "auxiliary files = `"$Aux5`";",
            "file GTAPSETS = sets.har;",
            "file GTAPDATA = basedata.har;",
            "file GTAPSUM = `"$d\SUMMARY.har`";",
            "file WELVIEW = `"$d\DECOMP.har`";",
            "file GTAPVOL = `"$d\GTAPVol.har`";",
            "updated file GTAPDATA = `"$d\$name.upd`";",
            "solution file = `"$d\$name`";",
            "CPU = yes;",
            "NDS = yes;",
            ""
        ) -join "`r`n"
        $cmf = Join-Path $d "$name.cmf"
        Set-Content -Path $cmf -Value ($head + $exp) -NoNewline
        # el .exp nombra su .prm (y los .shk) sin ruta: se corre desde el paquete
        Push-Location $Pkg
        $t0 = Get-Date
        try {
            & gemsim -cmf $cmf *> (Join-Path $d "$name.log")
            $rc = $LASTEXITCODE
        } finally {
            Pop-Location
        }
        $secs = [math]::Round(((Get-Date) - $t0).TotalSeconds, 1)
        $sl4 = Join-Path $d "$name.sl4"
        # gemsim escribe el .sl4 aunque falle (ME4A/B: "Accuracy too low" con Gragg
        # 2-4-6): OK exige rc=0 y el cierre normal del log, no solo el archivo.
        $log = Join-Path $d "$name.log"
        $fin = (Test-Path $log) -and (Select-String -Path $log -SimpleMatch "completed without error" -Quiet)
        $ok = ($rc -eq 0) -and $fin -and (Test-Path $sl4) -and (Test-Path (Join-Path $d "$name.upd"))
        $m = if ($johansen) { "Johansen->Gragg" } else { "como el .exp" }
        Result $step $ok "rc=$rc fin_normal=$fin sl4=$(Test-Path $sl4) metodo=$m ${secs}s"
    }
}
}

# --- 6. cambios de cada ejercicio de Burfisher: EV y u (sltoht) -------------
# sltoht en batch: <solucion> / c / n / har / <salida> / e (run_gempack_matrix.convert_sl4).
function Dump-Changes([string]$step, [string]$sl4, [string]$dest) {
    New-Item -ItemType Directory -Force -Path $dest | Out-Null
    $stem = [System.IO.Path]::GetFileNameWithoutExtension($sl4)
    $copy = Join-Path $dest "$stem.sl4"
    if ($sl4 -ne $copy) { Copy-Item $sl4 $copy -Force }
    Set-Content -Path (Join-Path $dest "sl4dump_$stem.sti") -Value "`r`n$stem`r`nc`r`nn`r`nhar`r`nsl4dump_$stem.har`r`ne`r`n" -NoNewline -Encoding ascii
    Push-Location $dest
    try {
        & $Sltoht -sti "sl4dump_$stem.sti" *> "sltoht_console_$stem.txt"
        $rc = $LASTEXITCODE
    } finally {
        Pop-Location
    }
    $har = Join-Path $dest "sl4dump_$stem.har"
    Result $step (Test-Path $har) "rc=$rc har=$(Test-Path $har) ($dest)"
}

if ($Pasos -contains 6) {
$D6 = Join-Path $Out "6_sl4dump"
# (a) las soluciones del libro: el .sl4 va comprimido en los fixtures
foreach ($gz in (Get-ChildItem "$Fix\burfisher" -Filter *.sl4.gz | Sort-Object Name)) {
    $stem = $gz.Name -replace "\.sl4\.gz$", ""
    $d = Join-Path $D6 "libro\$stem"
    New-Item -ItemType Directory -Force -Path $d | Out-Null
    $sl4 = Join-Path $d "$stem.sl4"
    $in = [System.IO.File]::OpenRead($gz.FullName)
    $zip = New-Object System.IO.Compression.GZipStream($in, [System.IO.Compression.CompressionMode]::Decompress)
    $outFs = [System.IO.File]::Create($sl4)
    try { $zip.CopyTo($outFs) } finally { $outFs.Close(); $zip.Close(); $in.Close() }
    Dump-Changes "6 sltoht libro $stem" $sl4 $d
}
# (b) las del paso 5 (Gragg en todos), si corrio en esta misma salida
$Out5 = Join-Path $Out "5_burfisher\out"
if (Test-Path $Out5) {
    foreach ($sl4 in (Get-ChildItem $Out5 -Recurse -Filter *.sl4 | Sort-Object Name)) {
        Dump-Changes "6 sltoht gragg $($sl4.BaseName)" $sl4.FullName (Join-Path $D6 "gragg\$($sl4.BaseName)")
    }
} else {
    Note "6: no hay $Out5 (el paso 5 no corrio en esta salida): solo las del libro"
}
}

# --- 7. niveles de las corridas del paso 5 ----------------------------------
if ($Pasos -contains 7) {
$Out5 = Join-Path $Out "5_burfisher\out"
$D6g = Join-Path $Out "6_sl4dump\gragg"
if ((Test-Path $Out5) -and (Test-Path $D6g)) {
    foreach ($sl4 in (Get-ChildItem $Out5 -Recurse -Filter *.sl4 | Sort-Object Name)) {
        $stem = $sl4.BaseName
        Export-Levels "7 niveles gragg $stem" $sl4.FullName (Join-Path $D6g "$stem\sl4dump_$stem.har") (Join-Path $Out "7_niveles_gragg\$stem")
    }
} else {
    Note "7: no hay $Out5 o $D6g (los pasos 5 y 6 tienen que correr en esta misma salida)"
}
}

# --- manifiesto y resumen --------------------------------------------------
$Manifest = Join-Path $Out "manifest.txt"
Get-ChildItem $Out -Recurse -File | Where-Object { $_.FullName -ne $Manifest } | ForEach-Object {
    $h = Get-FileHash $_.FullName -Algorithm SHA256
    "{0}  {1}  {2}" -f $h.Hash.ToLower(), $_.Length, $_.FullName.Substring($Out.Length + 1)
} | Set-Content $Manifest
$Summary | Set-Content (Join-Path $Out "resumen.txt")

Write-Host ""
Write-Host "==== RESUMEN ===="
$Summary | ForEach-Object { Write-Host $_ }
$rel = $Out.Substring($Repo.Length + 1)
Write-Host ""
# Las salidas crudas (.sl4, .har, consolas) van a una rama de DATOS que nunca se
# mergea, no a la rama de trabajo: lo que necesitan los tests se copia despues a
# tests/fixtures. Solo se exime el hook de archivos grandes (los .sl4 pesan MB);
# los demas hooks corren.
$leaf = Split-Path $Out -Leaf
Write-Host "Para traer los resultados a la Mac (rama de datos, no se mergea):"
Write-Host "  git switch -c windows/corridas-$leaf"
Write-Host "  git add -f `"$rel`""
Write-Host "  `$env:SKIP = 'check-added-large-files'; git commit -m `"data(windows): corridas del paper $leaf`"; Remove-Item Env:SKIP"
Write-Host "  git push -u origin windows/corridas-$leaf"
Write-Host "  git switch -"
