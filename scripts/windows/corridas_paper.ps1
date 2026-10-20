<#
Corridas del paper que solo se pueden hacer en Windows (GEMPACK y sltoht).

Corre todo de una vez y deja los resultados en una carpeta con un manifiesto
(SHA-256 de cada archivo) y un resumen OK/FALLA por paso. Un paso que falla no
corta los demas. Nada se borra ni se cambia en el repo: cada paso trabaja sobre
una copia dentro de la carpeta de salida.

  1. sltoht real (#148, oraculo 3): export_sl4_levels.py sobre los .sl4 de
     Burfisher de tests\fixtures\gtap7_gempack\burfisher_sl4. Deja el texto de
     `sltoht -SHL -SIC` (<stem>_levels.txt) y el .har de niveles validado.
  2. sltoht real (#133): lo mismo sobre gtap7_3x3 tm10 y gtap7_20x41 capFix de
     runs\gempack_matrix. Copia ademas el .sl4 del 20x41 (#135).
  3. SIMPLE de Horridge (#134): GEMPACK (tablo + gemsim fixcap/mobcap) y GAMS
     (simpleNLP, simpleMCP, mpsgevh) sobre una copia de runs\horridge_simple.
  4. Niveles de y, qinv, rore, pinv, pva y pe: make_levels_tab.py parcha el
     gtapv7.tab de -GtapDir (orig_level en esas seis), tablo lo compila para
     GEMSIM, y se re-corren con gemsim el 3x3 tm10 y el 20x41 capFix sobre una
     copia de su carpeta de runs\gempack_matrix. El .prm capFix se rehace desde
     datasets\<ds>\default.prm con _force_rdlt (cambia solo el int32 de RDLT: el
     .prm que escribia write_har no lo lee GEMSIM 11.3). export_sl4_levels.py
     exporta los niveles contra el sl4dump de la corrida SIN parche: una variable
     cuyo % de cambio no coincide se descarta y queda nombrada en
     export_console.txt ("xcheck-mismatch=N [...]").

-Pasos elige que pasos correr (por defecto todos): -Pasos 4

Uso, desde la raiz del repo:
  powershell -ExecutionPolicy Bypass -File scripts\windows\corridas_paper.ps1
  powershell -ExecutionPolicy Bypass -File scripts\windows\corridas_paper.ps1 -GP C:\GP -Gams "C:\GAMS\53\gams.exe"
  powershell -ExecutionPolicy Bypass -File scripts\windows\corridas_paper.ps1 -Pasos 4
#>
param(
    [string]$GP = "C:\GP",
    [string]$Gams = "",
    [string]$Out = "",
    [string]$GtapDir = "C:\runGTAP375",
    [int[]]$Pasos = @(1, 2, 3, 4)
)

$ErrorActionPreference = "Continue"
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
foreach ($e in @("TBL45A", "ME9C", "GTAPV7")) {
    Export-Levels "1 sltoht $e" "$Fix\burfisher_sl4\$e.sl4" "$Fix\burfisher_sl4\sl4dump_$e.har" (Join-Path $Out "1_burfisher\$e")
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
        foreach ($job in @(@("gtap7_3x3", "tm10_s8-16-32", "sl4dump_gtap7_3x3_tm10.har", -1),
                           @("gtap7_20x41", "tm10_s8-16-32_capfix", "sl4dump_gtap7_20x41_tm10_s8-16-32_capfix.har", 0))) {
            $ds, $tag, $dump, $rdlt = $job
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
                Copy-Item "datasets\$ds\default.prm" $prm -Force
                & uv run python -c "import sys; sys.path.insert(0, 'scripts/gtap'); from pathlib import Path; from run_gempack_matrix import _force_rdlt; _force_rdlt(Path(sys.argv[1]), int(sys.argv[2]))" $prm $rdlt *> (Join-Path $dest "force_rdlt_console.txt")
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
