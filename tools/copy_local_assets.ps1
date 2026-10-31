# Copy only the fixed inputs, configurations, compact training samples, and text results.
# Uses the ORIGINAL experiment project as read-only source; never copies raw PUMA TIFF/GeoJSON.
param(
  [string]$SourceRoot = 'E:\PUMA_Stain_Robustness'
)
$ErrorActionPreference = 'Stop'
$SourceRoot = (Resolve-Path -LiteralPath $SourceRoot).Path
$DestRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$script:Copied = 0
$script:Missing = New-Object System.Collections.Generic.List[string]
function Copy-Relative([string]$Rel, [bool]$Required=$false) {
  $from = Join-Path $SourceRoot ($Rel.Replace('/', [IO.Path]::DirectorySeparatorChar))
  $to = Join-Path $DestRoot ($Rel.Replace('/', [IO.Path]::DirectorySeparatorChar))
  if (Test-Path -LiteralPath $from -PathType Leaf) {
    $parent = Split-Path -Parent $to
    New-Item -ItemType Directory -Path $parent -Force | Out-Null
    Copy-Item -LiteralPath $from -Destination $to -Force
    $script:Copied++
  } elseif ($Required) { $script:Missing.Add($Rel) }
}
function Copy-Pattern([string]$Dir,[string]$Pattern,[bool]$Required=$false) {
  $folder = Join-Path $SourceRoot ($Dir.Replace('/', [IO.Path]::DirectorySeparatorChar))
  $found = @(Get-ChildItem -LiteralPath $folder -Filter $Pattern -File -ErrorAction SilentlyContinue)
  if ($found.Count -eq 0 -and $Required) {$script:Missing.Add("$Dir/$Pattern")}
  foreach ($file in $found) { Copy-Relative "$Dir/$($file.Name)" $false }
}
Copy-Relative 'inputs/splits/cv_splits_seed13_folds5.npz' $true
Copy-Relative 'inputs/training_samples/common_training_pixels_seed13.npz' $true
Copy-Relative 'outputs/stain_calibration/fold_perturbation_parameters.csv' $true
Copy-Relative 'outputs/stain_calibration/fold_stain_statistics.csv'
Copy-Relative 'outputs/stain_calibration/stain_stats_per_roi.csv'
foreach ($f in 1..5) {
  $fold='{0:d2}' -f $f
  Copy-Relative "outputs/enhancement_targets/fold${fold}_enhancement_target.json" $true
  Copy-Relative "outputs/percentile_mechanism/fold${fold}_paired_mechanism.csv" $true
  Copy-Relative "outputs/percentile_mechanism/fold${fold}_run_config.json" $true
  Copy-Relative "outputs/percentile_mechanism/fold${fold}_integrity.json" $true
}
foreach ($m in @('no_correction','clahe','macenko','reinhard')) {
  Copy-Relative "outputs/clean_rf/$m/run_config.json"
  Copy-Relative "outputs/clean_rf/$m/clean_metrics_per_roi.csv" $true
  Copy-Relative "outputs/stain_robustness/$m/run_config.json"
  foreach ($f in 1..5) {
    $fold='{0:d2}' -f $f
    Copy-Relative "outputs/stain_robustness/$m/folds/fold${fold}_robustness_metrics.csv" $true
  }
}
foreach ($dir in @('outputs/stage16_statistical_audit', 'outputs/stage17_manuscript', 'outputs/revision_stage20', 'outputs/revision_stage21', 'outputs/revision_stage22', 'outputs/revision_stage23')) {
  foreach ($pattern in @('*.csv','*.json','*.md')) { Copy-Pattern $dir $pattern }
}
foreach ($name in @('condition_effects.png','condition_effects.svg','paired_roi_scatter.png','paired_roi_scatter.svg')) {
  Copy-Relative "outputs/ieee_figures/$name"
}
Write-Host "Copied $Copied files into $DestRoot"
if ($Missing.Count) {
  Write-Warning "The following REQUIRED files are absent. Do NOT claim the repository is a complete reproduction package yet:"
  $Missing | ForEach-Object { Write-Host "MISSING: $_" }
} else { Write-Host 'All selected essential artifacts found. Additional provenance checks are still recommended.' }
Write-Host 'The original PUMA image/annotation folders were NOT copied.'
