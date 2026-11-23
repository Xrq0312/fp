param(
    [string[]]$Variants = @("baseline", "multiscale", "joint"),
    [string]$OutputDir = ".\results\ablation",
    [switch]$IncludeLargeDatasets,
    [switch]$SkipBuild
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

if (-not $SkipBuild) {
    cmake --preset vcpkg-release
    cmake --build --preset vcpkg-release
}

$fixtureDir = Join-Path $OutputDir "fixtures"
New-Item -ItemType Directory -Force -Path $fixtureDir | Out-Null
$fixture = Join-Path $fixtureDir "stepped_box.ply"
$python = Get-ChildItem "$env:LOCALAPPDATA\Programs\Python\Python*\python.exe" -ErrorAction SilentlyContinue |
    Select-Object -First 1 -ExpandProperty FullName
if (-not $python) {
    throw "A real Python installation is required; the Windows Store alias is not sufficient."
}
& $python .\tests\generate_stepped_box.py $fixture

$datasets = @(
    [pscustomobject]@{ Name = "stepped_box"; Path = $fixture; Profile = "model-mm" }
)
if ($IncludeLargeDatasets) {
    $datasets += @(
        [pscustomobject]@{ Name = "top_down"; Path = "..\data\data02\data02\top_down.ply"; Profile = "top-down" },
        [pscustomobject]@{ Name = "top_left_cut2"; Path = "..\data\data01\data01\top_left_cut2.ply"; Profile = "top-left-cut2" }
    )
}

$rows = @()
foreach ($variant in $Variants) {
    foreach ($dataset in $datasets) {
        $runDir = Join-Path $OutputDir "$variant\$($dataset.Name)"
        New-Item -ItemType Directory -Force -Path $runDir | Out-Null
        $arguments = @(
            $dataset.Path,
            "--profile", $dataset.Profile,
            "--output-dir", $runDir,
            "--pipeline", $variant
        )
        $watch = [Diagnostics.Stopwatch]::StartNew()
        $output = & .\build\Release\edge_aware_plane_extract.exe @arguments 2>&1
        $exitCode = $LASTEXITCODE
        $watch.Stop()
        $output | Set-Content -Encoding utf8 (Join-Path $runDir "run.log")
        if ($exitCode -ne 0) {
            throw "$variant/$($dataset.Name) failed with exit code $exitCode`n$output"
        }

        $quality = $null
        if ($dataset.Name -eq "stepped_box") {
            $resultPath = Join-Path $runDir "stepped_box"
            $metricsPath = Join-Path $runDir "metrics.json"
            & $python .\tests\evaluate_stepped_box.py `
                ([IO.Path]::ChangeExtension($fixture, ".labels.csv")) `
                ([IO.Path]::ChangeExtension($fixture, ".ground_truth.json")) `
                $resultPath --json $metricsPath | Out-Null
            $quality = Get-Content $metricsPath -Raw | ConvertFrom-Json
        }

        $values = @{}
        foreach ($line in $output) {
            if ($line -match '^([^:]+):\s*(.+)$') {
                $values[$matches[1]] = $matches[2]
            }
        }
        $rows += [pscustomobject]@{
            Variant = $variant
            Dataset = $dataset.Name
            Seconds = [math]::Round($watch.Elapsed.TotalSeconds, 3)
            InputPoints = $values["Input points"]
            Surfels = $values["Surfels"]
            PlanePatches = $values["Plane patches"]
            OutputPoints = $values["Output points"]
            Coverage = if ([double]$values["Input points"] -gt 0) {
                [double]$values["Output points"] / [double]$values["Input points"]
            } else { 0.0 }
            Corners = $values["Extracted corners"]
            AssignmentMargin = $values["Assignment margin"]
            PlanePrecision = if ($quality) { $quality.plane_precision } else { $null }
            PlaneRecall = if ($quality) { $quality.plane_recall } else { $null }
            PlaneMacroIoU = if ($quality) { $quality.plane_macro_iou } else { $null }
            OverSegmentation = if ($quality) { $quality.plane_oversegmentation } else { $null }
            PointCoverage = if ($quality) { $quality.point_coverage } else { $null }
            CornerPrecision = if ($quality) { $quality.corner_precision } else { $null }
            CornerRecall = if ($quality) { $quality.corner_recall } else { $null }
            CornerMeanError = if ($quality) { $quality.corner_mean_error } else { $null }
        }
    }
}

$csv = Join-Path $OutputDir "summary.csv"
$rows | Export-Csv -NoTypeInformation -Encoding utf8 $csv
$rows | Format-Table -AutoSize
Write-Host "Ablation summary: $((Resolve-Path $csv).Path)"
