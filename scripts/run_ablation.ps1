param(
    [string[]]$Variants = @("baseline", "multiscale", "joint", "topology"),
    [string[]]$DatasetNames = @(),
    [string]$OutputDir = ".\results\ablation",
    [ValidateSet("global", "adjacency")]
    [string]$CornerMode = "global",
    [switch]$EdgeResidualProposals,
    [switch]$BinaryPlaneOutput,
    [ValidateSet("none", "soft", "hard")]
    [string]$ParallelRelations = "none",
    [switch]$FiniteBoundaryPostprocess,
    [switch]$FiniteSupportRegionOptimization,
    [ValidateSet("pca", "pca-vcm-consensus")]
    [string]$RegionNormalEstimator = "pca",
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
$steppedFixture = Join-Path $fixtureDir "stepped_box.ply"
$crossingFixture = Join-Path $fixtureDir "crossing_planes.ply"
$nearParallelFixture = Join-Path $fixtureDir "near_parallel_planes.ply"
$python = Get-ChildItem "$env:LOCALAPPDATA\Programs\Python\Python*\python.exe" -ErrorAction SilentlyContinue |
    Select-Object -First 1 -ExpandProperty FullName
if (-not $python) {
    throw "A real Python installation is required; the Windows Store alias is not sufficient."
}
& $python .\tests\generate_stepped_box.py $steppedFixture
& $python .\tests\generate_crossing_planes.py $crossingFixture
& $python .\tests\generate_near_parallel_planes.py $nearParallelFixture

$datasets = @(
    [pscustomobject]@{ Name = "stepped_box"; Path = $steppedFixture; Profile = "model-mm"; Synthetic = $true },
    [pscustomobject]@{ Name = "crossing_planes"; Path = $crossingFixture; Profile = "model-mm"; Synthetic = $true },
    [pscustomobject]@{ Name = "near_parallel_planes"; Path = $nearParallelFixture; Profile = "model-mm"; Synthetic = $true }
)
if ($IncludeLargeDatasets) {
    $datasets += @(
        [pscustomobject]@{ Name = "top_left_cut"; Path = "..\data\data01\data01\top_left_cut.ply"; Profile = "model-mm"; Synthetic = $false },
        [pscustomobject]@{ Name = "top_left_cut2"; Path = "..\data\data01\data01\top_left_cut2.ply"; Profile = "top-left-cut2"; Synthetic = $false },
        [pscustomobject]@{ Name = "top_down"; Path = "..\data\data02\data02\top_down.ply"; Profile = "top-down"; Synthetic = $false },
        [pscustomobject]@{ Name = "top_down2"; Path = "..\data\data02\data02\top_down2.ply"; Profile = "top-down"; Synthetic = $false }
    )
}
if ($DatasetNames.Count -gt 0) {
    $datasets = @($datasets | Where-Object { $DatasetNames -contains $_.Name })
    if ($datasets.Count -ne $DatasetNames.Count) {
        throw "One or more requested datasets are unknown: $($DatasetNames -join ', ')"
    }
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
            "--pipeline", $variant,
            "--corner-mode", $CornerMode,
            "--parallel-relations", $ParallelRelations
        )
        if ($EdgeResidualProposals) {
            $arguments += "--edge-residual-proposals"
        }
        if ($BinaryPlaneOutput -and -not $dataset.Synthetic) {
            $arguments += "--binary-plane-output"
        }
        if ($FiniteBoundaryPostprocess) {
            $arguments += "--finite-boundary-postprocess"
        }
        if ($FiniteSupportRegionOptimization) {
            $arguments += "--finite-support-region-optimization"
            $arguments += @("--region-normal-estimator", $RegionNormalEstimator)
        }
        $watch = [Diagnostics.Stopwatch]::StartNew()
        $output = & .\build\Release\edge_aware_plane_extract.exe @arguments 2>&1
        $exitCode = $LASTEXITCODE
        $watch.Stop()
        $output | Set-Content -Encoding utf8 (Join-Path $runDir "run.log")
        if ($exitCode -ne 0) {
            throw "$variant/$($dataset.Name) failed with exit code $exitCode`n$output"
        }

        $quality = $null
        $stepQuality = $null
        $stepCornerQuality = $null
        $stepInstanceQuality = $null
        $geometryQuality = $null
        if ($dataset.Synthetic) {
            $resultPath = Join-Path $runDir $dataset.Name
            $metricsPath = Join-Path $runDir "metrics.json"
            & $python .\tests\evaluate_stepped_box.py `
                ([IO.Path]::ChangeExtension($dataset.Path, ".labels.csv")) `
                ([IO.Path]::ChangeExtension($dataset.Path, ".ground_truth.json")) `
                $resultPath --json $metricsPath | Out-Null
            $quality = Get-Content $metricsPath -Raw | ConvertFrom-Json
            if ($dataset.Name -eq "near_parallel_planes") {
                $geometryMetricsPath = Join-Path $runDir "geometry_metrics.json"
                & $python .\tests\evaluate_plane_parameters.py `
                    ([IO.Path]::ChangeExtension($dataset.Path, ".ground_truth.json")) `
                    (Join-Path $resultPath "plane_parameters.csv") `
                    --json $geometryMetricsPath | Out-Null
                $geometryQuality = Get-Content $geometryMetricsPath -Raw | ConvertFrom-Json
            }
        } elseif ($dataset.Name -in @("top_left_cut2", "top_down2")) {
            $stepPath = if ($dataset.Name -eq "top_left_cut2") {
                "..\data\data01\data2.STEP"
            } else {
                "..\data\data01\data1.STEP"
            }
            if (Test-Path $stepPath) {
                $resultPath = Join-Path $runDir $dataset.Name
                $stepMetricsPath = Join-Path $runDir "step_metrics.json"
                & $python .\tests\evaluate_step_planes.py `
                    $stepPath `
                    (Join-Path $resultPath "plane_parameters.csv") `
                    --json $stepMetricsPath | Out-Null
                $stepQuality = Get-Content $stepMetricsPath -Raw | ConvertFrom-Json
                $stepCornerMetricsPath = Join-Path $runDir "step_corner_metrics.json"
                & $python .\tests\evaluate_step_corners.py `
                    $stepPath `
                    (Join-Path $resultPath "extracted_corners.ply") `
                    --json $stepCornerMetricsPath | Out-Null
                $stepCornerQuality = Get-Content $stepCornerMetricsPath -Raw | ConvertFrom-Json
                $stepInstanceMetricsPath = Join-Path $runDir "step_instance_metrics.json"
                $cleanPath = $dataset.Path
                $cleanScale = if ($dataset.Name -eq "top_left_cut2") { 1000 } else { 1 }
                & $python .\tests\evaluate_step_instances.py `
                    $stepPath `
                    $cleanPath `
                    $resultPath `
                    --input-scale $cleanScale `
                    --json $stepInstanceMetricsPath | Out-Null
                $stepInstanceQuality = Get-Content $stepInstanceMetricsPath -Raw | ConvertFrom-Json
            }
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
            StepMatchedPatches = if ($stepQuality) { $stepQuality.matched_patches } else { $null }
            StepPatchPrecision = if ($stepQuality) { $stepQuality.patch_precision_proxy } else { $null }
            StepMatchedPoints = if ($stepQuality) { $stepQuality.matched_point_fraction } else { $null }
            StepCornerPrecision = if ($stepCornerQuality) { $stepCornerQuality.corner_precision } else { $null }
            StepCornerRecall = if ($stepCornerQuality) { $stepCornerQuality.corner_recall } else { $null }
            StepCornerF1 = if ($stepCornerQuality) { $stepCornerQuality.corner_f1 } else { $null }
            StepCornerMeanError = if ($stepCornerQuality) { $stepCornerQuality.corner_mean_error } else { $null }
            StepInstancePrecision = if ($stepInstanceQuality) { $stepInstanceQuality.plane_precision } else { $null }
            StepInstanceRecall = if ($stepInstanceQuality) { $stepInstanceQuality.plane_recall } else { $null }
            StepInstanceF1 = if ($stepInstanceQuality) { $stepInstanceQuality.plane_f1 } else { $null }
            StepInstanceMacroIoU = if ($stepInstanceQuality) { $stepInstanceQuality.instance_macro_iou } else { $null }
            StepOversegmented = if ($stepInstanceQuality) { $stepInstanceQuality.oversegmented_truth_instances } else { $null }
            StepUndersegmented = if ($stepInstanceQuality) { $stepInstanceQuality.undersegmented_predictions } else { $null }
            NormalMeanError = if ($geometryQuality) { $geometryQuality.normal_mean_error_deg } else { $null }
            NormalMaxError = if ($geometryQuality) { $geometryQuality.normal_max_error_deg } else { $null }
            PlaneOffsetMeanError = if ($geometryQuality) { $geometryQuality.offset_mean_error } else { $null }
        }
    }
}

$csv = Join-Path $OutputDir "summary.csv"
$rows | Export-Csv -NoTypeInformation -Encoding utf8 $csv
$rows | Format-Table -AutoSize
Write-Host "Ablation summary: $((Resolve-Path $csv).Path)"
