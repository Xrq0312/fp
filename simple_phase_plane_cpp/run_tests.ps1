param(
    [string]$Exe = ".\build\Release\simple_phase_plane_unwrap.exe"
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$resultRoot = Join-Path $PSScriptRoot "test_results"
New-Item -ItemType Directory -Force -Path $resultRoot | Out-Null

$data1Phase = (Get-ChildItem -LiteralPath (Join-Path $root "data_phase_unwrap") `
    -Filter "PHASEX.bmp" -File -Recurse | Select-Object -First 1).FullName

$tests = @(
    @{ Name="data2_raw"; Input="data_phase_unwrap\data2\RAW"; Roi=@(100,1200,4701,3648) },
    @{ Name="data1_phasex"; Input=$data1Phase; Roi=@(0,0,2560,2560) },
    @{ Name="t2_set1_x"; Input="data_phase_unwrap\t2\DataExport_outputPhaseXImage.bmp"; Roi=@(600,700,7601,4701) },
    @{ Name="t2_set1_y"; Input="data_phase_unwrap\t2\DataExport_outputPhaseYImage.bmp"; Roi=@(600,700,7601,4701) },
    @{ Name="t2_set2_x"; Input="data_phase_unwrap\t2\DataExport_outputPhaseXImage_2.bmp"; Roi=@(600,700,7601,4701) },
    @{ Name="t2_set2_y"; Input="data_phase_unwrap\t2\DataExport_outputPhaseYImage_2.bmp"; Roi=@(600,700,7601,4701) }
)

$summary = @()
foreach ($test in $tests) {
    $inputPath = if ([System.IO.Path]::IsPathRooted($test.Input)) {
        $test.Input
    } else {
        Join-Path $root $test.Input
    }
    $outputPath = Join-Path $resultRoot $test.Name
    if (Test-Path -LiteralPath $outputPath) {
        Remove-Item -LiteralPath $outputPath -Recurse -Force
    }
    & $Exe $inputPath $outputPath @($test.Roi)
    if ($LASTEXITCODE -ne 0) {
        throw "Test failed: $($test.Name)"
    }
    $values = @{}
    Get-Content -LiteralPath (Join-Path $outputPath "report.txt") | ForEach-Object {
        $parts = $_ -split "=", 2
        if ($parts.Count -eq 2) { $values[$parts[0]] = $parts[1] }
    }
    $summary += [PSCustomObject]@{
        dataset = $test.Name
        roi_bounds = $values.roi_bounds_x0_y0_x1_y1
        roi_size = $values.roi_size_width_height
        fit_samples = $values.fit_samples
        guide_dx = $values.guide_dx
        guide_dy = $values.guide_dy
        plane_a_b_c = $values.plane_a_b_c
        residual_mean = $values.residual_mean
        residual_rms = $values.residual_rms
    }
}

$summary | Export-Csv -LiteralPath (Join-Path $resultRoot "summary.csv") -NoTypeInformation -Encoding UTF8
