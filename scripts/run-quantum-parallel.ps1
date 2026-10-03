param(
    [string]$Python = "python",
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$TrainingArgs
)
$ErrorActionPreference = "Stop"
$QuantumProjectRoot = Split-Path -Parent $PSScriptRoot
$env:PYTHONPATH = Join-Path $QuantumProjectRoot "src"
$env:PYTHONUTF8 = "1"
Push-Location $QuantumProjectRoot
try {
    & $Python -m quantum_models.parallel_dense20 @TrainingArgs
    if ($LASTEXITCODE -ne 0) { throw "Quantum training exited with code $LASTEXITCODE" }
}
finally {
    Pop-Location
}
