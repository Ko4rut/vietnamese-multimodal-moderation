# Execute from the repo root with:
# .\scripts\run_inference_api.ps1 -ModelsRoot "E:\path\to\experiments"
param(
    [string] $ModelsRoot,
    [string] $ModelDir,
    [string] $DefaultModel,
    [int] $Port = 8000
)

$ErrorActionPreference = "Stop"
if ($ModelsRoot -and $ModelDir) {
    throw "Use either -ModelsRoot or -ModelDir, not both."
}
if ($ModelsRoot) {
    if (-not (Test-Path -Path $ModelsRoot -PathType Container)) {
        throw "ModelsRoot directory does not exist: $ModelsRoot"
    }
    $env:VTC_MODELS_ROOT = (Resolve-Path $ModelsRoot).Path
    Remove-Item Env:VTC_MODEL_DIR -ErrorAction SilentlyContinue
} elseif ($ModelDir) {
    if (-not (Test-Path -Path $ModelDir -PathType Container)) {
        throw "ModelDir directory does not exist: $ModelDir"
    }
    $env:VTC_MODEL_DIR = (Resolve-Path $ModelDir).Path
    Remove-Item Env:VTC_MODELS_ROOT -ErrorAction SilentlyContinue
}
if ($DefaultModel) {
    $env:VTC_DEFAULT_MODEL = $DefaultModel
}
Write-Host "FastAPI docs: http://127.0.0.1:$Port/docs"
fastapi dev --host 127.0.0.1 --port $Port
