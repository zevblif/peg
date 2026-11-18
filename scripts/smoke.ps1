#Requires -Version 5.1
# Two-document smoke test: confirms the chat endpoint, the embedding endpoint and
# the pinned GraphRAG configuration work together before spending on a full run.
param(
    [string]$Python = $(if ($env:PEG_PYTHON) { $env:PEG_PYTHON } else { "python" }),
    [string]$Root = "artifacts/runs/smoke"
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$env:PYTHONPATH = "src"

foreach ($name in "PEG_CHAT_MODEL", "PEG_CHAT_API_BASE", "PEG_CHAT_API_KEY") {
    if (-not (Get-Item "env:$name" -ErrorAction SilentlyContinue)) {
        throw "missing environment variable $name (see README §3)"
    }
}

& $Python -m peg_bench.patch_settings --root $Root --csv "data/main/smoke.csv"
if ($LASTEXITCODE -ne 0) { throw "patch_settings failed" }

& $Python -m peg_bench.run_graphrag index --root $Root
if ($LASTEXITCODE -ne 0) { throw "smoke index failed" }

Write-Host "smoke ok: $Root/output/community_reports.parquet"
