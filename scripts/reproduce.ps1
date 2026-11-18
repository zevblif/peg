#Requires -Version 5.1
# Full reproduction: data subsets -> index -> update -> PEG states -> evaluation
# -> zero-generation supplements. Generation-level and rebuild-variance runs are
# separate because each spends a large number of chat calls (see README).
# Set the environment variables described in README §2 before running.
param(
    [string]$Python = $(if ($env:PEG_PYTHON) { $env:PEG_PYTHON } else { "python" }),
    [string]$Runs = "artifacts/runs"
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$env:PYTHONPATH = "src"

foreach ($name in "PEG_CHAT_MODEL", "PEG_CHAT_API_BASE", "PEG_CHAT_API_KEY") {
    if (-not (Get-Item "env:$name" -ErrorAction SilentlyContinue)) {
        throw "missing environment variable $name (see README §2)"
    }
}

function Invoke-Step([string]$label, [scriptblock]$body) {
    Write-Host "== $label"
    & $body
    if ($LASTEXITCODE -ne 0) { throw "$label failed with exit code $LASTEXITCODE" }
}

Invoke-Step "build subsets" {
    & $Python -m peg_bench.build_hotpot
    & $Python -m peg_bench.build_stability
    & $Python -m peg_bench.make_inputs
}

Invoke-Step "index base (300 paragraphs)" {
    & $Python -m peg_bench.patch_settings --root "$Runs/base" --csv "data/main/base.csv"
    & $Python -m peg_bench.run_graphrag index --root "$Runs/base"
}

Invoke-Step "update with delta (100 paragraphs)" {
    & $Python -m peg_bench.patch_settings --root "$Runs/update" --csv "data/main/input_base_delta.csv"
    & $Python -m peg_bench.run_graphrag update --root "$Runs/update"
}

Invoke-Step "update with edits and deletions" {
    & $Python -m peg_bench.patch_settings --root "$Runs/edit" --csv "data/main/input_after_edit.csv"
    & $Python -m peg_bench.run_graphrag update --root "$Runs/edit"
}

Invoke-Step "build PEG states" {
    & $Python -m peg_bench.build_peg_state
}

Invoke-Step "evaluate" {
    & $Python -m peg_bench.evaluate `
        --base-root "$Runs/base" --update-root "$Runs/update" --edit-root "$Runs/edit" `
        --peg-state "artifacts/states/peg_base_delta" --peg-behavior-state "artifacts/states/peg_edit_delete" `
        --output "results/peg_gr_001.json"
}

Invoke-Step "evaluate host arms on the complete-vector-store instance" {
    & $Python -m peg_bench.evaluate `
        --base-root "$Runs/base" --update-root "$Runs/edit" --edit-root "$Runs/edit" `
        --peg-state "artifacts/states/peg_base_delta" --peg-behavior-state "artifacts/states/peg_edit_delete" `
        --output "results/peg_gr_002.json"
}

Invoke-Step "controls, coverage, intervals, entity mapping" {
    & $Python -m peg_bench.controls --paragraph-results "results/peg_gr_002.json"
}

Invoke-Step "baselines" {
    & $Python -m peg_bench.baselines
}

Invoke-Step "locality and latency" {
    & $Python -m peg_bench.local_effects
}

Write-Host "done: results/peg_gr_001.json, peg_gr_002.json, controls.json, baselines.json, local_effects.json"
Write-Host "separate: peg_bench.generation (about 140 chat calls); peg_bench.index_variance (3 rebuilds)"
