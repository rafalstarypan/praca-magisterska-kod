@echo off
REM ============================================================
REM  E3: Inference benchmarking — all models, GPU + CPU
REM
REM  Usage:
REM    scripts\run_e3.bat
REM
REM  Benchmarks 8 model configs on GPU and CPU (BioASQ test set).
REM  Results saved to results/e3_benchmark/*.json
REM ============================================================

setlocal enabledelayedexpansion

set "FAILED=0"
set "STEP=0"
set "TOTAL=16"

REM --- 6 Full FT models on GPU ---
for %%M in (bert biobert pubmedbert roberta distilbert clinicalbert) do (
    set /a "STEP+=1"
    echo [!STEP!/%TOTAL%] %%M GPU ...
    python -u src/benchmark.py --model_path results/bioasq/%%M/best_model --tokenizer %%M --device cuda --label %%M
    if !errorlevel! neq 0 (
        echo [!STEP!/%TOTAL%] ERROR: %%M GPU failed!
        set "FAILED=1"
    )
    echo.
)

REM --- 2 LoRA models on GPU (merged) ---
for %%M in (pubmedbert roberta) do (
    set /a "STEP+=1"
    echo [!STEP!/%TOTAL%] %%M LoRA GPU ...
    python -u src/benchmark.py --model_path results/bioasq/%%M_lora_r8/best_model --tokenizer %%M --device cuda --lora --base_model_path results/squad/%%M/best_model --label %%M_lora_r8
    if !errorlevel! neq 0 (
        echo [!STEP!/%TOTAL%] ERROR: %%M LoRA GPU failed!
        set "FAILED=1"
    )
    echo.
)

REM --- 6 Full FT models on CPU ---
for %%M in (bert biobert pubmedbert roberta distilbert clinicalbert) do (
    set /a "STEP+=1"
    echo [!STEP!/%TOTAL%] %%M CPU ...
    python -u src/benchmark.py --model_path results/bioasq/%%M/best_model --tokenizer %%M --device cpu --label %%M
    if !errorlevel! neq 0 (
        echo [!STEP!/%TOTAL%] ERROR: %%M CPU failed!
        set "FAILED=1"
    )
    echo.
)

REM --- 2 LoRA models on CPU (merged) ---
for %%M in (pubmedbert roberta) do (
    set /a "STEP+=1"
    echo [!STEP!/%TOTAL%] %%M LoRA CPU ...
    python -u src/benchmark.py --model_path results/bioasq/%%M_lora_r8/best_model --tokenizer %%M --device cpu --lora --base_model_path results/squad/%%M/best_model --label %%M_lora_r8
    if !errorlevel! neq 0 (
        echo [!STEP!/%TOTAL%] ERROR: %%M LoRA CPU failed!
        set "FAILED=1"
    )
    echo.
)

REM --- Aggregate ---
echo [AGG] Aggregating results...
python -u scripts/aggregate_e3.py
echo.

REM --- Summary ---
echo ============================================================
if "%FAILED%"=="1" (
    echo  DONE with errors. Check logs above.
) else (
    echo  ALL 16 BENCHMARKS COMPLETED SUCCESSFULLY.
)
echo ============================================================

endlocal
