@echo off
REM ============================================================
REM  E3: Re-run LoRA merged benchmarks with more warmup/runs
REM  Fixes: initial warmup before measurement loop, 10 runs
REM ============================================================

setlocal enabledelayedexpansion

set "FAILED=0"

echo [1/4] PubMedBERT LoRA GPU ...
python -u src/benchmark.py --model_path results/bioasq/pubmedbert_lora_r8/best_model --tokenizer pubmedbert --device cuda --lora --base_model_path results/squad/pubmedbert/best_model --label pubmedbert_lora_r8 --n_warmup 30 --n_runs 10
if !errorlevel! neq 0 ( echo ERROR & set "FAILED=1" )
echo.

echo [2/4] RoBERTa LoRA GPU ...
python -u src/benchmark.py --model_path results/bioasq/roberta_lora_r8/best_model --tokenizer roberta --device cuda --lora --base_model_path results/squad/roberta/best_model --label roberta_lora_r8 --n_warmup 30 --n_runs 10
if !errorlevel! neq 0 ( echo ERROR & set "FAILED=1" )
echo.

echo [3/4] PubMedBERT LoRA CPU ...
python -u src/benchmark.py --model_path results/bioasq/pubmedbert_lora_r8/best_model --tokenizer pubmedbert --device cpu --lora --base_model_path results/squad/pubmedbert/best_model --label pubmedbert_lora_r8 --n_warmup 30 --n_runs 10
if !errorlevel! neq 0 ( echo ERROR & set "FAILED=1" )
echo.

echo [4/4] RoBERTa LoRA CPU ...
python -u src/benchmark.py --model_path results/bioasq/roberta_lora_r8/best_model --tokenizer roberta --device cpu --lora --base_model_path results/squad/roberta/best_model --label roberta_lora_r8 --n_warmup 30 --n_runs 10
if !errorlevel! neq 0 ( echo ERROR & set "FAILED=1" )
echo.

echo [AGG] Aggregating...
python -u scripts/aggregate_e3.py

echo ============================================================
if "%FAILED%"=="1" ( echo DONE with errors. ) else ( echo ALL 4 BENCHMARKS COMPLETED. )
echo ============================================================

endlocal
