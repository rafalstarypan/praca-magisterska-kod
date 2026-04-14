@echo off
REM ============================================================
REM  E2: LoRA fine-tuning runs
REM
REM  Usage:
REM    scripts\run_e2_lora.bat --model pubmedbert
REM    scripts\run_e2_lora.bat --model pubmedbert --lora_r 4
REM
REM  Runs LoRA fine-tuning on BioASQ + COVID-QA (5-fold) for
REM  the given model. Assumes SQuAD-pretrained model exists.
REM ============================================================

setlocal enabledelayedexpansion

REM --- Parse arguments ---
set "MODEL="
set "LORA_R_ARG="

:parse_args
if "%~1"=="" goto :done_args
if /I "%~1"=="--model" (
    set "MODEL=%~2"
    shift & shift
    goto :parse_args
)
if /I "%~1"=="--lora_r" (
    set "LORA_R_ARG=--lora_r %~2"
    shift & shift
    goto :parse_args
)
shift
goto :parse_args
:done_args

if "%MODEL%"=="" (
    echo ERROR: --model is required
    echo Usage: scripts\run_e2_lora.bat --model pubmedbert [--lora_r 4]
    exit /b 1
)

set "PRETRAINED=results/squad/%MODEL%/best_model"

if not exist "%PRETRAINED%" (
    echo ERROR: Pretrained model not found: %PRETRAINED%
    exit /b 1
)

REM --- Summary ---
echo ============================================================
echo  E2: LoRA fine-tuning
echo  Model:          %MODEL%
echo  Pretrained from: %PRETRAINED%
if defined LORA_R_ARG (
    echo  LoRA rank:      custom %LORA_R_ARG%
) else (
    echo  LoRA rank:      default (from lora.yaml)
)
echo ============================================================
echo.

set "FAILED=0"

REM --- BioASQ ---
echo [1/6] BioASQ LoRA ...
python -u src/train.py --model %MODEL% --dataset bioasq --lora %LORA_R_ARG% --pretrained_from %PRETRAINED%
if !errorlevel! neq 0 (
    echo [1/6] ERROR: BioASQ failed!
    set "FAILED=1"
) else (
    echo [1/6] BioASQ completed.
)
echo.

REM --- COVID-QA 5 folds ---
for %%F in (0 1 2 3 4) do (
    set /a "STEP=%%F+2"
    echo [!STEP!/6] COVID-QA fold %%F ...
    python -u src/train.py --model %MODEL% --dataset covidqa --fold %%F --lora %LORA_R_ARG% --pretrained_from %PRETRAINED%
    if !errorlevel! neq 0 (
        echo [!STEP!/6] ERROR: Fold %%F failed!
        set "FAILED=1"
    ) else (
        echo [!STEP!/6] Fold %%F completed.
    )
    echo.
)

REM --- Summary ---
echo ============================================================
if "%FAILED%"=="1" (
    echo  DONE with errors. Check logs above.
) else (
    echo  ALL 6 RUNS COMPLETED SUCCESSFULLY.
)
echo ============================================================

endlocal
