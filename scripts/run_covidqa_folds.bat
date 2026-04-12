@echo off
REM ============================================================
REM  Run 5-fold COVID-QA training for a single model+variant.
REM
REM  Usage:
REM    scripts\run_covidqa_folds.bat --model bert
REM    scripts\run_covidqa_folds.bat --model bert --pretrained_from results/squad/bert/best_model
REM
REM  This will run folds 0-4 sequentially and print a summary.
REM ============================================================

setlocal enabledelayedexpansion

REM --- Parse arguments ---
set "MODEL="
set "PRETRAINED="
set "EXTRA_ARGS="

:parse_args
if "%~1"=="" goto :done_args
if /I "%~1"=="--model" (
    set "MODEL=%~2"
    shift & shift
    goto :parse_args
)
if /I "%~1"=="--pretrained_from" (
    set "PRETRAINED=%~2"
    set "EXTRA_ARGS=--pretrained_from %~2"
    shift & shift
    goto :parse_args
)
shift
goto :parse_args
:done_args

if "%MODEL%"=="" (
    echo ERROR: --model is required
    echo Usage: scripts\run_covidqa_folds.bat --model bert [--pretrained_from path]
    exit /b 1
)

REM --- Summary ---
echo ============================================================
echo  COVID-QA 5-fold CV
echo  Model:          %MODEL%
if defined PRETRAINED (
    echo  Pretrained from: %PRETRAINED%
    echo  Variant:         z SQuAD
) else (
    echo  Variant:         bez SQuAD
)
echo ============================================================
echo.

REM --- Run folds ---
set "FAILED=0"
for %%F in (0 1 2 3 4) do (
    echo [%%F/4] Starting fold %%F ...
    python -u src/train.py --model %MODEL% --dataset covidqa --fold %%F %EXTRA_ARGS%
    if !errorlevel! neq 0 (
        echo [%%F/4] ERROR: Fold %%F failed with exit code !errorlevel!
        set "FAILED=1"
    ) else (
        echo [%%F/4] Fold %%F completed successfully.
    )
    echo.
)

REM --- Summary ---
echo ============================================================
if "%FAILED%"=="1" (
    echo  DONE with errors. Check logs above.
) else (
    echo  ALL 5 FOLDS COMPLETED SUCCESSFULLY.
)
echo ============================================================

endlocal
