@echo off
setlocal

REM Usage (PowerShell):
REM   .\run_process_refactored_llm.bat ".\inbound\" "manhattan_life_raw_data.csv" "2025-09-17T14:06:00" "AWM01" "manhattan_life"

REM ------- Args -------
set "INBOUND=%~1"
set "FILENAME=%~2"
set "TRANDATE=%~3"
set "PAYCODE=%~4"
set "ISSUER=%~5"

REM ------- Basic validation -------
if "%INBOUND%"==""  (echo Missing arg 1: INBOUND (file-location) & exit /b 1)
if "%FILENAME%"=="" (echo Missing arg 2: FILENAME (file-name) & exit /b 1)
if "%TRANDATE%"=="" (echo Missing arg 3: TRANDATE & exit /b 1)
if "%PAYCODE%"==""  (echo Missing arg 4: PAYCODE & exit /b 1)
if "%ISSUER%"==""   (echo Missing arg 5: ISSUER & exit /b 1)

REM Path of this script (so we can locate the .py next to it)
set "SCRIPT_DIR=%~dp0"
set PYTHONUNBUFFERED=1

REM ------- Single-line python call (no ^ continuations) -------
python "%SCRIPT_DIR%1_process_statements_process_llm.py" --file-location "%INBOUND%" --file-name "%FILENAME%" --trandate "%TRANDATE%" --paycode "%PAYCODE%" --issuer "%ISSUER%" -v

set ERR=%ERRORLEVEL%
echo Done.
exit /b %ERR%
