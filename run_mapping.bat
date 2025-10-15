@echo off
setlocal enabledelayedexpansion

REM ============================
REM Usage:
REM   run_mapping.bat <ISSUER> <PAYCODE> <TRANDATE> <CSV_PATH> <TEMPLATE_DIR>
REM Example:
REM   run_mapping.bat molina FromApp 2025-10-01 "C:\data\in.csv" "C:\work\carrier_prompts"
REM ============================

if "%~5"=="" (
  echo Usage: %~nx0 ^<ISSUER^> ^<PAYCODE^> ^<TRANDATE YYYY-MM-DD^> ^<CSV_PATH^> ^<TEMPLATE_DIR^>
  exit /b 2
)

set ISSUER=%~1
set PAYCODE=%~2
set TRANDATE=%~3
set CSV_PATH=%~4
set TEMPLATE_DIR=%~5

REM ---- Optional: set/override Azure OpenAI env here (or do it in System env vars) ----
REM set AZURE_OPENAI_API_KEY=YOUR_KEY
REM set AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
REM set AZURE_OPENAI_DEPLOYMENT=gpt-4o-mini
REM set AZURE_OPENAI_API_VERSION=2024-02-15-preview

REM ---- Optional: performance and output config (matches your processor module) ----
REM set ENABLE_RAY=auto
REM set RAY_PARTITIONS=8
REM set RAY_MIN_ROWS_TO_USE=300000
REM set OUT_DIR=.\outbound
REM set OUT_FORMAT=parquet
REM set PARQUET_COMPRESSION=snappy

echo Running LLM pipeline...
python "%~dp0cli_runner.py" ^
  --issuer "%ISSUER%" ^
  --paycode "%PAYCODE%" ^
  --trandate "%TRANDATE%" ^
  --csv_path "%CSV_PATH%" ^
  --template_dir "%TEMPLATE_DIR%"

set EXITCODE=%ERRORLEVEL%
if %EXITCODE% NEQ 0 (
  echo Failed with exit code %EXITCODE%.
  exit /b %EXITCODE%
)

echo Done.
endlocal
