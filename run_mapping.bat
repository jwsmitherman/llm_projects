@echo off
setlocal enableextensions

REM ============================================================
REM Usage:
REM   run_mapping.bat <ISSUER> <PAYCODE> <TRANDATE YYYY-MM-DD> <LOAD_TASK_ID> <COMPANY_ISSUER_ID> <CSV_PATH> <TEMPLATE_DIR>
REM Example:
REM   run_mapping.bat "Manhattan Life" "Default" "2025-11-03" "13449" "2204" "C:\path\file.csv" "C:\path\templates"
REM ============================================================

if "%~7"=="" (
  echo Usage: ^<ISSUER^> ^<PAYCODE^> ^<TRANDATE YYYY-MM-DD^> ^<LOAD_TASK_ID^> ^<COMPANY_ISSUER_ID^> ^<CSV_PATH^> ^<TEMPLATE_DIR^>
  exit /b 2
)

REM ------------------ Positional args ------------------
set "ISSUER=%~1"
set "PAYCODE=%~2"
set "TRANDATE=%~3"
set "LOAD_TASK_ID=%~4"
set "COMPANY_ISSUER_ID=%~5"
set "CSV_PATH=%~6"
set "TEMPLATE_DIR=%~7"

REM ------------------ Stable working dir ------------------
set "SCRIPT_DIR=%~dp0"
pushd "%SCRIPT_DIR%"

REM ------------------ Python interpreter ------------------
set "PYTHON_EXE=%LocalAppData%\Programs\Python\Python313\python.exe"
if not exist "%PYTHON_EXE%" (
  echo [WARN] Preferred Python not found at "%PYTHON_EXE%". Falling back to PATH.
  set "PYTHON_EXE=python"
)

REM ============================================================
REM New / optional: Azure OpenAI (uncomment and set if not using system env)
REM set "AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com"
REM set "AZURE_OPENAI_API_KEY=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
REM set "AZURE_OPENAI_API_VERSION=2024-02-15-preview"
REM set "AZURE_OPENAI_DEPLOYMENT=gpt-4o-mini"
REM ============================================================

REM ============================================================
REM New fields: performance & output config (read by Python)
REM (keep these values or change as needed)
REM ============================================================
set "ENABLE_RAY=auto"              REM auto | on | off
set "RAY_PARTITIONS=8"
set "RAY_MIN_ROWS_TO_USE=300000"

REM Absolute OUT_DIR so Explorer and Python agree
set "OUT_DIR=%SCRIPT_DIR%outbound"
if not exist "%OUT_DIR%" mkdir "%OUT_DIR%"

set "OUT_FORMAT=csv"               REM csv | parquet
set "PARQUET_COMPRESSION=snappy"   REM if OUT_FORMAT=parquet

REM ------------------ Info banner ------------------
echo [INFO] Script dir : "%SCRIPT_DIR%"
echo [INFO] Using Python: "%PYTHON_EXE%"
echo [INFO] CWD        : "%CD%"
echo [INFO] OUT_DIR    : "%OUT_DIR%"
echo [INFO] OUT_FORMAT : "%OUT_FORMAT%"
echo [INFO] Ray        : ENABLE_RAY=%ENABLE_RAY% PARTS=%RAY_PARTITIONS% MIN_ROWS=%RAY_MIN_ROWS_TO_USE%
echo [INFO] Azure OAI  : endpoint=%AZURE_OPENAI_ENDPOINT% dep=%AZURE_OPENAI_DEPLOYMENT%

REM ------------------ Run pipeline ------------------
echo Running LLM pipeline...
"%PYTHON_EXE%" "%SCRIPT_DIR%cli_runner.py" ^
  --issuer "%ISSUER%" ^
  --paycode "%PAYCODE%" ^
  --trandate "%TRANDATE%" ^
  --load_task_id "%LOAD_TASK_ID%" ^
  --company_issuer_id "%COMPANY_ISSUER_ID%" ^
  --csv_path "%CSV_PATH%" ^
  --template_dir "%TEMPLATE_DIR%"

set "EXITCODE=%ERRORLEVEL%"

if not "%EXITCODE%"=="0" (
  echo [ERROR] Failed with exit code %EXITCODE%.
  popd
  endlocal
  exit /b %EXITCODE%
)

echo [INFO] Done. Outputs should be under:
echo        "%OUT_DIR%"

popd
endlocal
