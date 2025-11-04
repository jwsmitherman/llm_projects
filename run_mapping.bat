@echo off
setlocal enableextensions

REM ============================================
REM Usage:
REM run_mapping.bat ISSUER PAYCODE TRANDATE LOAD_TASK_ID COMPANY_ISSUER_ID CSV_PATH TEMPLATE_DIR
REM Example:
REM run_mapping.bat "Manhattan Life" "Default" "2025-11-03" "13449" "2204" "C:\path\file.csv" "C:\path\templates"
REM ============================================

if "%~7"=="" (
  echo Usage: ISSUER PAYCODE TRANDATE LOAD_TASK_ID COMPANY_ISSUER_ID CSV_PATH TEMPLATE_DIR
  exit /b 2
)

REM Positional arguments
set "ISSUER=%~1"
set "PAYCODE=%~2"
set "TRANDATE=%~3"
set "LOAD_TASK_ID=%~4"
set "COMPANY_ISSUER_ID=%~5"
set "CSV_PATH=%~6"
set "TEMPLATE_DIR=%~7"

REM Set working directory
set "SCRIPT_DIR=%~dp0"
pushd "%SCRIPT_DIR%"

REM Python interpreter
set "PYTHON_EXE=%LocalAppData%\Programs\Python\Python313\python.exe"
if not exist "%PYTHON_EXE%" (
  echo Warning: Preferred Python not found at %PYTHON_EXE%. Falling back to system Python.
  set "PYTHON_EXE=python"
)

REM Azure OpenAI environment variables (optional)
REM set "AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com"
REM set "AZURE_OPENAI_API_KEY=your_api_key"
REM set "AZURE_OPENAI_API_VERSION=2024-02-15-preview"
REM set "AZURE_OPENAI_DEPLOYMENT=gpt-4o-mini"

REM Performance and output configuration
set "ENABLE_RAY=auto"
set "RAY_PARTITIONS=8"
set "RAY_MIN_ROWS_TO_USE=300000"
set "OUT_DIR=%SCRIPT_DIR%outbound"
if not exist "%OUT_DIR%" mkdir "%OUT_DIR%"
set "OUT_FORMAT=csv"
set "PARQUET_COMPRESSION=snappy"

REM Display configuration info
echo Script directory: %SCRIPT_DIR%
echo Using Python: %PYTHON_EXE%
echo Current directory: %CD%
echo Output directory: %OUT_DIR%
echo Output format: %OUT_FORMAT%
echo Ray settings: ENABLE_RAY=%ENABLE_RAY% PARTITIONS=%RAY_PARTITIONS% MIN_ROWS=%RAY_MIN_ROWS_TO_USE%
echo Azure OpenAI: Endpoint=%AZURE_OPENAI_ENDPOINT% Deployment=%AZURE_OPENAI_DEPLOYMENT%

REM Run the LLM pipeline
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
  echo Error: Pipeline failed with exit code %EXITCODE%.
  popd
  endlocal
  exit /b %EXITCODE%
)

echo Done. Outputs are located in:
echo %OUT_DIR%

popd
endlocal
