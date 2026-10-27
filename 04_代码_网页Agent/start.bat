@echo off
REM ============================================================
REM  Vehicle Recognition Agent - launcher
REM
REM  RULES FOR EDITING THIS FILE:
REM    1. ASCII only. Chinese chars break cmd on GBK systems
REM       (code page 936) because cmd parses the whole file first.
REM    2. Keep CRLF line endings. LF-only makes cmd eat the first
REM       character of every line.
REM    3. Never put < > ( ) | & ^ inside an echo line.
REM ============================================================
setlocal enabledelayedexpansion
cd /d "%~dp0"

echo ============================================================
echo   Vehicle Recognition Agent - starting...
echo ============================================================
echo.

REM ---------- locate Python ----------
REM Priority: VR_PYTHON env var  >  local venv  >  PATH python
set "PY=%VR_PYTHON%"
if defined PY if not exist "%PY%" set "PY="

if not defined PY (
    if exist "..\.venv\Scripts\python.exe" set "PY=..\.venv\Scripts\python.exe"
)
if not defined PY (
    where python >nul 2>nul && set "PY=python"
)

if not defined PY (
    echo [ERROR] Python not found.
    echo.
    echo   Options:
    echo     1. Install Python 3.12 from python.org and tick
    echo        "Add python.exe to PATH"
    echo     2. Set the VR_PYTHON environment variable to your
    echo        python.exe full path, for example:
    echo          set VR_PYTHON=D:\Python312\python.exe
    echo.
    pause
    exit /b 1
)

set "PYVER=unknown"
"%PY%" -c "import sys;print(sys.version.split()[0])" > "%TEMP%\vr_agent_pyver.txt" 2>nul
if exist "%TEMP%\vr_agent_pyver.txt" set /p PYVER=<"%TEMP%\vr_agent_pyver.txt"
del "%TEMP%\vr_agent_pyver.txt" >nul 2>nul

echo [1/4] Python: %PY%
echo       version: %PYVER%
echo.

set "PIPOPT=-i https://pypi.tuna.tsinghua.edu.cn/simple --quiet"

echo [2/4] Checking dependencies...
"%PY%" -c "import onnxruntime" 1>nul 2>nul
if errorlevel 1 (
  echo       installing onnxruntime ...
  "%PY%" -m pip install onnxruntime %PIPOPT%
)

"%PY%" -c "import openai" 1>nul 2>nul
if errorlevel 1 (
  echo       installing openai ...
  "%PY%" -m pip install openai %PIPOPT%
)

"%PY%" -c "import tavily" 1>nul 2>nul
if errorlevel 1 (
  echo       installing tavily-python ...
  "%PY%" -m pip install tavily-python %PIPOPT%
)

"%PY%" -c "import PIL" 1>nul 2>nul
if errorlevel 1 (
  echo       installing pillow ...
  "%PY%" -m pip install pillow %PIPOPT%
)

"%PY%" -c "import numpy" 1>nul 2>nul
if errorlevel 1 (
  echo       installing numpy ...
  "%PY%" -m pip install numpy %PIPOPT%
)
echo       done.
echo.

echo [3/4] Checking config and model...
"%PY%" -c "import sys; sys.path.insert(0,'.'); import config_loader as c; print('  DeepSeek key :', 'OK' if c.has_llm() else 'MISSING'); print('  Tavily key   :', 'OK' if c.has_search() else 'MISSING'); print('  Local model  :', 'OK' if c.ONNX_PATH.exists() else 'MISSING')"
echo.

echo [4/4] Starting web server...
echo.
echo   Browser will open automatically.
echo   Press Ctrl+C or close this window to stop.
echo ============================================================
echo.

"%PY%" -u server.py

echo.
echo Server stopped.
pause
endlocal
