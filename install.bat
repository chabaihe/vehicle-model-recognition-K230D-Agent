@echo off
REM ============================================================
REM  Vehicle Recognition Project - one-click installer (main env)
REM
REM  RULES FOR EDITING THIS FILE:
REM    1. ASCII only. Chinese chars break cmd on GBK systems
REM       (code page 936) because cmd parses the whole file first.
REM    2. Keep CRLF line endings. LF-only files make cmd eat the
REM       first character of every line.
REM ============================================================
setlocal enabledelayedexpansion
cd /d "%~dp0"
title Vehicle Recognition - Environment Installer

echo ============================================================
echo   Vehicle Recognition - Main Environment Installer
echo ============================================================
echo.

REM ---------- [1/5] locate Python ----------
set "PY="
where python >nul 2>nul && set "PY=python"
if not defined PY (
    if exist "%LOCALAPPDATA%\Programs\Python\Python312\python.exe" (
        set "PY=%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
    )
)
if not defined PY (
    echo [ERROR] Python not found on PATH.
    echo.
    echo   Please install Python 3.12 from:
    echo     https://www.python.org/downloads/
    echo.
    echo   On the first install screen, tick:
    echo     [x] Add python.exe to PATH
    echo.
    pause
    exit /b 1
)

set "PYVER=unknown"
"%PY%" -c "import sys;print(sys.version.split()[0])" > "%TEMP%\vr_pyver.txt" 2>nul
if exist "%TEMP%\vr_pyver.txt" set /p PYVER=<"%TEMP%\vr_pyver.txt"
del "%TEMP%\vr_pyver.txt" >nul 2>nul

echo [1/5] Python found
echo       path    : %PY%
echo       version : %PYVER%
echo.

REM ---------- [2/5] version check ----------
"%PY%" -c "import sys; sys.exit(0 if sys.version_info[:2] >= (3,9) else 1)" >nul 2>nul
if errorlevel 1 (
    echo [ERROR] Python is too old. Need 3.9+, 3.12 recommended.
    echo         Current version: %PYVER%
    pause
    exit /b 1
)
echo [2/5] Python version check passed
echo.

REM ---------- [3/5] PyTorch (GPU build) ----------
set "TORCH_OK=0"
"%PY%" -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" >nul 2>nul
if not errorlevel 1 set "TORCH_OK=1"

if "!TORCH_OK!"=="1" (
    echo [3/5] PyTorch with CUDA already available - skipping download.
) else (
    echo [3/5] PyTorch GPU build not found - installing now.
    echo.
    where nvidia-smi >nul 2>nul
    if errorlevel 1 (
        echo       WARNING: no NVIDIA GPU detected.
        echo                Installing CPU-only PyTorch.
        echo                Training will be VERY slow on CPU.
        echo.
        "%PY%" -m pip install torch torchvision
    ) else (
        echo       NVIDIA GPU detected. Installing CUDA 12.6 build...
        echo       Download size is about 2.5 GB, please wait 5-15 minutes.
        echo.
        "%PY%" -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
        if errorlevel 1 (
            echo.
            echo       cu126 build failed, retrying with CUDA 12.1 build...
            "%PY%" -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
        )
    )
)
echo.

REM ---------- [4/5] requirements.txt ----------
echo [4/5] Installing remaining dependencies from requirements.txt ...
echo.
"%PY%" -m pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
if errorlevel 1 (
    echo.
    echo [WARN] Tsinghua mirror failed, retrying with default PyPI index...
    "%PY%" -m pip install -r requirements.txt
    if errorlevel 1 (
        echo.
        echo [ERROR] Dependency installation failed.
        echo         Check your network, then run this file again.
        pause
        exit /b 1
    )
)
echo.

REM ---------- [5/5] verify ----------
echo [5/5] Verifying installation...
echo.
"%PY%" check_env.py
if errorlevel 1 (
    echo.
    echo [WARN] Some required packages are still missing, see list above.
    pause
    exit /b 1
)

echo.
echo ============================================================
echo   DONE. Main environment is ready.
echo ============================================================
echo.
echo   Next steps:
echo     1. Prepare dataset  - see the reproduction guide, Step 2
echo     2. Train the model  - run train_car1000.py in the training folder
echo     3. Try the web Agent - double-click start.bat in the agent folder
echo.
echo   For the K230 conversion environment, run: install-k230.bat
echo.
pause
endlocal
exit /b 0
