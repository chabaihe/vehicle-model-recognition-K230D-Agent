@echo off
REM ============================================================
REM  Vehicle Recognition Project - one-click installer for the
REM  K230 model-conversion environment (nncase)
REM
REM  Usage:
REM      install-k230.bat
REM      install-k230.bat [path to python 3.10-3.12]
REM      install-k230.bat [path to python] [venv dir]
REM
REM  IMPORTANT - WHY THE VENV LIVES OUTSIDE THE REPO:
REM    nncase is a .NET library and its hostfxr loader cannot
REM    handle non-ASCII paths. If the venv sits under a folder
REM    whose name contains Chinese characters, importing nncase
REM    fails with:
REM        RuntimeError: Failed to initialize hostfxr: 0x80008094
REM    So the venv defaults to %LOCALAPPDATA%\vr_nncase_env,
REM    which is normally ASCII-only.
REM
REM  RULES FOR EDITING THIS FILE:
REM    1. ASCII only - GBK code page breaks on Chinese chars.
REM    2. Keep CRLF line endings.
REM    3. Never put these characters inside an echo line:
REM         < > ( ) | & ^
REM       Inside an if-block a closing paren ends the block early,
REM       and the angle brackets are treated as redirection.
REM ============================================================
setlocal enabledelayedexpansion
cd /d "%~dp0"
title Vehicle Recognition - K230 Conversion Environment

echo ============================================================
echo   K230 Conversion Environment Installer for nncase
echo ============================================================
echo.

REM ---------- step 1 of 6: locate a Python 3.10-3.12 ----------
set "PY=%~1"

if not defined PY (
    for %%P in (
        "%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
        "%LOCALAPPDATA%\Programs\Python\Python311\python.exe"
        "%LOCALAPPDATA%\Programs\Python\Python310\python.exe"
        "C:\Python312\python.exe"
        "C:\Python311\python.exe"
        "C:\Python310\python.exe"
    ) do (
        if not defined PY if exist %%P set "PY=%%~P"
    )
)
if not defined PY (
    where python >nul 2>nul && set "PY=python"
)

if not defined PY (
    echo [ERROR] Python not found.
    echo         Install Python 3.12 from:
    echo           https://www.python.org/downloads/
    echo.
    pause
    exit /b 1
)

set "PYVER=unknown"
"%PY%" -c "import sys;print(sys.version.split()[0])" > "%TEMP%\vr_pyver2.txt" 2>nul
if exist "%TEMP%\vr_pyver2.txt" set /p PYVER=<"%TEMP%\vr_pyver2.txt"
del "%TEMP%\vr_pyver2.txt" >nul 2>nul

echo [1/6] Candidate Python: %PY%
echo       version: %PYVER%
echo.

REM ---------- step 2 of 6: version MUST be 3.10 - 3.12 ----------
"%PY%" -c "import sys; sys.exit(0 if (3,10) <= sys.version_info[:2] <= (3,12) else 1)" >nul 2>nul
if errorlevel 1 (
    echo [ERROR] Python version must be 3.10 - 3.12 for nncase 2.11.0
    echo         Current version: %PYVER%
    echo.
    echo   Why: nncase 2.11.0 only publishes wheels up to cp312.
    echo        Python 3.13 and 3.14 cannot install it at all.
    echo.
    echo   How to fix:
    echo     1. Install Python 3.12 from python.org
    echo     2. Run this file again and pass the full path, e.g.
    echo        install-k230.bat D:\Python312\python.exe
    echo.
    pause
    exit /b 1
)
echo [2/6] Python version OK
echo.

REM ---------- step 3 of 6: choose an ASCII-only venv path ----------
set "VENV=%~2"
if not defined VENV set "VENV=%LOCALAPPDATA%\vr_nncase_env"

"%PY%" -c "import sys;sys.exit(0 if sys.argv[1].isascii() else 1)" "!VENV!"
if errorlevel 1 (
    echo [WARN] This path contains non-ASCII characters:
    echo        !VENV!
    echo        nncase uses .NET and CANNOT load from such a path.
    echo        Falling back to: C:\vr_nncase_env
    set "VENV=C:\vr_nncase_env"
)
echo [3/6] Venv location: !VENV!
echo.

REM ---------- step 4 of 6: create isolated venv ----------
if exist "!VENV!\Scripts\python.exe" (
    echo [4/6] Reusing existing venv.
) else (
    echo [4/6] Creating isolated venv...
    "%PY%" -m venv "!VENV!"
    if errorlevel 1 (
        echo [ERROR] Failed to create virtual environment at:
        echo         !VENV!
        echo         Run this file again with a writable ASCII path
        echo         as the second argument.
        pause
        exit /b 1
    )
)
set "VPY=!VENV!\Scripts\python.exe"
echo       venv python: !VPY!
echo.

REM ---------- step 5 of 6: install requirements-k230.txt ----------
echo [5/6] Installing nncase 2.11.0 plus nncase-kpu 2.11.0 ...
echo.
echo       NOTE: nncase-kpu has NO Windows wheel on PyPI, so
echo             requirements-k230.txt points at the official
echo             GitHub release wheel directly. That is expected.
echo.
"!VPY!" -m pip install -r requirements-k230.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
if errorlevel 1 (
    echo.
    echo [WARN] Mirror install failed, retrying with default PyPI...
    "!VPY!" -m pip install -r requirements-k230.txt
    if errorlevel 1 (
        echo.
        echo [ERROR] nncase installation failed. Possible causes:
        echo           - no network access to github.com
        echo           - Python version outside 3.10-3.12
        echo.
        echo         You can also download these two wheels manually
        echo         from the nncase GitHub Releases page, v2.11.0:
        echo           nncase-2.11.0-cp312-cp312-win_amd64.whl
        echo           nncase_kpu-2.11.0-py2.py3-none-win_amd64.whl
        echo         Then install them with:
        echo           pip install path\to\nncase-...whl
        echo           pip install path\to\nncase_kpu-...whl
        pause
        exit /b 1
    )
)
echo.

REM ---------- step 6 of 6: verify KPU plugin ----------
echo [6/6] Verifying nncase and KPU plugin ...
echo.
"!VPY!" check_env.py --k230
if errorlevel 1 (
    echo.
    echo [WARN] K230 environment is incomplete, see list above.
    echo.
    echo   If you see "Failed to initialize hostfxr", the venv path
    echo   almost certainly contains Chinese characters. Re-run with
    echo   an ASCII-only path, for example:
    echo     install-k230.bat D:\Python312\python.exe C:\vr_nncase_env
    echo.
    pause
    exit /b 1
)

echo.
echo ============================================================
echo   DONE. K230 conversion environment is ready.
echo ============================================================
echo.
echo   Venv path: !VENV!
echo.
echo   Option A - activate the venv first:
echo     "!VENV!\Scripts\activate.bat"
echo     python convert_to_kmodel_car1000.py --help
echo.
echo   Option B - call the venv python directly:
echo     "!VPY!" convert_to_kmodel_car1000.py --help
echo.
echo   NOTE: run the conversion script inside the K230 deployment
echo         code folder, or pass full paths to it.
echo.
pause
endlocal
exit /b 0
