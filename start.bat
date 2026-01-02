@echo off
chcp 65001 >nul
title License Plate Recognition System

echo.
echo ============================================================
echo        License Plate Detection and Recognition System
echo ============================================================
echo.

:: Get script directory
cd /d "%~dp0"

:: Use Python 3.13 from Anaconda
set PYTHON_EXE=C:\Users\guo\anaconda3\python.exe

:: Check Python
echo [1/4] Checking Python...
"%PYTHON_EXE%" --version
if errorlevel 1 (
    echo [ERROR] Python 3.13 not found
    pause
    exit /b 1
)
echo       OK

:: Delete old venv if exists and create new one
echo.
echo [2/4] Setting up virtual environment...
if exist "venv" (
    echo       Removing old environment...
    rmdir /s /q venv
)
echo       Creating new environment with Python 3.13...
"%PYTHON_EXE%" -m venv venv
echo       OK

:: Activate virtual environment
call venv\Scripts\activate.bat

:: Install dependencies
echo.
echo [3/4] Installing dependencies (this may take a few minutes)...
python -m pip install --upgrade pip -q

echo       Installing core packages...
pip install numpy opencv-python Pillow -q

echo       Installing web framework...
pip install flask flask-cors werkzeug -q

echo       Installing deep learning packages...
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu -q
pip install ultralytics -q

echo       Installing OCR...
pip install easyocr -q

echo       OK - All dependencies installed

:: Start server
echo.
echo [4/4] Starting server...
echo.
echo ============================================================
echo   Open browser: http://localhost:5000
echo   Press Ctrl+C to stop
echo ============================================================
echo.

python run.py

echo.
echo Server stopped
pause
