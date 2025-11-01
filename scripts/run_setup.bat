@echo off
REM ONNX Translation Model Setup Script for Windows
REM Automates the entire model download and optimization process

echo.
echo ╔════════════════════════════════════════════════════════════╗
echo ║  ONNX Translation Model Setup                              ║
echo ║  Lingol Mobile - EN to TR Translation Model                ║
echo ╚════════════════════════════════════════════════════════════╝
echo.

REM Check Python installation
python --version >nul 2>&1
if errorlevel 1 (
    echo ✗ ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8+ from https://www.python.org/downloads/
    pause
    exit /b 1
)

echo ✓ Python detected
echo.

REM Navigate to scripts directory
cd /d "%~dp0"

REM Check if virtual environment exists
if not exist "venv" (
    echo Creating Python virtual environment...
    python -m venv venv
    if errorlevel 1 (
        echo ✗ ERROR: Failed to create virtual environment
        pause
        exit /b 1
    )
    echo ✓ Virtual environment created
) else (
    echo ✓ Virtual environment already exists
)

echo.
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Check if dependencies are installed
if not exist "venv\Lib\site-packages\transformers" (
    echo.
    echo ══════════════════════════════════════════════════════════
    echo  Installing Python dependencies...
    echo  This may take 5-10 minutes on first run
    echo ══════════════════════════════════════════════════════════
    echo.
    
    python -m pip install --upgrade pip
    pip install -r requirements.txt
    
    if errorlevel 1 (
        echo ✗ ERROR: Failed to install dependencies
        pause
        exit /b 1
    )
    
    echo.
    echo ✓ All dependencies installed successfully
) else (
    echo ✓ Dependencies already installed
)

echo.
echo ══════════════════════════════════════════════════════════
echo  Testing Model Availability (Quick Check)
echo ══════════════════════════════════════════════════════════
echo.

python test_models.py

echo.
echo ══════════════════════════════════════════════════════════
echo  Starting Model Download and Conversion
echo  Estimated time: 10-15 minutes
echo  Required space: ~500MB (temp) + ~150MB (final)
echo ══════════════════════════════════════════════════════════
echo.

REM Run download and conversion script
python download_onnx_models.py

if errorlevel 1 (
    echo.
    echo ✗ ERROR: Model setup failed
    pause
    exit /b 1
)

echo.
echo ══════════════════════════════════════════════════════════
echo  Running Advanced Optimizations
echo ══════════════════════════════════════════════════════════
echo.

REM Run optimization script
python optimize_models.py

if errorlevel 1 (
    echo.
    echo ⚠ WARNING: Advanced optimization failed
    echo Model is still usable but may not be fully optimized
)

echo.
echo ╔════════════════════════════════════════════════════════════╗
echo ║  SETUP COMPLETED SUCCESSFULLY!                             ║
echo ╚════════════════════════════════════════════════════════════╝
echo.
echo ✓ Model location: ..\onnx_models\en-tr\
echo ✓ Ready for Flutter integration
echo.
echo Next steps:
echo   1. Add to pubspec.yaml: onnx_translation: ^0.1.2
echo   2. Copy model files to Flutter assets or use CDN
echo   3. See onnx_models\en-tr\README.md for usage
echo.

REM Deactivate virtual environment
call venv\Scripts\deactivate.bat

echo Press any key to exit...
pause >nul
