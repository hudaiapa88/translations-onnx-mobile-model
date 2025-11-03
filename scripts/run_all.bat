@echo off
REM ============================================================================
REM Multi-Language ONNX Translation Models Setup
REM Automated pipeline for downloading, converting, and optimizing 42 models
REM ============================================================================

echo.
echo ========================================================================
echo   Multi-Language ONNX Translation Models Setup
echo   Processing 42 language pairs: tr, en, de, fr, it, pt, es
echo ========================================================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not installed or not in PATH
    echo Please install Python 3.8 or higher
    pause
    exit /b 1
)

echo [INFO] Python found
echo.

REM Navigate to scripts directory
cd /d "%~dp0"

REM Step 0: Install requirements
echo ========================================================================
echo STEP 0: Installing Python Dependencies
echo ========================================================================
echo.

if not exist requirements.txt (
    echo [ERROR] requirements.txt not found
    pause
    exit /b 1
)

echo Installing required packages...
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

if errorlevel 1 (
    echo [ERROR] Failed to install dependencies
    pause
    exit /b 1
)

echo.
echo [SUCCESS] Dependencies installed
echo.

REM Step 1: Show language configuration
echo ========================================================================
echo STEP 1: Language Configuration
echo ========================================================================
echo.

python language_config.py

if errorlevel 1 (
    echo [ERROR] Language configuration check failed
    pause
    exit /b 1
)

echo.
pause
echo.

REM Step 2: Download and convert all models
echo ========================================================================
echo STEP 2: Downloading and Converting Models
echo ========================================================================
echo.
echo This will take several hours (estimated 4-6 hours)
echo PyTorch models will be automatically deleted after conversion
echo.
echo Press CTRL+C to cancel, or
pause
echo.

python download_all_languages.py

if errorlevel 1 (
    echo [ERROR] Download/conversion failed
    echo Check download_log.json for progress
    pause
    exit /b 1
)

echo.
echo [SUCCESS] All models downloaded and converted
echo.

REM Step 3: Optimize all models
echo ========================================================================
echo STEP 3: Optimizing Models
echo ========================================================================
echo.
echo Applying additional optimizations for size reduction...
echo.

python optimize_all_models.py

if errorlevel 1 (
    echo [WARNING] Optimization had issues, but models should still work
)

echo.
echo [SUCCESS] Optimization complete
echo.

REM Step 4: Test all models
echo ========================================================================
echo STEP 4: Testing Models
echo ========================================================================
echo.
echo Testing all 42 language pairs...
echo.

python test_all_models.py

if errorlevel 1 (
    echo [WARNING] Some models failed testing
    echo Review test_results.json for details
) else (
    echo [SUCCESS] All models passed testing!
)

echo.

REM Final summary
echo ========================================================================
echo SETUP COMPLETE!
echo ========================================================================
echo.
echo Models location: ..\onnx_models\
echo.
echo Next steps:
echo   1. Review test_results.json for model status
echo   2. Upload to Hugging Face for CDN distribution
echo   3. Use in your Flutter app with onnx_translation package
echo.
echo Hugging Face Upload Command:
echo   huggingface-cli login
echo   huggingface-cli upload [username]/translation-models ./onnx_models .
echo.
echo ========================================================================
echo.

pause
