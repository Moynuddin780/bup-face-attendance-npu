@echo off
echo ============================================================
echo  BUP Face Attendance System - Starting...
echo ============================================================
echo.

REM Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found. Please install Python 3.9+
    pause
    exit /b 1
)

REM Install requirements if needed
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
)

call venv\Scripts\activate.bat

echo Installing/checking requirements...
pip install -r requirements.txt -q

REM Initialize DB if not exists
if not exist "bup_db.json" (
    echo Initializing database...
    python -c "from app import save_db, _default_db; save_db(_default_db()); print('Database created.')"
)

echo.
echo ============================================================
echo  Starting BUP Attendance Web Server...
echo  Open your browser: http://localhost:5000
echo  Default login: superadmin / bup@2024
echo ============================================================
echo.

python app.py

pause
