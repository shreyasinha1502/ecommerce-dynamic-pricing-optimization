@echo off
setlocal

echo Setting up project environment...
if not exist ".venv\Scripts\python.exe" (
    python -m venv .venv
)

call .venv\Scripts\activate.bat

echo Installing dependencies...
pip install -r requirements.txt

echo Running dynamic pricing project...
python src\dynamic_pricing_project.py

echo.
echo Project finished. Check outputs\figures and outputs\tables
pause
