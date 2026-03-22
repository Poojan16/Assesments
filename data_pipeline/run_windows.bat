@echo off
set PY=C:\Users\Disha\AppData\Local\Programs\Python\Python311\python.exe
set ROOT=c:\Users\Disha\Desktop\Pujan's Work\data_pipeline_service\data_pipeline

echo [1/6] Creating virtual environment...
%PY% -m venv "%ROOT%\venv"
if errorlevel 1 (echo FAILED: venv creation & exit /b 1)

echo [2/6] Upgrading pip...
"%ROOT%\venv\Scripts\python.exe" -m pip install --upgrade pip --quiet
if errorlevel 1 (echo FAILED: pip upgrade & exit /b 1)

echo [3/6] Installing requirements...
"%ROOT%\venv\Scripts\pip.exe" install -r "%ROOT%\requirements.txt"
if errorlevel 1 (echo FAILED: pip install & exit /b 1)

echo [4/6] Copying .env...
if not exist "%ROOT%\.env" copy "%ROOT%\.env.example" "%ROOT%\.env"

echo [5/6] Running Django migrations...
"%ROOT%\venv\Scripts\python.exe" "%ROOT%\dashboard\manage.py" migrate
if errorlevel 1 (echo FAILED: migrate & exit /b 1)

echo [6/6] Generating sample data...
"%ROOT%\venv\Scripts\python.exe" "%ROOT%\generate_data.py" --rows 5000
if errorlevel 1 (echo FAILED: generate_data & exit /b 1)

echo.
echo === Setup complete ===
echo To start Django: "%ROOT%\venv\Scripts\python.exe" "%ROOT%\dashboard\manage.py" runserver
echo To run tests:    "%ROOT%\venv\Scripts\pytest.exe" "%ROOT%\tests"
