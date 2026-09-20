@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Falta .venv. Ejecuta setup.bat primero.
  pause
  exit /b 1
)
.venv\Scripts\python.exe app.py
pause
