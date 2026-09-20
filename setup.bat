@echo off
setlocal
cd /d "%~dp0"

echo === H3 Prompt Studio - setup local ===

where uv >nul 2>&1
if %errorlevel%==0 goto :uv

echo uv no encontrado. Usando Python 3.12...
py -3.12 -m venv .venv
if errorlevel 1 goto :error
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
python -m pip install torch==2.14.0 torchvision --index-url https://download.pytorch.org/whl/cu130
python -m pip install -r requirements.txt
goto :check

:uv
echo uv detectado.
uv venv --python 3.12 .venv
if errorlevel 1 goto :error
uv pip install --python .venv\Scripts\python.exe torch==2.14.0 torchvision --index-url https://download.pytorch.org/whl/cu130
if errorlevel 1 goto :error
uv pip install --python .venv\Scripts\python.exe -r requirements.txt
if errorlevel 1 goto :error

:check
.venv\Scripts\python.exe check_gpu.py
if errorlevel 1 goto :error

echo.
echo Setup terminado. Copia Qwen3-VL-4B-Instruct dentro de:
echo models\Qwen3-VL-4B-Instruct\
echo.
echo Luego ejecuta run.bat
pause
exit /b 0

:error
echo.
echo ERROR durante la instalacion.
pause
exit /b 1
