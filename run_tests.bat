@echo off
setlocal
cd /d "%~dp0"

echo === H3 Prompt Studio - tests ===
echo.

set FAIL=0

echo [1/5] Presets y SceneSpec
.venv\Scripts\python.exe test_scene.py
if errorlevel 1 set FAIL=1
echo.

echo [2/5] Compilador de prompt H3 (T2VA + Ref2VA)
.venv\Scripts\python.exe test_compiler.py
if errorlevel 1 set FAIL=1
echo.

echo [3/5] Interprete (normalizacion y continuidad)
.venv\Scripts\python.exe test_interpreter.py
if errorlevel 1 set FAIL=1
echo.

echo [4/5] Runtime H3: cuantizacion INT8 + ConvRot
.venv\Scripts\python.exe test_quant.py
if errorlevel 1 set FAIL=1
echo.

echo [5/5] Runtime H3: lector de checkpoint
.venv\Scripts\python.exe test_checkpoint.py
if errorlevel 1 set FAIL=1
echo.

if %FAIL%==1 (
    echo === ALGUNOS TESTS FALLARON ===
    pause
    exit /b 1
)

echo === TODOS LOS TESTS PASARON ===
pause
exit /b 0
