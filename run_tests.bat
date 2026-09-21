@echo off
setlocal
cd /d "%~dp0"

echo === H3 Prompt Studio - tests ===
echo.

set FAIL=0

echo [1/3] Presets y SceneSpec
.venv\Scripts\python.exe test_scene.py
if errorlevel 1 set FAIL=1
echo.

echo [2/3] Compilador de prompt H3
.venv\Scripts\python.exe test_compiler.py
if errorlevel 1 set FAIL=1
echo.

echo [3/3] Interprete (normalizacion y continuidad)
.venv\Scripts\python.exe test_interpreter.py
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
