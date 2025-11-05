@echo off
setlocal ENABLEDELAYEDEXPANSION

REM Change to the folder of this .bat
cd /d "%~dp0"

set FILE=coffee_machine_app_fixed.py

if not exist "%FILE%" (
  echo Could not find %FILE% in %cd% 
  echo Make sure coffee_machine_app_fixed.py is in the same folder as this .bat file.
  pause
  exit /b 1
)

echo Starting Coffee Machine server...
echo (This window will show logs; close it to stop the server.)
echo.

REM Try py launcher with Python 3 first
where py >nul 2>nul
if %errorlevel%==0 (
  py -3 "%FILE%"
  if %errorlevel%==0 goto :end
  py "%FILE%"
  if %errorlevel%==0 goto :end
)

REM Fall back to python, then python3
where python >nul 2>nul
if %errorlevel%==0 (
  python "%FILE%"
  if %errorlevel%==0 goto :end
)

where python3 >nul 2>nul
if %errorlevel%==0 (
  python3 "%FILE%"
  if %errorlevel%==0 goto :end
)

echo.
echo Could not run Python automatically.
echo Please install Python from https://www.python.org/downloads/ and ensure it is on your PATH.
echo Then double-click this file again.
echo.

:end
echo.
echo Server has stopped or exited.
pause
