@echo off

set VENV_DIR=
set REQUIREMENTS_FILE=

if not defined PYTHON (set PYTHON=python)
if not defined VENV_DIR (set "VENV_DIR=%~dp0%venv")
if not defined REQUIREMENTS_FILE (set "REQUIREMENTS_FILE=requirements-nowheel.txt")

call "%VENV_DIR%\Scripts\activate.bat"
call pip -V
if NOT [%1] == [--ignore-upgrade] call pip install --upgrade -r %REQUIREMENTS_FILE%
call python main.py
