@echo off

set VENV_DIR=
:: Requirements file to use. Defaults to nowheel to avoid mis-installation of dependencies
set REQUIREMENTS_FILE=

if not defined PYTHON (set PYTHON=python)
if not defined VENV_DIR (set "VENV_DIR=%~dp0%venv")
:: Doesn't update wheels by default
if not defined REQUIREMENTS_FILE (set "REQUIREMENTS_FILE=requirements-nowheel.txt")

if not exist %VENV_DIR%\ (
    echo "Please create a venv and install dependencies before starting TabbyAPI! Exiting..."
    exit
)

:: Argument parsing
for %%A in (%*) do (
    if %%A == "--ignore-upgrade" set IGNORE_UPGRADE=y
)

call "%VENV_DIR%\Scripts\activate.bat"
call pip -V
if defined IGNORE_UPGRADE call pip install --upgrade -r %REQUIREMENTS_FILE%
call python main.py
