@echo off

:: Creates a venv if it doesn't exist and runs the start script for requirements upgrades
:: This is intended for users who want to start the API and have everything upgraded and installed

:: cd to the parent directory
cd "%~dp0.."

:: Don't create a venv if a conda environment is active
if exist "%CONDA_PREFIX%" (
    echo It looks like you're in a conda environment. Skipping venv check.
) else (
    if not exist "venv\" (
        echo Venv doesn't exist! Please run start.bat instead.
        exit 0
    )

    call .\venv\Scripts\activate.bat
)

:: Call the python script with batch args
call python start.py --update-deps %*

pause
