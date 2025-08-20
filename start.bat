@echo off

:: Creates a venv if it doesn't exist and runs the start script for requirements upgrades
:: This is intended for users who want to start the API and have everything upgraded and installed

cd "%~dp0"

where uv >nul 2>&1
if %errorlevel% equ 0 (
    echo "HAS UV"
    set HAS_UV=1
) else (
    set HAS_UV=0
)

:: Don't create a venv if a conda environment is active
if exist "%CONDA_PREFIX%" (
    echo It looks like you're in a conda environment. Skipping venv check.
) else (
    if not exist "venv\" (
        echo Venv doesn't exist! Creating one for you.

        if %HAS_UV% equ 1 (
            echo "It looks like you're using uv. Running appropriate commands."
            uv venv venv -p 3.12
        ) else (
            python -m venv venv
        )

        if exist "start_options.json" (
            echo Removing old start_options.json
            del start_options.json
        )
    )

    call .\venv\Scripts\activate.bat
)

:: Call the python script with batch args
call python start.py %*

pause
