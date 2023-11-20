@echo off

set VENV_DIR=

if not defined PYTHON (set PYTHON=python)
if not defined VENV_DIR (set "VENV_DIR=%~dp0%venv")

call "%VENV_DIR%\Scripts\activate.bat"
call pip -V
call python main.py
