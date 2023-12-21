@echo off
setlocal

::Change to script's directory
cd /d %~dp0

::Get tool versions
for /f "tokens=2" %%i in ('ruff --version') do set RUFF_VERSION="%%i"

::Check tool versions
call :tool_version_check "ruff" %RUFF_VERSION% "0.1.9"

::Format and lint files
call ruff format
call ruff check

::Check if any files were changed
git diff --quiet
if errorlevel 1 (
    echo Reformatted files. Please review and stage the changes.
    echo Changes not staged for commit:
    echo.
    git --no-pager diff --name-only
    exit /b 1
)

exit /b 0

:tool_version_check
if not "%2"=="%3" (
    echo Wrong %1 version installed: %3 is required, not %2.
    exit /b 1
)
exit /b 0
