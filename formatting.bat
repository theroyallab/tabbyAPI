@echo off
setlocal

REM Change to script's directory
cd /d %~dp0

REM Get tool versions
for /f "tokens=2" %%i in ('yapf --version') do set YAPF_VERSION=%%i
for /f "tokens=2" %%i in ('pylint --version') do set PYLINT_VERSION=%%i
for /f "tokens=2" %%i in ('mypy --version') do set MYPY_VERSION=%%i

REM Check tool versions
call :tool_version_check "yapf" %YAPF_VERSION% "required_version"
call :tool_version_check "pylint" %PYLINT_VERSION% "required_version"
call :tool_version_check "mypy" %MYPY_VERSION% "required_version"

REM Format files
if "%1"=="--files" (
    shift
    yapf --in-place --recursive --parallel %*
) else if "%1"=="--all" (
    for /r %%i in (*.py) do yapf --in-place --recursive --parallel "%%i"
) else (
    REM Format only the files that changed in last commit.
    REM This part is tricky in Batch and might require additional tools like Git for Windows.
)

REM Run Pylint
for /r %%i in (*.py) do pylint "%%i"

REM Check if any files were changed
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
