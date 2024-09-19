@echo off

:: check if less is installed
:: run scoop install less
:: https://scoop.sh/#/apps?q=less&id=e084d861765203aae2d64ada4e59ef350df0f25b
where less >nul 2>&1
if %errorlevel%==0 (
    mypy start.py | less
) else (
    mypy start.py
)
