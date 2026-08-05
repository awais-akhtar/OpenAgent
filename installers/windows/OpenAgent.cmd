@echo off
setlocal
set "OPENAGENT_INSTALL=%LOCALAPPDATA%\OpenAgent"
set "OPENAGENT_DESKTOP=%OPENAGENT_INSTALL%\.venv\Scripts\openagent-desktop.exe"
set "OPENAGENT_DATA=%OPENAGENT_INSTALL%\data"

if exist "%OPENAGENT_DESKTOP%" (
  "%OPENAGENT_DESKTOP%" --data-dir "%OPENAGENT_DATA%" %*
  exit /b %ERRORLEVEL%
)

python -m openagent.desktop --data-dir "%OPENAGENT_DATA%" %*
