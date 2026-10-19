@echo off
setlocal
set "OPENAGENT_INSTALL=%LOCALAPPDATA%\OpenAgent"
set "OPENAGENT_DESKTOP=%OPENAGENT_INSTALL%\.venv\Scripts\openagent-desktop.exe"
set "OPENAGENT_DATA=%OPENAGENT_INSTALL%\data"
set "OPENAGENT_RUNTIME_DIR=D:\openagent\.runtime-desktop"
rem Prefer the checked-out source when it is present. This keeps the desktop
rem launcher aligned with the current application instead of an older wheel.
if exist "D:\openagent\openagent\desktop.py" if exist "D:\openagent\.venv\Scripts\python.exe" (
  "D:\openagent\.venv\Scripts\python.exe" -m openagent.desktop --data-dir "D:\openagent\data" %*
  exit /b %ERRORLEVEL%
)

if exist "%OPENAGENT_DESKTOP%" (
  "%OPENAGENT_DESKTOP%" --data-dir "%OPENAGENT_DATA%" %*
  exit /b %ERRORLEVEL%
)

python -m openagent.desktop --data-dir "%OPENAGENT_DATA%" %*
