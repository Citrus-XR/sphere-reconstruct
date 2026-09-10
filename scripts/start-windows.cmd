@echo off
setlocal
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0start-windows.ps1" %*
set "launcher_exit_code=%errorlevel%"
if not "%launcher_exit_code%"=="0" (
    echo.
    echo Sphere Reconstruct startup failed. Exit code: %launcher_exit_code%
    echo See the error above and runtime\logs\launcher-*.log.
)
if not "%SPHERE_LAUNCHER_NO_PAUSE%"=="1" pause
exit /b %launcher_exit_code%
