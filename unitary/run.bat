@echo off
rem Windows: start the UI (runs install first if needed). Close this window or Ctrl-C to stop.
cd /d "%~dp0"
call install.bat run %*
