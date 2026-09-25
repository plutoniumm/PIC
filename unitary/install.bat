@echo off
rem Windows: find a Python 3.11+ and hand over to bootstrap.py (venv + requirements).
rem Then start the UI with:  py bootstrap.py run   (or double-click run.bat)
cd /d "%~dp0"
where py >nul 2>nul && (
  py -3 -c "import sys; sys.exit(sys.version_info < (3, 11))" && ( py -3 bootstrap.py %* & goto :eof )
)
where python >nul 2>nul && (
  python -c "import sys; sys.exit(sys.version_info < (3, 11))" && ( python bootstrap.py %* & goto :eof )
)
echo Python 3.11 or newer is needed: https://www.python.org/downloads/ (tick "Add to PATH")
exit /b 1
