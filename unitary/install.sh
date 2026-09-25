#!/bin/sh
# macOS / Linux: find a Python 3.11+ and hand over to bootstrap.py (venv + requirements).
# Then start the UI with:  python3 bootstrap.py run
cd "$(dirname "$0")" || exit 1
for py in python3.13 python3.12 python3.11 python3; do
  if command -v "$py" >/dev/null 2>&1 && "$py" -c 'import sys; sys.exit(sys.version_info < (3, 11))'; then
    exec "$py" bootstrap.py "$@"
  fi
done
echo "Python 3.11 or newer is needed: https://www.python.org/downloads/ (or: brew install python / apt install python3)" >&2
exit 1
