#!/usr/bin/env bash
# ./do -- tiny task dispatcher for this repo.
#
#   ./do laser state 1     # power ON  (state is an alias for hw)
#   ./do laser state 0     # power OFF (ramp down, then disable)
#   ./do laser set 10      # ramp to +10 dBm output
#   ./do laser status      # full device readout
#
# Picks the first python that can `import serial`; override with PYTHON=/path/to/python.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

pick_python() {
  local cands=() p
  [ -n "${PYTHON:-}" ] && cands+=("$PYTHON")
  cands+=(python3 python /usr/local/Caskroom/miniconda/base/envs/pic/bin/python)
  for p in "${cands[@]}"; do
    if command -v "$p" >/dev/null 2>&1 && "$p" -c 'import serial' >/dev/null 2>&1; then
      echo "$p"; return 0
    fi
  done
  return 1
}

usage() {
  sed -n '2,9p' "$ROOT/do" | sed 's/^# \{0,1\}//'
  exit "${1:-2}"
}

case "${1:-}" in
  laser)
    shift
    sub="${1:-}"; [ $# -gt 0 ] && shift
    [ "$sub" = state ] && sub=hw   # `state` is just a nicer alias for `hw`
    [ -z "$sub" ] && usage
    PY="$(pick_python)" || {
      echo "no python with pyserial found -- run: pip install -r laser/requirements.txt" >&2
      exit 3
    }
    exec env PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" "$PY" -m laser "$sub" "$@"
    ;;
  ""|-h|--help|help) usage 0 ;;
  *) echo "./do: unknown command '$1'" >&2; usage ;;
esac
