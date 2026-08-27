#!/usr/bin/env bash
# ./do -- tiny task dispatcher for this repo.
#
#   ./do laser state 1     # power ON  (state is an alias for hw)
#   ./do laser state 0     # power OFF (ramp down, then disable)
#   ./do laser set 10      # ramp to +10 dBm output
#   ./do laser set 5 -t 30 # ramp to +5 dBm, hold 30 s, then auto power-off
#   ./do laser status      # full device readout
#   ./do measure           # live dashboard: scrolling PD reads, pinned max / avg-of-10 / max-avg
#   ./do measure out.csv   # ... or append the raw stream as CSV to a file
#   ./do heaters           # rapid drift re-anchor: re-find every heater's null, update
#                          #   pic_data/pic_b_config.json in place (laser on); --limit N = subset
#   ./do heaters schedule  # build the confound-free parallel-sweep schedule (no hardware)
#                          #   -> pic_data/parallel_schedule.json  (--only-knobs, thresholds)
#   ./do heaters parallel --channels 85,109,0,1  # validated lockstep parallel sweep (laser on)
#
# Picks the first python that can import the needed modules; override with PYTHON=/path/to/python.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

pick_python() {  # arg: python test snippet the interpreter must satisfy (default: import serial)
  local test="${1:-import serial}" cands=() p
  [ -n "${PYTHON:-}" ] && cands+=("$PYTHON")
  cands+=(python3 python /usr/local/Caskroom/miniconda/base/envs/pic/bin/python)
  for p in "${cands[@]}"; do
    if command -v "$p" >/dev/null 2>&1 && "$p" -c "$test" >/dev/null 2>&1; then
      echo "$p"; return 0
    fi
  done
  return 1
}

usage() {
  sed -n '2,15p' "$ROOT/do" | sed 's/^# \{0,1\}//'
  exit "${1:-2}"
}

case "${1:-}" in
  laser)
    shift
    sub="${1:-}"; [ $# -gt 0 ] && shift
    [ "$sub" = state ] && sub=hw   # `state` is just a nicer alias for `hw`
    [ -z "$sub" ] && usage
    PY="$(pick_python 'import serial')" || {
      echo "no python with pyserial found -- run: pip install -r laser/requirements.txt" >&2
      exit 3
    }
    exec env PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" "$PY" -m laser "$sub" "$@"
    ;;
  measure)
    shift
    PY="$(pick_python 'import serial, numpy')" || {
      echo "no python with pyserial+numpy found -- use the 'pic' conda env" >&2
      exit 3
    }
    exec env PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" "$PY" -m src.pic measure "$@"
    ;;
  heaters)
    shift
    sub=reanchor   # default; a leading flag (e.g. --limit) also falls through to reanchor
    case "${1:-}" in schedule|parallel|reanchor) sub="$1"; shift ;; esac
    case "$sub" in
      schedule)  # confound-free list-coloring scheduler -> pic_data/parallel_schedule.json (no hw)
        PY="$(pick_python 'import json')" || { echo "no python found" >&2; exit 3; }
        exec env PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" "$PY" \
          "$ROOT/scripts/parallel_groups.py" "$@"
        ;;
      parallel)  # validated lockstep parallel sweep (laser on). Needs --channels.
        # TODO: parallel_char validates a channel list against solo sweeps; a full runner that
        # consumes pic_data/parallel_schedule.json round-by-round and merges the recovered nulls
        # into pic_b_config.json is not written yet. Pass --channels explicitly for now.
        PY="$(pick_python 'import serial, numpy, scipy')" || {
          echo "no python with pyserial+numpy+scipy found -- use the 'pic' conda env" >&2; exit 3; }
        exec env PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" "$PY" \
          "$ROOT/scripts/parallel_char.py" "$@"
        ;;
      reanchor)  # re-anchor each characterized heater's null and write Vnull/V0 back to the config.
        # All-zero baseline (matches how most nulls were measured); add
        # `--base-map pic_data/dac_heater_map_b128.csv` to light-route the dim ones. --limit N = subset.
        PY="$(pick_python 'import serial, numpy, scipy')" || {
          echo "no python with pyserial+numpy+scipy found -- use the 'pic' conda env" >&2; exit 3; }
        exec env PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" "$PY" \
          "$ROOT/scripts/reanchor.py" --update "$@"
        ;;
    esac
    ;;
  ""|-h|--help|help) usage 0 ;;
  *) echo "./do: unknown command '$1'" >&2; usage ;;
esac
