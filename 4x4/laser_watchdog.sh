#!/usr/bin/env bash
# Turn the laser off if it is lit and nothing is using it.
#
# The session watchdog in pic.session only covers a laser this stack turned on; a diode lit
# by hand, or left behind by a killed process, has nothing watching it. This is the backstop
# for that case: every PERIOD it asks whether any rig command is running, and if not, powers
# the laser down. It never touches a laser that is already off, and never interrupts a run.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PORT="${LASER_PORT:-/dev/cu.usbserial-AU05XLI8}"
PERIOD="${PERIOD:-600}"

in_use() {   # anything holding a bench serial port?
  # Ask the OS who has the ports open, rather than pattern-matching command lines. A list of
  # known commands misses anything run outside it -- a scratchpad script, an interactive
  # session -- and this watchdog then powers the laser down in the middle of a live run,
  # which is worse than the unattended diode it exists to prevent.
  lsof /dev/cu.usbmodem* /dev/cu.usbserial-* 2>/dev/null | grep -q . && return 0
  return 1
}

while true; do
  sleep "$PERIOD"
  if in_use; then continue; fi
  st="$("$ROOT/do" laser --port "$PORT" status 2>/dev/null | awk '/master output enable/ {print $NF}')"
  if [ "$st" = "1" ]; then
    echo "[$(date +%H:%M:%S)] laser lit with no run in progress -- powering off"
    "$ROOT/do" laser --port "$PORT" hw 0 2>&1 | tail -1
  fi
done
