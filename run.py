"""
python run.py <file.pic> [flags]
       --mock  : No HW
       --dry   : Parse and plan only
       --no-csv: No CSV
       --port=/dev/cu.usbserial-XXX (optional)

# script 1: ramp every heater, indented Pythonic body, pulse 10x every 4s (~40s)
for H in heaters:
    V = 0.05 * H

pulse every 4s
repeat 10 times
name ramp
---

# script 2: a few hand-set channels (blank lines and loose spacing are fine)
V_5  =  2 V
V_9  =  4

loop  = 2
iters = 5
name handset
---

# script 3: inline for still works; sweep channel 5 -> one CSV per value
for H in 0..5: V = 1.0

sweep V_5 from 0 to 4 step 0.5
repeat 3 times
loop = 1
name s5
done
"""

import sys

from picscript import main

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
