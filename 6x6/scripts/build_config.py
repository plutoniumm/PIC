"""Consolidate everything we know about PIC B into one loadable config JSON.

Reads the per-channel map (pic_data/dac_heater_map_b128.csv, itself built by
scripts/map_b128.py) and folds in the device / firmware / laser / PD metadata, so a
single file is enough to drive the chip:

    import json
    cfg = json.load(open("pic_data/pic_b_config.json"))
    ch = cfg["channels"][7]              # everything about DAC channel 7
    live = [c["dac"] for c in cfg["channels"] if c["elec"] == "ok"]   # drivable set

Provisional -- regenerate after any re-characterization:
    python scripts/build_config.py            # -> pic_data/pic_b_config.json
"""

from __future__ import annotations

import collections
import csv
import json
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MAP = ROOT / "pic_data" / "dac_heater_map_b128.csv"
OUT = ROOT / "pic_data" / "pic_b_config.json"
# freshest 0-4 V census raw files, earliest first (later overrides for re-swept channels).
CENSUS = ["pic_data/census/census_20260730_195023.csv",
          "pic_data/census/census_20260730_203820.csv"]
FOOT_THR_MV = 8.0
NOT_DRIVABLE = ("dead", "short")   # operating points here aren't independently actionable


def num(x):
    if x in ("", None):
        return None
    try:
        return int(x) if float(x) == int(float(x)) and "." not in str(x) else float(x)
    except ValueError:
        return x


def footprint():
    """ch -> {pd: max|delta| mV over the sweep}, from the fresh 07-30 census (not the
    stale 07-23 fringe_fits that predates the CS fix)."""
    fp = {}
    for path in CENSUS:
        p = ROOT / path
        if not p.exists():
            continue
        base = collections.defaultdict(list)
        drv = collections.defaultdict(lambda: collections.defaultdict(list))
        for r in csv.DictReader(p.open()):
            ch = int(r["ch"])
            pds = [float(r[f"pd{i}"]) for i in range(14)]
            (base[ch] if r["phase"] == "base" else drv[ch][float(r["drive_v"])]).append(pds)
        for ch in drv:
            b = np.mean(base[ch], axis=0) if base[ch] else np.zeros(14)
            d = np.max([np.abs(np.mean(drv[ch][v], axis=0) - b) for v in drv[ch]], axis=0)
            fp[ch] = {i: round(float(d[i] * 1000), 1) for i in range(14)}
    return fp


def main():
    rows = [r for r in csv.DictReader(l for l in MAP.open() if not l.startswith("#"))]
    fp = footprint()
    channels = []
    for r in sorted(rows, key=lambda r: int(r["dac"])):
        dac = int(r["dac"])
        elec = r["elec"]
        moved = {p: s for p, s in fp.get(dac, {}).items() if s >= FOOT_THR_MV}
        pds = sorted(moved, key=lambda p: -moved[p])
        # operating points are only actionable on independently drivable channels
        drivable = elec not in NOT_DRIVABLE
        channels.append({
            "dac": dac,
            "chip": int(r["chip"]),
            "header": r["header"],
            "pin": num(r["pin"]),
            "net": num(r["net"]),
            "stage": r["stage"],
            "vendor": [num(r["vendor_a"]), num(r["vendor_b"])],
            "elec": elec,                             # ok | dead | weak | short
            "swing_mV": round(max(moved.values()), 1) if moved else 0.0,  # fresh 07-30
            "pds": pds,
            "V0": num(r["V0"]) if drivable else None,     # max-pass volts (transparent)
            "Vnull": num(r["Vnull"]) if drivable else None,  # null volts (0 light, phi=pi)
            "Vpi": num(r["Vpi"]) if drivable else None,      # sqrt(pi/phi2)
            "fringe_pd": num(r["fringe_pd"]) if drivable else None,
            "fringe_src": r["fringe_src"] if drivable else "",  # census run (thermal frame)
        })

    drivable = [c["dac"] for c in channels if c["elec"] == "ok"]
    characterized = [c["dac"] for c in channels if c["V0"] is not None]

    cfg = {
        "meta": {
            "chip": "PIC B (Section B; A is the mirror twin)",
            "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
            "status": "PROVISIONAL -- schematic + electrical + optical, not fully verified",
            "source_map": str(MAP.relative_to(ROOT)),
            "update": "rebuild with scripts/build_config.py after re-characterization; "
                      "re-anchor drift with a 1-point null-find warm-started from V0/Vnull",
        },
        "devices": {
            "pic_port": "/dev/cu.usbserial-1110",   # CH340; may re-enumerate -- set per session
            "laser_port": "/dev/cu.usbserial-AU05XLI8",  # FTDI
            "baud": 115200,
        },
        "firmware": {
            "sketch": "Arduino/pic128/pic128.ino",
            "num_dac": 128,
            "num_pd": 14,
            "cs_pins": [10, 9, 8, 7, 5, 4, 3, 2],   # chip k = ch 16k..16k+15; corrected 07-30
            "v_clamp": 4.0,                          # 0-4 V; single-heater safe, watch multi-heater thermal
            "adc_avg_n": 10,                         # firmware averages 10 sweeps per reply
            "per_write_config": ["0x03 00 84", "0x09 00 00", "0x05 FF FF"],  # after every value write
        },
        "laser": {
            "power_dbm": 15,
            "max_dbm": 15,
            "warning": "DO NOT exceed +15 dBm -- PD damage risk. Coupling ~13x (~11 dB) below "
                       "the 07-23 reference (fiber alignment drifted; not recoverable).",
            "zero_heater_fingerprint_mV_at_15dBm": {
                "8": 206, "12": 210, "10": 168, "5": 165, "1": 147, "3": 146},
        },
        "photodiodes": {
            "dead": [],                              # none on PIC B
            "strong": [3, 5, 6, 7, 8, 9, 10, 12],
            "weak": [0, 1, 2, 4, 11, 13],
        },
        "summary": {
            "nets": sum(c["stage"] != "spare" for c in channels),
            "drivable": len(drivable),
            "characterized": len(characterized),
            "dead": [c["dac"] for c in channels if c["elec"] == "dead"],
            "short": [c["dac"] for c in channels if c["elec"] == "short"],
            "weak": [c["dac"] for c in channels if c["elec"] == "weak"],
        },
        "channels": channels,
    }
    OUT.write_text(json.dumps(cfg, indent=2))
    s = cfg["summary"]
    print(f"wrote {OUT.relative_to(ROOT)}")
    print(f"  {s['nets']} nets | drivable {s['drivable']} | characterized (V0/Vpi) {s['characterized']}")
    print(f"  dead {s['dead']}  short {s['short']}  weak {s['weak']}")


if __name__ == "__main__":
    main()
