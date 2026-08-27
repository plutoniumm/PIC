"""CLI for the 4x4 rig.

    python -m pic selftest              # no hardware: every model self-test, plus a mock rig
    python -m pic status                # what is connected and what state it is in
    python -m pic measure               # live photodiode readout, heaters at 0 V
    python -m pic char --pd 0           # sweep every heater, fit fringes, write the calibration
    python -m pic program --random      # decompose a target unitary onto the chip and read back

Every hardware subcommand takes --mock, and the mock path is a physical model of the chip
rather than a stub, so it exercises the same code the real run does. --sim swaps that for
the bench as delivered -- six wired heaters, a 3 V clamp, the measured loss table and the
measured noise -- which is what to run before booking time on the real one.
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np

from .config import FIRMWARE_VMAX, VOLTAGE_MAX
from theory.clements import NMODE
from .layout import LABEL_OF_DAC, N_HEATERS
from .rig import Rig


def _rig(a, model=None):
    sim = getattr(a, "sim", False)
    kind = "mock" if (a.mock or sim) else "hw"
    return Rig(laser=kind, board="sim" if sim else kind, switch=kind,
               tec="mock" if (a.mock or sim) else a.tec, model=model,
               laser_port=a.laser_port, pic_port=a.pic_port,
               dynamic=getattr(a, "dynamic", False)).open()


def cmd_selftest(a):
    from theory import calib, clements, program, twin

    print("clements  ", end="", flush=True)
    print(f"round trip {clements._selftest():.1e}")
    print("twin      ", end="", flush=True)
    print("ideal {:.1e}  unitary {:.1e}  batched {:.1e}".format(*twin._selftest()[:3]))
    print("calib     ", end="", flush=True)
    n, tot = calib._selftest()
    print(f"{n}/{tot} random targets reachable")
    print("program   ", end="", flush=True)
    e, r, f = program._selftest(n=5)
    print(f"exact {e:.6f}  with coupler error {r:.4f} -> refined {f:.4f}")

    print("drift     ", end="", flush=True)
    from theory import drift as tdrift
    d = tdrift._selftest()
    print(f"probe sees {d['rank_alg']}/{d['rank_alg'] + d['gdim']} phase dirs "
          f"({d['gdim']} gauge), gains to {d['gain_err']:.0e}, "
          f"probe err {d['err_uncorrected']:.4f} -> {d['err_corrected']:.4f}")
    print("dynamic   ", end="", flush=True)
    from .drift import _selftest as drift_selftest
    r = drift_selftest()
    print(f"closed loop {r['before']:.4f} -> {r['after']:.4f}, "
          f"port-3 coupling {r['port3_db']:+.1f} dB, "
          f"below-noise refused {r['refused']}/{r['n_quiet']}")

    print("normalise ", end="", flush=True)
    from .normalise import _selftest as norm_selftest
    n = norm_selftest()
    print(f"PD full scale {n['norm'].full.min()*1e3:.0f}-{n['norm'].full.max()*1e3:.0f} mV, "
          f"rows {n['raw_row']:.2f}->{n['out_row']:.2f}, cols {n['raw_col']:.2f}->{n['both_col']:.2f}")

    print("mock rig  ", end="", flush=True)
    with Rig(laser="mock", board="mock", tec="mock", model="mock") as rig:
        v = np.zeros(N_HEATERS)
        got, pred = rig.measure(v), rig.predict(v)
        print(f"measure vs predict max diff {np.abs(got - pred).max():.1e}")

    print("bench sim ", end="", flush=True)
    from .sim import _selftest as sim_selftest
    s = sim_selftest()
    print(f"measured power table to {s['table_max_db']:.3f} dB, "
          f"read noise {s['sd_mV'][0]:.1f}-{s['sd_mV'][1]:.1f} mV, "
          f"Vpi recovered to {max(s['vpi_err'].values()):.2f} V "
          f"({s['n_ok_at_clamp']}/6 at the {FIRMWARE_VMAX:.0f} V clamp)")
    return 0


def cmd_status(a):
    with _rig(a) as rig:
        s = rig.status()
        print(f"laser on   : {s['laser_on']}")
        print(f"input port : {s['switch']['port']}")
        print(f"chip TEC   : {s['tec']['temperature_c']:.2f} C "
              f"(setpoint {s['tec']['setpoint_c']:.2f}, "
              f"{'stable' if s['tec']['stable'] else 'SETTLING'})")
        print(f"calibration: {s['calib']}")
    return 0


def cmd_measure(a):
    with _rig(a) as rig:
        v = np.zeros(N_HEATERS)
        print("photodiode volts, heaters at 0 V. Ctrl-C to stop.")
        try:
            with rig.session(duration_s=a.duration, power_dbm=a.dbm) as s:
                print(f"emission verified: {s.emitted}  (bfm {s.bfm_off:.3f} -> {s.bfm_on:.3f})")
                while not s.expired():
                    y = rig.measure(v)
                    print("  ".join(f"{x:7.4f}" for x in y), flush=True)
                    s.keepalive(a.period)
        except KeyboardInterrupt:
            print("\nstopped")
    return 0


def cmd_char(a):
    from .acquisition import estimate_seconds, grid
    from .characterize import characterize, digest, random_bases

    levels = grid() if a.levels is None else grid(a.levels)
    bases = random_bases(a.bases)
    with _rig(a) as rig:
        ports = None if a.ports is None else [int(x) for x in a.ports.split(",")]
        n_ports = len(ports) if ports else (1 if a.no_switch else NMODE)
        dur = estimate_seconds(levels.size * N_HEATERS * len(bases) * n_ports,
                               a.settle, a.repeats)
        print(f"characterizing {N_HEATERS} heaters x {levels.size} levels x {len(bases)} "
              f"base biases x {n_ports} input port(s); session {dur / 60:.0f} min")
        with rig.session(duration_s=dur, power_dbm=a.dbm) as s:
            if not s.emitted:
                print("WARNING: no emission detected -- every fit below will be noise.")
            res, calib = characterize(rig.board, s, pd=a.pd, levels=levels, bases=bases,
                                      switch=None if a.no_switch else rig.switch, ports=ports,
                                      settle_s=a.settle, repeats=a.repeats)
            if not a.no_normalise:
                # after the fringes, so the sweep is not disturbed by the search; the
                # references belong to this session's coupling and have to be retaken
                # whenever the fibre moves
                from .normalise import apply_to, measure as measure_norm
                print("\nphotodiode full scale (blocked = 0, transparent = 1):")
                norm = measure_norm(rig, ports=ports, repeats=a.repeats)
                apply_to(calib, norm)
                print(norm.summary())
        print()
        print(digest(res))
        missed = [r["label"] for r in res.values()
                  if not r["ok"] and ":alpha" not in r["label"] and ":aux" not in r["label"]]
        if missed:
            print(f"\n{len(missed)} mesh heater(s) still unidentified: {missed}")
            print("An external phase on a first-column MZI is a global phase when a single "
                  "input port is lit, whichever port that is, so no amount of switching finds "
                  "it. Split the input across two adjacent ports and re-run those channels "
                  "(see pic.characterize.probe_inputs).")
        if a.write:
            print(f"\nwrote {calib.save()}")
    return 0


def cmd_program(a):
    from theory import fidelity, random_unitary, unitary_for

    U = (random_unitary(np.random.default_rng(a.seed)) if a.random
         else np.load(a.target)["U"] if a.target.endswith(".npz")
         else np.loadtxt(a.target, dtype=complex))
    with _rig(a) as rig:
        v, ok, y = np.zeros(N_HEATERS), None, None
        with rig.session(duration_s=a.duration, power_dbm=a.dbm):
            v, ok, y = rig.realize(U)
        print(f"target unitary:\n{np.round(U, 3)}")
        print(f"\nheater volts ({int(ok.sum())}/{N_HEATERS} reachable, max {v.max():.2f} V):")
        for i, x in enumerate(v):
            print(f"  {LABEL_OF_DAC[i]:<14} {x:5.2f} V{'' if ok[i] else '   UNREACHABLE'}")
        print(f"\noutput photodiode volts: {np.round(y, 4)}")
        print(f"ideal-mesh fidelity of the programmed phases: "
              f"{fidelity(U, unitary_for(rig.calib.phases(v))):.4f}")
    return 0


def cmd_dataset(a):
    """Capture combos x all four input ports -- the transfer matrix the archive lacks.

    Every existing dataset draws a fresh random combination per port, so no DAC state was
    ever measured at all four inputs. That is the one thing needed to assemble T = |U|^2 per
    combination, which is what makes the unitarity constraint usable as a training term and
    what the drift probe reads. Holding the combination and cycling the switch is the whole
    difference, and it costs nothing extra to take."""
    import csv

    from .acquisition import estimate_seconds, random_vectors
    from .layout import REACHABLE_DACS

    rng = np.random.default_rng(a.seed)
    combos = random_vectors(a.combos, rng=rng, channels=REACHABLE_DACS)
    dur = estimate_seconds(a.combos * NMODE, a.settle, a.repeats) + a.combos * NMODE * 1.1
    print(f"{a.combos} combinations x {NMODE} ports = {a.combos * NMODE} reads; "
          f"about {dur / 60:.0f} min (the switch, not the readout, is the slow part)")

    with _rig(a) as rig:
        with rig.session(duration_s=dur * 1.3, power_dbm=a.dbm) as s:
            if not s.emitted:
                print("WARNING: no emission detected -- this dataset would be noise.")
                return 2
            rows = []
            for i, v in enumerate(combos):
                for k in range(NMODE):
                    rig.select_input(k)
                    y = np.mean([rig.outputs(v) for _ in range(a.repeats)], axis=0)
                    rows.append([i, k] + list(np.round(v[REACHABLE_DACS], 4))
                                + list(np.round(y, 6)))
                if (i + 1) % 25 == 0:
                    print(f"  {i + 1}/{a.combos}", flush=True)

    hdr = (["combo", "port"] + [f"dac{c}" for c in REACHABLE_DACS]
           + [f"pd{j}" for j in range(NMODE)])
    with open(a.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(hdr)
        w.writerows(rows)
    print(f"wrote {a.out}: {len(rows)} rows, {a.combos} full transfer matrices")
    return 0


def main(argv=None):
    # common options live on a parent parser so they can be given after the subcommand,
    # which is how anyone actually types them
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--mock", action="store_true", help="run with no hardware attached")
    common.add_argument("--sim", action="store_true",
                        help="no hardware, but the bench as delivered: 6 wired heaters, a "
                             "3 V clamp, the measured loss table and noise (pic.sim)")
    common.add_argument("--tec", default="mock", help="TEC: 'mock', 'none', or a serial port")
    common.add_argument("--laser-port", default=None)
    common.add_argument("--pic-port", default=None)
    common.add_argument("--dbm", type=float, default=8.0,
                        help="laser output power. 8 dBm is what the best archive set was "
                             "taken at and leaves the TIA headroom (brightest of 77,280 "
                             "logged reads is 0.587 V); the 6x6's 13 dBm has never been "
                             "put on this chip")
    common.add_argument("--dynamic", action="store_true",
                        help="real-time drift correction: re-probe the four input ports "
                             "periodically and pre-distort the commanded phases by the "
                             "drift that explains the change. Off by default -- correcting "
                             "a drift smaller than the probe noise makes the chip worse")

    ap = argparse.ArgumentParser(prog="pic", description=__doc__.splitlines()[0],
                                 parents=[common])
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("selftest", help="run every model self-test; no hardware",
                   parents=[common]).set_defaults(fn=cmd_selftest)
    sub.add_parser("status", help="laser, TEC and calibration state",
                   parents=[common]).set_defaults(fn=cmd_status)

    p = sub.add_parser("measure", help="live photodiode readout", parents=[common])
    p.add_argument("--duration", type=float, default=120.0)
    p.add_argument("--period", type=float, default=0.2)
    p.set_defaults(fn=cmd_measure)

    p = sub.add_parser("char", help="sweep every heater and fit its fringe", parents=[common])
    p.add_argument("--pd", type=int, default=None,
                   help="fit against one photodiode; default is whichever sees each heater best")
    p.add_argument("--levels", type=int, default=None,
                   help="sweep points per heater; the default clears the Nyquist gate")
    p.add_argument("--ports", default=None,
                   help="input ports to sweep through, e.g. 0,2 (default: all four)")
    p.add_argument("--no-switch", action="store_true",
                   help="the fibre is plugged straight into one port; do not try to switch")
    p.add_argument("--bases", type=int, default=3,
                   help="base biases to sweep from; a heater dark in one may show in another")
    p.add_argument("--settle", type=float, default=0.2)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--write", action="store_true", help="save to pic_data/calib.json")
    p.add_argument("--no-normalise", action="store_true",
                   help="skip the PD full-scale pass (blocked = 0, transparent = 1)")
    p.set_defaults(fn=cmd_char)

    p = sub.add_parser("program", help="put a target unitary on the chip", parents=[common])
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--random", action="store_true")
    g.add_argument("--target", help="path to a 4x4 complex matrix (.npz with U, or text)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--duration", type=float, default=60.0)
    p.set_defaults(fn=cmd_program)

    p = sub.add_parser("dataset", help="combos x all 4 ports -> CSV of transfer matrices",
                       parents=[common])
    p.add_argument("--combos", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--settle", type=float, default=0.2)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--out", default="pic_data/transfer.csv")
    p.set_defaults(fn=cmd_dataset)

    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    raise SystemExit(main())
