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
import os
import sys
import time

import numpy as np

from .config import FIRMWARE_VMAX, VOLTAGE_MAX
from theory.clements import NMODE
from .config import TEC_TOLERANCE_C, VOLTAGE_MAX_CH
from .layout import LABEL_OF_DAC, N_HEATERS
from .rig import Rig


def _rig(a, model=None):
    sim = getattr(a, "sim", False)
    kind = "mock" if (a.mock or sim) else "hw"
    return Rig(laser=kind, board="sim" if sim else kind, switch=kind,
               tec="mock" if (a.mock or sim) else a.tec, model=model,
               laser_port=a.laser_port, pic_port=a.pic_port,
               dynamic=getattr(a, "dynamic", False),
               keep_laser=getattr(a, "keep_laser", False)).open()


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
    b = n["bench"]
    print(f"PD full scale {n['norm'].full.min()*1e3:.0f}-{n['norm'].full.max()*1e3:.0f} mV, "
          f"rows {n['raw_row']:.2f}->{n['out_row']:.2f}, cols {n['raw_col']:.2f}->{n['both_col']:.2f}"
          + (f"; {b['n']} bench states doubly stochastic to "
             f"{np.median(b['stored']):.2f}->{np.median(b['column']):.2f} on the sweep"
             if b else "; no bench session on disk"))

    print("mock rig  ", end="", flush=True)
    with Rig(laser="mock", board="mock", tec="mock", model="mock") as rig:
        v = np.zeros(N_HEATERS)
        got, pred = rig.measure(v), rig.predict(v)
        print(f"measure vs predict max diff {np.abs(got - pred).max():.1e}")

    print("ising     ", end="", flush=True)
    from theory import ising as tising
    from theory.calib import Calibration as _Cal
    from theory.twin import Twin as _Twin
    _cal = _Cal.load_or_nominal()
    rd = tising.reach_dim(_Twin(), _cal, tising.measured(_cal))
    ir = tising._selftest(instances=3, ns=(3, 4), steps=250, restarts=12)
    fmt = lambda n: next(r for r in ir if r["case"].startswith("measured h") and r["n"] == n)
    # the ground-state rate needs more instances than a gate can afford; the hosting error
    # and the reachable dimension are the numbers that are stable at three
    print(f"mesh steers {rd['dim']:.2f} of {rd['of']} coupling directions "
          f"(n=3 needs 1, n=4 needs 4); hosts n=3 to {fmt(3)['design_err']:.3f}, "
          f"n=4 to {fmt(4)['design_err']:.2f}")

    print("matvec    ", end="", flush=True)
    from .matvec import _selftest as matvec_selftest
    m = matvec_selftest()
    print(f"{m['free']} steerable channels (widest {m['span_max']:.2f} pi), 2x2 on rails "
          f"out{m['rails'][0]} in{m['rails'][1]} hosted to {m['fit_err']:.4f}, "
          f"vec err {m['vec_err']:.4f} -> {m['noisy_err']:.3f} at 1.2% read noise, "
          f"sign {m['sign_acc']:.0%}; probe vs sim truth "
          f"{m['probe_err']['static']:.2f} static -> {m['probe_err']['column']:.2f} swept")

    print("matmat    ", end="", flush=True)
    from theory.matmat import compose as mm_compose, validate_block
    from .matvec import bench_box
    _bx = bench_box()
    mm = validate_block(_bx, k=2, n=(4, 4), cols=3, restarts=8, steps=200)
    gp = mm_compose(_bx, k=2, noise=0.012, trials=3, restarts=8, steps=200)
    print(f"4x4 in {mm['tiles']} 2x2 tiles: {mm['writes']} programs, "
          f"{mm['reads']:.0f} reads/column, Y err {mm['vec_err']:.4f}, "
          f"sign {mm['sign_acc']:.0%}; composed product sits {gp['dist_c']:.3f} off O(2), "
          f"err {gp['raw']:.3f} -> {gp['polar']:.3f} projected")

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


def _bar(frac, width=18):
    """A fixed-width bar. Over-full is marked rather than clipped: normalised readings above
    1.0 are legitimate -- full scale is a measured reference, not a ceiling."""
    n = int(max(0.0, min(frac, 1.0)) * width)
    return ("#" * n).ljust(width) + ("+" if frac > 1.001 else " ")


def cmd_measure(a):
    """Live photodiode dashboard. Raw volts, and the fraction of each detector's own full
    scale when a normalisation exists -- the fraction is the honest one to watch, because
    the detectors do not share a gain."""
    from theory.calib import Calibration

    with _rig(a) as rig:
        v = np.zeros(N_HEATERS)
        cal = Calibration.load_or_nominal()
        norm = cal.meta.get("normalisation", {})
        full = np.asarray(norm.get("full_v", []), float)
        dark = np.asarray(norm.get("dark_v", []), float)
        scaled = full.size == NMODE and np.all(full > 0)
        ports = [int(x) for x in a.ports.split(",")] if a.ports else None

        lo = np.full(NMODE, np.inf)
        hi = np.full(NMODE, -np.inf)
        print("heaters at 0 V. Ctrl-C to stop." if not ports else
              f"heaters at 0 V, cycling ports {ports}. Ctrl-C to stop.")
        if not scaled:
            print("no normalisation on file -- showing raw volts only "
                  "(run `./do char --write` to get full-scale references)")
        head = "  ".join(f"PD{k:<6}" for k in range(NMODE))
        print(f"{'port':>5}  {head}   {'sum':>7}")
        try:
            with rig.session(duration_s=a.duration, power_dbm=a.dbm) as s:
                print(f"emission verified: {s.emitted}  "
                      f"(bfm {s.bfm_off:.3f} -> {s.bfm_on:.3f})")
                k = 0
                while not s.expired():
                    sel = ""
                    if ports:
                        p = ports[k % len(ports)]
                        rig.select_input(p)
                        sel = str(p)
                        k += 1
                    y = rig.measure(v)
                    lo, hi = np.minimum(lo, y), np.maximum(hi, y)
                    if scaled:
                        f = (y - dark) / (full - dark)
                        cells = "  ".join(f"{x:6.4f}{_bar(q, 8)}" for x, q in zip(y, f))
                    else:
                        cells = "  ".join(f"{x:7.4f} " for x in y)
                    print(f"{sel:>5}  {cells}   {y.sum():7.4f}", flush=True)
                    s.keepalive(a.period)
        except KeyboardInterrupt:
            pass
        print("\nper-detector range over the run:")
        for k in range(NMODE):
            print(f"  PD{k}  min {lo[k]:7.4f}  max {hi[k]:7.4f}  swing {hi[k] - lo[k]:7.4f} V")
    return 0


def cmd_recal(a):
    """Re-anchor phi0 on the already-characterized channels, in seconds rather than minutes.

    A full characterization measures two numbers per heater. Only one of them moves: Vpi is
    heater geometry, phi0 is optical path length, and it is the path that drifts with
    temperature and time. So a re-anchor holds Vpi, amplitude and offset from the last full
    run and solves for phi0 alone -- see theory.calib.refit_phi0 for why that is the only
    tractable form on a chip whose heaters cover under half a pi.

    Each channel is swept only at the port and detector its full characterization found it
    brightest on, which is already recorded."""
    import json
    from pathlib import Path

    from theory.calib import Calibration, refit_phi0

    res_path = Path("pic_data/char_results.json")
    if not res_path.exists():
        raise SystemExit("no pic_data/char_results.json -- run `./do char --write` first")
    res = json.loads(res_path.read_text())
    cal = Calibration.load_or_nominal()
    fits = {int(k): v for k, v in res.items() if v.get("ok")}
    if not fits:
        raise SystemExit("no characterized channels to re-anchor")

    with _rig(a) as rig:
        n_reads = len(fits) * a.levels * a.repeats
        dur = max(60.0, n_reads * (a.settle + 0.3) + 30.0)
        print(f"re-anchoring {len(fits)} channel(s) x {a.levels} levels "
              f"(~{n_reads} reads, {dur / 60:.0f} min budget)")
        with rig.session(duration_s=dur, power_dbm=a.dbm) as s:
            print(f"{'heater':<14} {'phi0 was':>9} {'now':>9} {'drift':>9} {'rmse/amp':>9}")
            moved = {}
            for dac, f in sorted(fits.items()):
                pd, port = int(f["pd"]), int(f["port"])
                if port is not None:
                    rig.select_input(port)
                vmax = VOLTAGE_MAX_CH[dac]
                levels = np.linspace(0.0, vmax, a.levels)
                ys = []
                for lv in levels:
                    v = np.zeros(N_HEATERS)
                    v[dac] = lv
                    ys.append(np.mean([rig.measure(v) for _ in range(a.repeats)], axis=0)[pd])
                    s.keepalive()
                phi, _, _, rmse = refit_phi0(levels, np.asarray(ys), f["vpi"],
                                             amp=f["B"], offset=f["A"], prior=f["phi0"])
                d = (phi - f["phi0"] + np.pi) % (2 * np.pi) - np.pi
                moved[dac] = phi
                print(f"{f['label']:<14} {f['phi0'] / np.pi:>8.3f}p {phi / np.pi:>8.3f}p "
                      f"{d:>+8.3f}r {rmse / max(abs(f['B']), 1e-9):>9.2f}")
            for dac, phi in moved.items():
                cal.phi0[dac] = phi
    cal.meta["reanchored"] = {"channels": sorted(moved), "levels": a.levels}
    if a.write:
        print(f"wrote {cal.save()}")
    else:
        print("(not written -- pass --write to update pic_data/calib.json)")
    return 0


def cmd_tec(a):
    """Live chip temperature, with what the controller is asking for.

    Shows the commanded drive next to the temperature on purpose: this rig has had the loop
    running against an output that was not actuating, and a stream of temperatures alone
    looks identical whether the TEC is holding the die or the room is. Watch whether the
    error shrinks while the drive is non-zero -- if the drive rails and nothing moves, the
    controller is talking to nothing."""
    import time

    from .devices.tec import SerialTEC

    tec = SerialTEC(port=a.tec_port) if a.tec_port else SerialTEC(port=_find_tec())
    tec.open()
    if a.setpoint is not None:
        tec.target = float(a.setpoint)
        print(f"setpoint -> {a.setpoint:.2f} C")
    t0 = time.time()
    first = None
    railed = 0
    print(f"{'t':>6}  {'temp':>7}  {'set':>7}  {'err':>7}  {'drive':>7}   trend")
    try:
        while True:
            st = tec.status()
            c = st["temperature_c"]
            sp = st.get("controller_setpoint_c", st["setpoint_c"])
            drive = st.get("drive_v", float("nan"))
            if first is None:
                first = c
            err = c - sp
            if abs(drive) >= 1.999:
                railed += 1
            mark = "IN BAND" if abs(err) <= TEC_TOLERANCE_C else f"{c - first:+.2f} since start"
            print(f"{time.time() - t0:6.0f}  {c:7.2f}  {sp:7.2f}  {err:+7.2f}  "
                  f"{drive:+7.3f}   {mark}", flush=True)
            time.sleep(a.period)
    except KeyboardInterrupt:
        pass
    finally:
        tec.close()
    return 0


def _find_tec():
    """The TEC is a bare usbmodem like the board, so guess only when there is one left."""
    import glob

    from .interface import find_port

    board, _ = find_port(os.environ.get("PIC4_PORT"))
    cands = [p for p in sorted(glob.glob("/dev/cu.usbmodem*")) if p != board]
    if len(cands) != 1:
        raise SystemExit(f"cannot tell which port is the TEC (candidates {cands}); "
                         f"pass --tec-port")
    return cands[0]


def _merge_calib(new, res, path=None):
    """Fold a partial sweep into the calibration on disk instead of replacing it.

    A targeted re-run of one weak channel would otherwise reset every other heater to the
    nominal Vpi -- silently, because a nominal calibration looks exactly like a measured one
    apart from its metadata. Only channels this run actually fitted are taken from it."""
    from theory.calib import CONFIG_PATH, Calibration

    path = CONFIG_PATH if path is None else path
    try:
        old = Calibration.load(path)
    except (OSError, ValueError, KeyError):
        return new
    fitted = [int(d) for d, f in res.items() if f.get("ok")]
    if not fitted:
        return old
    for d in fitted:
        old.vpi[d], old.phi0[d] = new.vpi[d], new.phi0[d]
    old.pd_gain, old.pd_offset = new.pd_gain, new.pd_offset
    meta = dict(old.meta); meta.update(new.meta)
    meta["fitted_this_run"] = fitted
    old.meta = meta
    return old


def _save_results(res, path="pic_data/char_results.json", merge=False):
    """The per-heater fits, including each channel's (port x detector) fingerprint.

    `calib.json` keeps only Vpi and phi0 -- the two numbers programming needs. The
    fingerprint is what says *where in the mesh* a heater sits, and it cost a full sweep to
    measure, so it does not get to live only in stdout."""
    import json
    from pathlib import Path

    out = {}
    if merge:
        try:
            out = json.loads(Path(path).read_text())
        except (OSError, ValueError):
            out = {}
    for dac, f in res.items():
        out[str(dac)] = {k: (v.tolist() if hasattr(v, "tolist") else v)
                         for k, v in f.items()}
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2, default=float))
    print(f"wrote {p}")
    return p


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
            chans = None
            if a.channels:
                chans = []
                for part in a.channels.split(","):
                    if "-" in part:
                        lo, hi = part.split("-")
                        chans += list(range(int(lo), int(hi) + 1))
                    else:
                        chans.append(int(part))
            res, calib = characterize(rig.board, s, pd=a.pd, levels=levels, bases=bases,
                                      channels=chans,
                                      switch=None if a.no_switch else rig.switch, ports=ports,
                                      settle_s=a.settle, repeats=a.repeats)
            if a.write:   # persist the sweep before anything else can fail on top of it
                calib = _merge_calib(calib, res)
                calib.save()
                _save_results(res, merge=True)
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


def cmd_matvec(a):
    """Signed matvec from intensity alone. `--scan` is a fit, so it needs no rig at all."""
    from .matvec import main as matvec_main

    if a.scan:
        return matvec_main(None, a)
    with _rig(a) as rig:
        with rig.session(duration_s=a.duration, power_dbm=a.dbm):
            return matvec_main(rig, a)


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
    dur = estimate_seconds(a.combos * NMODE, a.settle, a.repeats) + NMODE * 1.1
    print(f"{a.combos} combinations x {NMODE} ports = {a.combos * NMODE} reads; "
          f"about {dur / 60:.0f} min (the switch, not the readout, is the slow part)")

    with _rig(a) as rig:
        with rig.session(duration_s=dur * 1.3, power_dbm=a.dbm) as s:
            if not s.emitted:
                print("WARNING: no emission detected -- this dataset would be noise.")
                return 2
            # Port outermost, combination innermost. The Sercalo needs ~1 s to settle and
            # the readout needs ~0.2 s, so switching per measurement spends most of the run
            # waiting on a mirror: n_combos * 4 moves instead of 4. Holding a port and
            # walking every combination through it gives identical data -- the same DAC
            # states are still measured at all four inputs, which is what makes T = |U|^2
            # assemblable -- for a fraction of the time. The cost is that one combination's
            # four port readings are now minutes apart rather than seconds, so slow drift
            # lands between them; that is what the drift probe is for, and it is a far
            # smaller error than losing the run to a 57-minute switch budget.
            rows = []
            for k in range(NMODE):
                rig.select_input(k)
                for i, v in enumerate(combos):
                    y = np.mean([rig.outputs(v) for _ in range(a.repeats)], axis=0)
                    rows.append([i, k] + list(np.round(v[REACHABLE_DACS], 4))
                                + list(np.round(y, 6)))
                    s.keepalive()
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
    # `ising` carries its own flag list and is forwarded whole rather than re-declared here;
    # argparse's REMAINDER cannot take the first token when it starts with a dash
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "ising":
        from .ising import main as ising_main
        return ising_main(argv[1:])

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
    p.add_argument("--ports", default=None,
                   help="cycle the input switch through these ports while streaming, "
                        "e.g. 0,1,2,3 -- a per-port pattern is what distinguishes guided "
                        "light from stray light on the detector array")
    p.set_defaults(fn=cmd_measure)

    p = sub.add_parser("recal", help="fast phi0 re-anchor on characterized channels",
                       parents=[common])
    p.add_argument("--levels", type=int, default=5)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--settle", type=float, default=0.4)
    p.add_argument("--write", action="store_true")
    p.set_defaults(fn=cmd_recal)

    p = sub.add_parser("tec", help="live chip temperature and controller drive")
    p.add_argument("setpoint", nargs="?", type=float, default=None,
                   help="retarget the controller to this temperature before streaming")
    p.add_argument("--tec-port", default=None, help="TEC serial port (else autodetect)")
    p.add_argument("--period", type=float, default=1.0)
    p.set_defaults(fn=cmd_tec)

    p = sub.add_parser("char", help="sweep every heater and fit its fringe", parents=[common])
    p.add_argument("--pd", type=int, default=None,
                   help="fit against one photodiode; default is whichever sees each heater best")
    p.add_argument("--levels", type=int, default=None,
                   help="sweep points per heater; the default clears the Nyquist gate")
    p.add_argument("--ports", default=None,
                   help="input ports to sweep through, e.g. 0,2 (default: all four)")
    p.add_argument("--channels", type=str, default=None,
                   help="DAC channels to sweep, e.g. 0-5 or 0,2,5 (default: all). The six "
                        "internal phase shifters are 0-5; external phases are output-side "
                        "and invisible in intensity, so there is nothing to find on them")
    p.add_argument("--keep-laser", action="store_true",
                   help="leave the laser lit on exit if it was already on when we opened "
                        "(default is always off: an unattended lit diode has no watchdog)")
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

    sub.add_parser("ising",
                   help="Ising ground states from the four-port intensity probe "
                        "(own flags; see `python -m pic.ising --help`)")

    p = sub.add_parser("matvec", help="signed y = B x from the photodiodes, no homodyne",
                       parents=[common])
    p.add_argument("--k", type=int, default=2, help="block size; 3 and 4 do not fit (see "
                                                    "theory.intensity_matvec)")
    p.add_argument("--target", default=None, help="path to a real matrix (text)")
    p.add_argument("--rails", default=None, help="block to use, e.g. 1,2:1,2 (out:in)")
    p.add_argument("--mode", default="shift", choices=("shift", "split"),
                   help="nonnegative decomposition; 'split' is the 6x6's and is noisier")
    p.add_argument("--scan", action="store_true",
                   help="rank every rail pair by residual and brightness; no hardware")
    p.add_argument("--block", type=int, default=None, metavar="K",
                   help="tile the target into KxK blocks (theory.matmat); K=2 is what this "
                        "die hosts")
    p.add_argument("--cols", type=int, default=3,
                   help="columns of X for --block, or pairs of matrices for --unitary")
    p.add_argument("--unitary", action="store_true",
                   help="compose two hosted rotations and check the product is still "
                        "orthogonal; polar and QR projections side by side")
    p.add_argument("--optical-input", action="store_true",
                   help="set the input weight with the laser instead of in software, so "
                        "the multiply happens in light; costs a retune per port")
    p.add_argument("--static-scale", action="store_true",
                   help="read one port at a time and divide by the calibration's stored "
                        "input_scale, as before the four-port sweep. Deprecated and wrong "
                        "on this bench (see pic.normalise); here to reproduce old runs")
    p.add_argument("--planner", default="auto", choices=("auto", "table", "twin"),
                   help="where the heater state comes from: 'table' picks the measured "
                        "state that best hosts the target, 'twin' fits volts through the "
                        "model. 'auto' is the table on hardware and the twin against either "
                        "simulator, because the table is measurements of the real die")
    p.add_argument("--refresh", action="store_true",
                   help="re-sweep the chip for every vector instead of reusing the held "
                        "state, so the vector error carries read noise; costs a thermal "
                        "settle per vector")
    p.add_argument("--repeats", type=int, default=1,
                   help="photodiode reads averaged per port; error falls as 1/sqrt(N)")
    p.add_argument("--restarts", type=int, default=16)
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--duration", type=float, default=120.0)
    p.set_defaults(fn=cmd_matvec)

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
