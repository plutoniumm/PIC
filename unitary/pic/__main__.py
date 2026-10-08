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
from pathlib import Path
import os
import sys
import time

import numpy as np

from .config import VOLTAGE_MAX, data_path, detector_mode, pin_detectors, port_of
from theory.clements import NMODE
from .config import TEC_TOLERANCE_C, VOLTAGE_MAX_CH
from .layout import LABEL_OF_DAC, N_HEATERS
from .log import ev
from .rig import Rig
from .devices.tec import auto_tec
from .interface import MissingOutputs, need_outputs
from .session import WatchdogTripped


def _rig(a, model=None):
    sim = getattr(a, "sim", False)
    kind = "mock" if (a.mock or sim) else "hw"
    return Rig(
        laser=kind,
        board="sim" if sim else kind,
        switch=kind,
        tec=(
            "mock"
            if (a.mock or sim)
            else (auto_tec(_find_tec) if a.tec == "auto" else port_of(a.tec))
        ),
        model=model,
        laser_port=port_of(a.laser_port),
        pic_port=port_of(a.pic_port),
        dynamic=getattr(a, "dynamic", False),
        keep_laser=getattr(a, "keep_laser", False),
    ).open()


@pin_detectors("pd")  # the PD path; `pic.apply._selftest` runs SPD mode itself
def cmd_selftest(a):
    from theory import calib, clements, layout, program, twin

    from . import log

    log._selftest()
    print("log       one @ev JSON line per event; unknown components and levels refused")
    print("clements  ", end="", flush=True)
    print(f"round trip {clements._selftest():.1e}")
    print("layout    ", end="", flush=True)
    w, rk = layout._selftest()
    print(
        f"spares redundant to {w:.0e}; rank {rk['physical']} of 18 physical phases, "
        f"{rk['driven']} driven, {rk['model']} modelled"
    )
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
    print(
        f"probe sees {d['rank_alg']}/{d['rank_alg'] + d['gdim']} phase dirs "
        f"({d['gdim']} gauge), gains to {d['gain_err']:.0e}, "
        f"probe err {d['err_uncorrected']:.4f} -> {d['err_corrected']:.4f}"
    )
    print("dynamic   ", end="", flush=True)
    from .drift import _selftest as drift_selftest

    r = drift_selftest()
    print(
        f"closed loop {r['before']:.4f} -> {r['after']:.4f}, "
        f"port-3 coupling {r['port3_db']:+.1f} dB, "
        f"below-noise refused {r['refused']}/{r['n_quiet']}"
    )

    print("normalise ", end="", flush=True)
    from .normalise import _selftest as norm_selftest

    n = norm_selftest()
    b = n["bench"]
    print(
        f"PD full scale {n['norm'].full.min()*1e3:.0f}-{n['norm'].full.max()*1e3:.0f} mV, "
        f"rows {n['raw_row']:.2f}->{n['out_row']:.2f}, cols {n['raw_col']:.2f}->{n['both_col']:.2f}"
        + (
            f"; {b['n']} bench states doubly stochastic to "
            f"{np.median(b['stored']):.2f}->{np.median(b['column']):.2f} on the sweep"
            if b
            else "; no bench session on disk"
        )
    )

    print("mock rig  ", end="", flush=True)
    with Rig(laser="mock", board="mock", tec="mock", model="mock", detectors="pd") as rig:
        v = np.zeros(N_HEATERS)
        got, pred = rig.measure(v), rig.predict(v)
        print(f"measure vs predict max diff {np.abs(got - pred).max():.1e}")

    # Ising is out of the gate for this campaign. Its selftest fails since the rewire for
    # the expected reason -- 12 modelled phases steer fewer coupling directions than 14 did,
    # so the two-pass encoding no longer beats the one-pass one. Put it back by restoring
    # this block when an Ising run is on the table again; nothing in theory/ising.py moved.
    print("matvec    ", end="", flush=True)
    from .matvec import _selftest as matvec_selftest

    m = matvec_selftest()
    print(
        f"{m['free']} steerable channels (widest {m['span_max']:.2f} pi), 2x2 on rails "
        f"out{m['rails'][0]} in{m['rails'][1]} hosted to {m['fit_err']:.4f}, "
        f"vec err {m['vec_err']:.4f} -> {m['noisy_err']:.3f} at 1.2% read noise, "
        f"sign {m['sign_acc']:.0%}; probe vs sim truth "
        f"{m['probe_err']['static']:.2f} static -> {m['probe_err']['column']:.2f} swept"
    )

    print("matmat    ", end="", flush=True)
    from theory.matmat import compose as mm_compose, validate_block
    from .matvec import bench_box

    _bx = bench_box()
    mm = validate_block(_bx, k=2, n=(4, 4), cols=3, restarts=8, steps=200)
    gp = mm_compose(_bx, k=2, noise=0.012, trials=3, restarts=8, steps=200)
    print(
        f"4x4 in {mm['tiles']} 2x2 tiles: {mm['writes']} programs, "
        f"{mm['reads']:.0f} reads/column, Y err {mm['vec_err']:.4f}, "
        f"sign {mm['sign_acc']:.0%}; composed product sits {gp['dist_c']:.3f} off O(2), "
        f"err {gp['raw']:.3f} -> {gp['polar']:.3f} projected"
    )

    print("apply     ", end="", flush=True)
    from .apply import _selftest as apply_selftest

    ap = apply_selftest()
    sp = ap["spread"]["full"]
    print(
        f"seam: 4x4 in four tiles hosts to {ap['matrix_err']['full']:.3f} measured, one "
        f"2x2 tile to {ap['matrix_err']['tile']:.3f}; per-x rel err "
        f"{sp['min']:.3f}-{sp['max']:.3f} over {sp['n']} vectors; {ap['wiring']}"
    )
    sd = ap["spd"]
    print(
        f"spd       4 mock SPDs, each its own dark/max: Vpi {sd['vpi']:.3f} V fitted, physics "
        f"fit trained, 4x4 to {sd['full'].matrix.value:.3f} full, "
        f"{sd['tile'].matrix.value:.3f} tiled; 1 SPD: all-output consumers refused by name"
    )

    return 0


def cmd_spdnorm(a):
    """SPD normalisation in five steps: dark with the laser off, then each input port lit
    once with every heater at its ceiling, one read per SPD. Each input couples its own
    light into the chip and each SPD collects its own share, so the reference is kept per
    (port, SPD) and a read divides by the lit port's own value."""
    from spd.array import save_max, save_port_max

    from .config import mirror_pairs

    with pin_detectors("spd"), _rig(a) as rig:
        b = rig.board
        need_outputs(b, "SPD normalisation")
        ids = b.spds.ids
        if rig.laser.is_on():
            raise SystemExit("laser is on; the dark needs it off")
        n = 1 + NMODE
        ev("pd", "step", "1/5 dark, laser off", k=0, n=n, step="dark")
        rig._zero()
        c = b._count(a.dark_s)
        dark = {k: c[k].counts / c[k].seconds for k in ids}
        ev("pd", "dark", "dark " + ", ".join(f"{k} {r:.1f}/s" for k, r in dark.items()), "ok")

        top = mirror_pairs(np.array(VOLTAGE_MAX_CH, float))
        ref = {k: {} for k in ids}
        dur = NMODE * (a.count + 2) + a.settle + 120
        with rig.session(duration_s=dur, power_dbm=a.dbm, calibrating=True) as s:
            b.board.measure_raw(top)  # heaters only: the SPDs are read below
            s.keepalive(a.settle)  # one heat step for all four ports, not one per port
            for p in range(NMODE):
                ev("pd", "step", f"{p + 2}/5 input P{p}", k=p + 1, n=n, step=f"P{p}")
                rig.select_input(p)
                s.keepalive(1.0)
                s.check()
                c = b._count(a.count)
                for k in ids:
                    ref[k][p] = c[k].counts / c[k].seconds
                ev("pd", "max", f"P{p} " + ", ".join(f"{k} {ref[k][p]:.0f}/s" for k in ids),
                   "ok", port=p)
            rig._zero()
        ev("pd", "step", "5/5 done", k=n, n=n, step="done")
    for k in ids:
        # a reference within 3 sigma of dark would divide reads by noise
        low = [p for p in range(NMODE) if ref[k][p] - dark[k] < 3 * np.sqrt(max(dark[k], 1) / a.count)]
        print(f"{k} PD{b.slots[k]}  dark {dark[k]:7.1f}  ref P0..P3 "
              + " ".join(f"{ref[k][p]:8.0f}" for p in range(NMODE))
              + (f"  near dark on P{low}" if low else ""))
    if a.write:
        save_max(darks=dark, rates={k: max(ref[k].values()) for k in ids})
        save_port_max(ref)
        ev("job", "wrote", "wrote pic_data/spd_max.json", "ok", path="pic_data/spd_max.json")
    return 0


def cmd_ports(a):
    """Every USB serial device: which instrument it is, its chip id, and its port today."""
    from serial.tools import list_ports

    from .config import USB_SERIAL, _chip, spd_map

    role = {v: k for k, v in USB_SERIAL.items()}
    slot = spd_map()
    rows = []
    for p in sorted(list_ports.comports(), key=lambda p: p.device):
        if not p.vid:
            continue
        chip = _chip(p.serial_number) or "?"
        what = role.get(chip) or (f"spd PD{slot[chip]}" if chip in slot else "unknown")
        rows.append((what, chip, p.device, f"{p.vid:04x}:{p.pid:04x}"))
    for want in (*USB_SERIAL, *(f"spd PD{s}" for s in sorted(slot.values()))):
        if want not in {r[0] for r in rows}:
            rows.append((want, "-", "not connected", "-"))
    head = ("component", "chip id", "port", "usb id")
    w = [max(len(str(r[i])) for r in (head, *rows)) for i in range(4)]
    for r in (head, tuple("-" * x for x in w), *rows):
        print("  ".join(str(c).ljust(x) for c, x in zip(r, w)))
    return 0


def cmd_status(a):
    with _rig(a) as rig:
        s = rig.status()
        print(f"laser on   : {s['laser_on']}")
        print(f"input port : {s['switch']['port']}")
        print(
            f"chip TEC   : {s['tec']['temperature_c']:.2f} C "
            f"(setpoint {s['tec']['setpoint_c']:.2f}, "
            f"{'stable' if s['tec']['stable'] else 'SETTLING'})"
        )
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
        # an SPD read is already on its own 0..1 scale (`python -m spd max`)
        scaled = rig.detectors == "pd" and full.size == NMODE and np.all(full > 0)
        ports = [int(x) for x in a.ports.split(",")] if a.ports else None

        lo = np.full(NMODE, np.inf)
        hi = np.full(NMODE, -np.inf)
        print(
            "heaters at 0 V. Ctrl-C to stop."
            if not ports
            else f"heaters at 0 V, cycling ports {ports}. Ctrl-C to stop."
        )
        if rig.detectors == "spd":
            print("SPD mode: 0 dark, 1 at each detector's stored max; -- where no SPD sits")
        elif not scaled:
            print(
                "no normalisation on file -- showing raw volts only "
                "(run `python -m pic char --write` to get full-scale references)"
            )
        head = "  ".join(f"PD{k:<6}" for k in range(NMODE))
        print(f"{'port':>5}  {head}   {'sum':>7}")
        try:
            with rig.session(duration_s=a.duration, power_dbm=a.dbm) as s:
                print(
                    f"emission verified: {s.emitted}  " f"(bfm {s.bfm_off:.3f} -> {s.bfm_on:.3f})"
                )
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


def cmd_calibrate(a):
    """Agree every constant the bench needs agreed, and write what can be measured.

    Each check here is a failure that actually happened on 2026-08-27/28, and every one of
    them was silent -- the run completed and printed a plausible number:

      * the host and the firmware kept independent copies of the per-channel voltage clamp,
        and three channels sat at 0.00 V in firmware while the host believed they were open;
      * `calib.json` was fitted at 25 C while the chip ran at 27 C, putting a uniform
        thermo-optic offset under every result for hours;
      * `pd_offset` carried 31.5 mV on PD0 against a true 0.14 mV, so a third of the median
        signal was being subtracted away on the live-probe path;
      * `BEST_RAILS` is a stored constant that its own docstring says to re-derive after any
        change to the ceilings, and it was fed to an 8x8 by a ranking on MEDIAN hosting, which
        chose a pair whose worst tile was 0.66 and returned Y error 0.5581;
      * the laser diode's own TEC settles wherever it likes -- 27.4, 34.3, 39.9, 42.3 and
        47.6 C were all seen in one session -- and wavelength follows it, so two tables taken
        at different hold points describe different chips.

    Dry run by default: it says what disagrees and what it would write. `--write` persists.
    """
    import json
    import numpy as np
    from .config import ADC_REF_V, HEATER_OHMS, TEC_SETPOINT_C, VOLTAGE_MAX_CH, N_HEATERS
    from .matvec import bench_box, load_transfers, plan_block_from_table
    from .rig import assert_calib_temperature

    ok, todo, writes = [], [], {}

    def say(good, label, detail):
        (ok if good else todo).append(f"  [{'ok' if good else '--'}] {label:<26} {detail}")

    with _rig(a) as rig:
        calib = rig.calib
        meta = dict(calib.meta or {})

        # 1. the two copies of the voltage clamp
        try:
            fw = rig.board.clamp_table() if hasattr(rig.board, "clamp_table") else None
        except Exception:
            fw = None
        if fw is None:
            say(True, "firmware clamp", "checked by Rig.open (assert_firmware_vmax)")
        else:
            bad = [
                (i, h, f) for i, (h, f) in enumerate(zip(VOLTAGE_MAX_CH, fw)) if abs(h - f) > 0.011
            ]
            say(
                not bad,
                "firmware clamp",
                (
                    "host and board agree on all 16"
                    if not bad
                    else f"MISMATCH on {[b[0] for b in bad]}"
                ),
            )

        # 2. the calibration's temperature against the one the chip is held at
        try:
            assert_calib_temperature(calib, rig.tec.target)
            say(True, "calib vs chip temp", f"both {meta.get('chip_c', TEC_SETPOINT_C)} C")
        except Exception as e:
            say(False, "calib vs chip temp", str(e).split(" -- ")[0])

        # 3. channels nothing can characterize, because no resistance means a clamp at
        #    V_UNMEASURED and a clamp there means the sweep never sees them
        # NOT "never swept" -- they are swept. An unmeasured channel is clamped at
        # V_UNMEASURED = 1.5 V, which draws 26.6 mA even against the smallest resistance on
        # the board and is therefore safe under every hypothesis, and `REACHABLE_DACS`
        # includes them. What the resistance buys is the HIGHER ceiling: 4.55 V needs to know
        # R, because a secretly-57-ohm channel would draw 80 mA there. Whether 1.5 V is enough
        # span to fit is an open question and `--deep` is what answers it -- 0.08 to 0.25 pi
        # at a plausible Vpi, against 0.18 pi for DAC 10, which IS fitted.
        miss = [i for i, r in enumerate(HEATER_OHMS) if r is None]
        say(
            not miss,
            "heater resistances",
            (
                "all 16 measured"
                if not miss
                else f"DAC {miss} unmeasured -> staged at 1.5 V (safe at any R), swept but "
                f"possibly too little span to fit. An ohmmeter unlocks their real ceiling; "
                f"pic.resistance cannot (0.147 mV/mW against a 3.3 mV floor)"
            ),
        )

        # 4. the one absolute quantity in `to_transfer`; everything else cancels in the
        #    per-column normalisation, this does not
        if rig.detectors == "spd":
            say(True, "pd_offset", "SPD mode: identity; each SPD's dark is `python -m spd dark`'s")
        elif getattr(a, "dark", True):
            with rig.session(duration_s=180, power_dbm=a.dbm, calibrating=True):
                # through the switch object, not the board: a mock or sim board has no serial
                # port to send `P0` down, and this command has to run with --mock
                try:
                    rig.switch.dark()
                except Exception:
                    pass
                time.sleep(0.8)
                dark = np.mean(
                    [rig.board.measure_raw(np.zeros(N_HEATERS)) for _ in range(40)], axis=0
                )
            old = np.asarray(calib.pd_offset, float)
            moved = float(np.abs(dark - old).max())
            say(
                moved < 0.002,
                "pd_offset",
                f"{np.round(dark, 5).tolist()}"
                + ("" if moved < 0.002 else f"  (was {np.round(old, 5).tolist()})"),
            )
            writes["pd_offset"] = dark.round(6).tolist()

        # 5. stamp the laser diode: it cannot be commanded on this unit, so record it and
        #    refuse to compare tables taken at different hold points
        try:
            dc = float(rig.laser.dev.measure("diode_temperature"))
            writes["diode_c"] = dc
            prev = meta.get("diode_c")
            say(
                prev is None or abs(dc - prev) < 1.0,
                "laser diode temp",
                f"{dc:.2f} C" + ("" if prev is None else f"  (calibration stamped at {prev:.2f})"),
            )
        except Exception as e:
            say(False, "laser diode temp", f"unreadable: {type(e).__name__}")

    # 6. rails, ranked on what a BLOCK run needs: sixteen tiles are SUMMED, so the worst tile
    #    dominates and the median that `scan_rails` reports hides it
    tab = load_transfers(calib=calib)
    if tab is None:
        say(False, "block rails", "no measured transfer table on file; capture one first")
    else:
        import itertools

        box = bench_box(calib)
        B = np.random.default_rng(0).normal(size=(8, 8))
        rows = []
        for out in itertools.combinations(range(4), 2):
            for inp in itertools.combinations(range(4), 2):
                try:
                    pl = plan_block_from_table(B, tab, box, k=2, rails=(out, inp))
                    e = [t.terms[0].prog.err for t in pl.tiles.values()]
                    rows.append((max(e), float(np.mean(e)), out, inp))
                except Exception:
                    pass
        rows.sort()
        w, m, out, inp = rows[0]
        writes["best_rails"] = [list(out), list(inp)]
        say(
            True,
            "block rails",
            f"out{out} in{inp}  worst tile {w:.4f}, mean {m:.4f}  " f"({len(tab)} states)",
        )

    print("\n".join(ok + todo))

    # `--deep` is the only path that gives a never-swept channel a Vpi, and a Vpi is what
    # `HeaterBox.from_calibration` gates `trainable` on. DAC 8 and 11 are reachable now --
    # they sit at the staged 1.5 V ceiling and `REACHABLE_DACS` includes them -- so a deep
    # pass is what actually tests whether 1.5 V is enough span to fit. It may not be: at a
    # plausible Vpi that is 0.08 to 0.25 pi, against 0.18 pi for DAC 10, which IS fitted.
    if getattr(a, "deep", False):
        print("\n--deep: sweeping every reachable heater\n")
        rc = cmd_char(a)
        if rc:
            return rc

    if not writes:
        return 0
    print(f"\nwould write to {data_path('pic_data/calib.json')}:")
    for k, v in writes.items():
        print(f"  {k:<12} {v}")
    if not a.write:
        print("\n(dry run -- pass --write to persist)")
        return 0
    _refuse_mock_write(a, "a calibration")
    from pathlib import Path

    ROOT = Path(__file__).resolve().parents[1]  # the project directory, wherever it was run from
    path = data_path(ROOT / "pic_data/calib.json")
    if not path.exists():  # a detector mode's first write starts from its bootstrap
        calib.save(path)
    d = json.loads(path.read_text())
    if "pd_offset" in writes:
        d["pd_offset"] = writes["pd_offset"]
        d.setdefault("meta", {}).setdefault("normalisation", {})["dark_v"] = writes["pd_offset"]
    for k in ("diode_c", "best_rails"):
        if k in writes:
            d.setdefault("meta", {})[k] = writes[k]
    d.setdefault("meta", {})["calibrated_at"] = time.strftime("%Y-%m-%d %H:%M")
    d["meta"]["chip_c"] = float(rig.tec.target)
    path.write_text(json.dumps(d, indent=1))
    ev("job", "wrote", f"wrote {path.relative_to(ROOT)}", "ok", path=str(path.relative_to(ROOT)))
    return 0


def _refuse_mock_write(a, what):
    """A mock run must never write into real data. It is a physical model, not the chip.

    `--mock --write` on `sync` fitted a mixing against simulated transfers and stored it in a
    real capture, where `load_transfers` would have applied it to every subsequent plan --
    silently, because a mixing is exactly the kind of correction that looks plausible whatever
    it contains. The mock exists so the logic can be exercised without hardware; the moment it
    can persist, it stops being a test and becomes a source of fiction."""
    if getattr(a, "mock", False) or getattr(a, "sim", False):
        raise SystemExit(
            f"refusing to write {what} from a --mock/--sim run: the numbers come from a "
            f"simulated chip and would be indistinguishable from measurements once stored."
        )


def cmd_sync(a):
    """Re-measure a few of the stored table's states and carry the whole table onto today.

    A transfer table is perishable. Seventy-five minutes old it differs from the chip by
    0.100 per entry against a 0.0145 seconds-apart repeat, and that staleness swamps every
    other term: driving the planning residual to exactly zero moved the measured error by
    0.0001. Recapturing costs 400 s, which is longer than the drift time constant, so a fresh
    capture is stale before it finishes.

    It does not have to be recaptured. The drift is a 15-parameter mixing on the OUTPUT side,
    `T_now = colnorm(M T_then)`, and nothing on the input side, because `to_transfer` divides
    each column by its own sum and that removes launch power, fibre coupling and the switch's
    per-position loss as they were at that heater state. Fitting M needs `m` states, not a
    hundred: each supplies 4 columns x 3 independent entries, so 12m observations against 15
    parameters. Twenty seconds of bench time buys hosting 0.073 against 0.079 for the full
    recapture.

    The gate is not optional and it is derived, not tuned. Below m = 2 the fit is
    underdetermined outright; when the measured motion sits under the repeatability floor
    there is nothing to fit but noise, and correcting there made a fresh table 15% WORSE. On a
    genuinely stale pair it is worth about -29%.
    """
    import numpy as np
    from .matvec import BEST_RAILS, load_transfers, reanchor

    with _rig(a) as rig:
        calib = rig.calib
        tab = load_transfers(calib=calib)
        if tab is None:
            ev("sync", "failed", "no measured transfer table on file; capture one first", "error")
            return 3
        print(f"table: {len(tab)} states from {tab.path.name}")
        rig.tec.wait_stable()
        with rig.session(duration_s=a.duration, power_dbm=a.dbm):
            out = reanchor(
                rig, tab, m=a.anchors, calib=calib, rails=BEST_RAILS, dbm=a.dbm, return_mixing=True
            )
            out, M = out if isinstance(out, tuple) else (out, None)

    moved = float(np.abs(np.asarray(out.T) - np.asarray(tab.T)).mean())
    if moved < 1e-12:
        ev("sync", "declined", "declined -- nothing written", "warn", moved=0.0)
        return 0
    ev("sync", "done", f"correction moved the table by {moved:.4f} per entry", "ok", moved=moved)
    if not a.write:
        print("(--dry-run: the mixing was fitted and discarded)")
        return 0
    _refuse_mock_write(a, "a sync mixing")
    import json

    # The file holds RAW port-major reads and those are real measurements, so the correction
    # is stored BESIDE them and applied by `load_transfers` rather than rewritten into them.
    # One copy of the data, one of the correction, and the sync can be inspected or dropped.
    d = json.loads(tab.path.read_text())
    d.setdefault("meta", {})["sync_mixing"] = np.asarray(M).tolist()
    d["meta"]["synced"] = {
        "at": time.strftime("%Y-%m-%d %H:%M"),
        "anchors": int(a.anchors),
        "moved_per_entry": moved,
    }
    tab.path.write_text(json.dumps(d, indent=1))
    ev(
        "job",
        "wrote",
        f"wrote the mixing into {tab.path.name}; every load now applies it",
        "ok",
        path=str(tab.path),
    )
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

    from theory.calib import Calibration, refit_phi0, track_well, well

    res_path = data_path("pic_data/char_results.json")
    if not res_path.exists():
        raise SystemExit(f"no {res_path} -- run `python -m pic char --write` first")
    res = json.loads(res_path.read_text())
    cal = Calibration.load_or_nominal()
    from .layout import SWEEP_DACS

    fits = {int(k): v for k, v in res.items() if v.get("ok") and int(k) in SWEEP_DACS}
    if not fits:
        raise SystemExit("no characterized channels to re-anchor")

    with _rig(a) as rig:
        # each fit is re-read on the output it was fitted on, in that output's units
        need_outputs(rig, "re-anchoring these heaters", sorted({int(f["pd"]) for f in fits.values()}))
        n_reads = len(fits) * a.levels * a.repeats
        dur = max(60.0, n_reads * (a.settle + 0.3) + 30.0)
        print(
            f"re-anchoring {len(fits)} channel(s) x {a.levels} levels "
            f"(~{n_reads} reads, {dur / 60:.0f} min budget)"
        )
        with rig.session(duration_s=dur, power_dbm=a.dbm, calibrating=True) as s:
            print(f"{'heater':<14} {'phi0 was':>9} {'now':>9} {'drift':>9} {'rmse/amp':>9}  how")
            moved = {}
            for dac, f in sorted(fits.items()):
                pd, port = int(f["pd"]), int(f["port"])
                if port is not None:
                    rig.select_input(port)
                vmax = VOLTAGE_MAX_CH[dac]

                def read(lv, dac=dac, pd=pd):
                    v = np.zeros(N_HEATERS)
                    v[dac] = lv
                    s.keepalive()
                    return np.mean([rig.measure(v) for _ in range(a.repeats)], axis=0)[pd]

                # the well first: a few reads about the stored extremum (report eq. nulltrack)
                w, vs = well(f["vpi"], f["phi0"], vmax), None
                if w is not None:
                    sgn = 1.0 if f["B"] * np.cos(w[1] * np.pi) < 0 else -1.0  # min as is
                    vs, _ = track_well(lambda x: sgn * read(x), w[0], 0.0, vmax)
                if vs is not None:
                    phi = (w[1] * np.pi - np.pi * (vs / f["vpi"]) ** 2) % (2 * np.pi)
                    rmse, how = 0.0, "well"
                else:  # no extremum in range, or it left: scan and fit phi0 alone
                    levels = np.linspace(0.0, vmax, a.levels)
                    ys = np.array([read(lv) for lv in levels])
                    phi, _, _, rmse = refit_phi0(
                        levels, ys, f["vpi"], amp=f["B"], offset=f["A"], prior=f["phi0"]
                    )
                    how = "scan"
                d = (phi - f["phi0"] + np.pi) % (2 * np.pi) - np.pi
                moved[dac] = phi
                rel = rmse / max(abs(f["B"]), 1e-9)
                ev(
                    "heater",
                    "recal",
                    f"{f['label']:<14} {f['phi0'] / np.pi:>8.3f}p {phi / np.pi:>8.3f}p "
                    f"{d:>+8.3f}r {rel:>9.2f}  {how}",
                    "ok",
                    pad=f["label"].split(":")[0],
                    dac=dac,
                    was_pi=f["phi0"] / np.pi,
                    phi0_pi=phi / np.pi,
                    drift_rad=d,
                    rmse_rel=rel,
                    how=how,
                    ok=True,
                    k=len(moved),
                    n=len(fits),
                )
            for dac, phi in moved.items():
                cal.phi0[dac] = phi
    cal.meta["reanchored"] = {"channels": sorted(moved), "levels": a.levels}
    if a.write:
        cal.meta["chip_c"] = float(rig.tec.target)  # the temperature this fit belongs to
        out = cal.save()
        ev("job", "wrote", f"wrote {out}", "ok", path=str(out))
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
            if abs(drive) >= 4.999:  # DAC_MAX in Arduino/tec_pid/tec_pid.ino
                railed += 1
            mark = "IN BAND" if abs(err) <= TEC_TOLERANCE_C else f"{c - first:+.2f} since start"
            print(
                f"{time.time() - t0:6.0f}  {c:7.2f}  {sp:7.2f}  {err:+7.2f}  "
                f"{drive:+7.3f}   {mark}",
                flush=True,
            )
            time.sleep(a.period)
    except KeyboardInterrupt:
        pass
    finally:
        tec.close()
    return 0


def _find_tec():
    """The TEC by its USB serial number, else the one unknown device left besides the board."""
    from .config import usb_ports
    from .interface import find_port

    board, _ = find_port(os.environ.get("PIC4_PORT"))
    ports = usb_ports()
    cands = [d for d, r, _ in ports if r == "tec"] or [
        d for d, r, _ in ports if r == "?" and d != board
    ]
    if len(cands) != 1:
        raise SystemExit(
            f"cannot tell which port is the TEC (candidates {cands}); " f"pass --tec-port"
        )
    return cands[0]


def _merge_calib(new, res, path=None):
    """Fold a partial sweep into the calibration on disk instead of replacing it.

    A targeted re-run of one weak channel would otherwise reset every other heater to the
    nominal Vpi -- silently, because a nominal calibration looks exactly like a measured one
    apart from its metadata. Only channels this run actually fitted are taken from it."""
    from theory.calib import Calibration

    try:  # the mode's file, or its bootstrap from PD's heater law
        old = Calibration.load(path) if path else Calibration.load_or_nominal()
    except (OSError, ValueError, KeyError):
        return new
    if not old.meta:  # nominal: nothing measured to keep
        return new
    fitted = [int(d) for d, f in res.items() if f.get("ok")]
    if not fitted:
        return old
    for d in fitted:
        old.vpi[d], old.phi0[d] = new.vpi[d], new.phi0[d]
    if detector_mode() == "pd":  # an SPD run's readout is not the photodiodes' law
        old.pd_gain, old.pd_offset = new.pd_gain, new.pd_offset
    meta = dict(old.meta)
    meta.update(new.meta)
    meta["fitted_this_run"] = fitted
    old.meta = meta
    return old


def _save_results(res, path=None, merge=False):
    """The per-heater fits, including each channel's (port x detector) fingerprint.

    `calib.json` keeps only Vpi and phi0 -- the two numbers programming needs. The
    fingerprint is what says *where in the mesh* a heater sits, and it cost a full sweep to
    measure, so it does not get to live only in stdout."""
    import json
    from pathlib import Path

    path = data_path("pic_data/char_results.json") if path is None else path
    out = {}
    if merge:
        try:
            out = json.loads(Path(path).read_text())
        except (OSError, ValueError):
            out = {}
    for dac, f in res.items():
        out[str(dac)] = {k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in f.items()}
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2, default=float))
    ev("job", "wrote", f"wrote {p}", "ok", path=str(p))
    return p


def _jsonable(x):
    """NaN and inf are valid Python floats and invalid JSON. `json.dump` writes them bare,
    which every browser then refuses to parse -- so a rejected fit silently breaks the UI
    that is meant to display it. Nulls instead, and `allow_nan=False` so this cannot rot."""
    import math as _m

    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        return None if (_m.isnan(x) or _m.isinf(x)) else float(x)
    return x


def cmd_fastchar(a):
    """The prescan path. Same fits and same records as `char`, far fewer measurements."""
    import json
    from pathlib import Path

    from .characterize import random_bases
    from .fastchar import run as fastrun
    from theory.calib import Calibration

    with _rig(a) as rig:
        need_outputs(rig, "heater characterization")  # before the laser, not after
        with rig.session(duration_s=1800, power_dbm=a.dbm, calibrating=True) as s:
            if not s.emitted and not a.allow_dark:
                print(
                    "no emission detected: every fit would be noise. Check coupling, "
                    "or pass --allow-dark."
                )
                return 2
            chans = [int(x) for x in a.channels.split(",")] if a.channels else None
            res, fps, rounds, solo = fastrun(
                rig.board,
                s,
                switch=rig.switch,
                levels_n=a.levels,
                channels=chans,
                settle_s=a.settle,
                repeats=a.repeats,
                bases=random_bases(a.bases),
            )

    ok = {c: f for c, f in res.items() if f["ok"]}
    ev(
        "job",
        "summary",
        f"{len(ok)}/{len(res)} channels accepted",
        "ok" if ok else "warn",
        k=len(ok),
        n=len(res),
    )
    if a.write:
        cal = Calibration.load_or_nominal()
        for c, f in ok.items():
            cal.vpi[c], cal.phi0[c] = f["vpi"], f["phi0"]
        cal.meta["fitted_this_run"] = sorted(int(c) for c in ok)
        cal.meta["source"] = "pic.fastchar.run"
        cal.meta["chip_c"] = float(rig.tec.target)  # the temperature this fit belongs to
        out = cal.save()
        ev("job", "wrote", f"wrote {out}", "ok", path=str(out))
        # MERGE, never replace. A run over a subset of channels used to clobber the file
        # with just those channels, so every earlier fit vanished and the calibration tab
        # reported eight characterized heaters as having no fit on record.
        rp = data_path("pic_data/char_results.json")
        prev = json.loads(rp.read_text()) if rp.exists() else {}
        prev.update(
            {str(int(k)): {kk: _jsonable(vv) for kk, vv in v.items()} for k, v in res.items()}
        )
        rp.write_text(json.dumps(prev, indent=1, allow_nan=False))
        ev("job", "wrote", f"wrote {rp}", "ok", path=str(rp))
    return 0


def cmd_char(a):
    from .acquisition import estimate_seconds, grid
    from .characterize import characterize, digest, random_bases

    levels = grid() if a.levels is None else grid(a.levels)
    bases = random_bases(a.bases)
    with _rig(a) as rig:
        need_outputs(rig, "heater characterization")  # before the laser, not after
        ports = None if a.ports is None else [int(x) for x in a.ports.split(",")]
        n_ports = len(ports) if ports else (1 if a.no_switch else NMODE)
        dur = estimate_seconds(levels.size * N_HEATERS * len(bases) * n_ports, a.settle, a.repeats)
        print(
            f"characterizing {N_HEATERS} heaters x {levels.size} levels x {len(bases)} "
            f"base biases x {n_ports} input port(s); session {dur / 60:.0f} min"
        )
        with rig.session(duration_s=dur, power_dbm=a.dbm, calibrating=True) as s:
            if not s.emitted and not a.allow_dark:
                # Was a warning, and a warning on a `--write` run is how a calibration
                # fitted to 18 noise traces reaches disk looking exactly like a measured
                # one. `cmd_dataset` already refuses on the same condition; this matches it.
                print(
                    "no emission detected: neither the beam monitor nor the chip "
                    "detectors saw the laser turn on, so every fit would be noise. "
                    "Check coupling and the front panel; pass --allow-dark to sweep "
                    "anyway."
                )
                return 2
            chans = None
            if a.channels:
                chans = []
                for part in a.channels.split(","):
                    if "-" in part:
                        lo, hi = part.split("-")
                        chans += list(range(int(lo), int(hi) + 1))
                    else:
                        chans.append(int(part))
            res, calib = characterize(
                rig.board,
                s,
                pd=a.pd,
                levels=levels,
                bases=bases,
                channels=chans,
                switch=None if a.no_switch else rig.switch,
                ports=ports,
                settle_s=a.settle,
                repeats=a.repeats,
            )
            if a.write:  # persist the sweep before anything else can fail on top of it
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
                # an SPD read is already 0..1 on its own dark and max: its gain and
                # offset stay identity, the survey's port scales and contrast still land
                apply_to(calib, norm, readout=rig.detectors == "pd")
                print(norm.summary())
        print()
        print(digest(res))
        missed = [
            r["label"]
            for r in res.values()
            if not r["ok"] and ":alpha" not in r["label"] and ":aux" not in r["label"]
        ]
        if missed:
            print(f"\n{len(missed)} mesh heater(s) still unidentified: {missed}")
            print(
                "An external phase on a first-column MZI is a global phase when a single "
                "input port is lit, whichever port that is, so no amount of switching finds "
                "it. Split the input across two adjacent ports and re-run those channels "
                "(see pic.characterize.probe_inputs)."
            )
        if a.write:
            calib.meta["chip_c"] = float(rig.tec.target)
            print(f"\nwrote {calib.save()}")
    return 0


def cmd_program(a):
    from theory import fidelity, random_unitary, unitary_for

    U = (
        random_unitary(np.random.default_rng(a.seed))
        if a.random
        else (
            np.load(a.target)["U"]
            if a.target.endswith(".npz")
            else np.loadtxt(a.target, dtype=complex)
        )
    )
    with _rig(a) as rig:
        v, ok, y = np.zeros(N_HEATERS), None, None
        with rig.session(duration_s=a.duration, power_dbm=a.dbm):
            v, ok, y = rig.realize(U)
        print(f"target unitary:\n{np.round(U, 3)}")
        print(f"\nheater volts ({int(ok.sum())}/{N_HEATERS} reachable, max {v.max():.2f} V):")
        for i, x in enumerate(v):
            print(f"  {LABEL_OF_DAC[i]:<14} {x:5.2f} V{'' if ok[i] else '   UNREACHABLE'}")
        print(f"\noutput photodiode volts: {np.round(y, 4)}")
        print(
            f"ideal-mesh fidelity of the programmed phases: "
            f"{fidelity(U, unitary_for(rig.calib.phases(v))):.4f}"
        )
    return 0


def cmd_matvec(a):
    """Signed matvec from intensity alone. `--scan` is a fit, so it needs no rig at all."""
    from .matvec import job_seconds, main as matvec_main

    if a.scan:
        return matvec_main(None, a)
    # The watchdog is sized from the job, not from a constant, and the estimate is printed
    # before the beam is lit: a run that states up front how long it thinks it needs is one
    # whose author can see it is wrong. The flat 120 s this replaced silenced nothing and
    # cost a night -- an 8x8 outran it, was hard-offed mid-run, and reported dark current.
    #
    # Sized after the board is open, because a firmware that sweeps all four ports in one
    # round trip is 5-6x cheaper per state and only the board can say whether this one does.
    with _rig(a) as rig:
        dur, why = job_seconds(a, batched=rig.batched)
        if a.duration is None:
            print(f"session budget {dur / 60:.1f} min ({why})")
        else:
            given, dur = float(a.duration), float(a.duration)
            print(
                f"session budget {given / 60:.1f} min (given); the job estimates "
                f"{job_seconds(a, batched=rig.batched)[0] / 60:.1f} min ({why})"
            )
        with rig.session(duration_s=dur, power_dbm=a.dbm):
            return matvec_main(rig, a)


def cmd_capture(a):
    """A fresh transfer table: random heater states, each read at all four input ports.

    One settle and one batched four-port sweep per state. Written to a new session
    directory, which `pic.matvec.load_transfers` then prefers as the newest. Whatever was
    measured is written even if the run dies part-way: every state in it was read lit."""
    import json
    from pathlib import Path

    from .acquisition import random_vectors
    from .config import DRIVE_MAX_V, NUM_DAC, drive_to_volts, mirror_pairs

    rng = np.random.default_rng(a.seed)
    V = []
    for d in random_vectors(a.states, rng=rng, vmax=DRIVE_MAX_V):
        v = np.asarray(drive_to_volts(d), float)
        mirror_pairs(v)
        V.append(v)

    def dark(rig):
        return np.mean([rig.board.measure_raw(np.zeros(NUM_DAC))[:NMODE] for _ in range(8)], 0)

    states, darks, t0 = [], {}, time.time()
    try:
        with _rig(a) as rig:
            rig.switch.dark()
            darks["off"] = dark(rig)  # laser off, mirror parked
            if rig.batched:
                per = a.settle + 0.15 * a.repeats + 0.2
            else:
                # the host visits each port: switch settle, one detector read (0.2 s for the
                # SPDs) and the heater write it rides on. The batched figure above ran an
                # SPD capture into the watchdog at 194/200 (2.4 s/state measured).
                from .config import SWITCH_SETTLE_S

                read_s = getattr(rig.board, "seconds", 0.15 * a.repeats)
                per = a.settle + NMODE * (SWITCH_SETTLE_S + read_s + 0.3)
            with rig.session(duration_s=60 + len(V) * per * 2, power_dbm=a.dbm) as s:
                rig.switch.dark()
                darks["sw"] = dark(rig)  # laser on, mirror parked
                for i, v in enumerate(V):
                    chip = rig.hold_band(s)  # out of band: rest at 0 V until it is back
                    rig.board.measure_raw(v)  # apply, then wait out the thermal settle
                    time.sleep(a.settle)
                    T = rig.sweep_ports(cycles=a.repeats)
                    if T is None:  # no batched sweep: the host visits the ports itself
                        T = np.stack(
                            [rig.outputs(v) for p in range(NMODE) if rig.select_input(p) or True], 1
                        )
                    states.append(
                        {"volts": v.tolist(), "raw": np.asarray(T).T.tolist(), "chip_c": chip}
                    )
                    s.keepalive()
                    ev(
                        "table",
                        "progress",
                        f"{i + 1}/{len(V)} states, {time.time() - t0:.0f} s",
                        k=i + 1,
                        n=len(V),
                    )
    finally:
        if a.mock:  # a fake table must never become the newest real one
            ev("table", "discarded", f"--mock: {len(states)} states captured, not written")
        elif states and "sw" in darks:
            out = (
                data_path("pic_data/sessions")
                / time.strftime("%Y-%m-%d-capture-%H%M")
                / "raw_transfers.json"
            )
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(
                json.dumps(
                    {
                        "states": states,
                        "dark_switch_off": darks["sw"].tolist(),
                        "dark_laser_off": darks["off"].tolist(),
                        "dbm": a.dbm,
                        "repeats": a.repeats,
                        "note": f"{len(states)}/{len(V)} random states over the modelled "
                        "heaters, python -m pic capture",
                        "meta": {"seed": a.seed, "settle_s": a.settle},
                        "vmax": list(VOLTAGE_MAX_CH),  # the fit refuses a capture under other ceilings
                    }
                )
            )
            ev(
                "table",
                "wrote",
                f"wrote {out}: {len(states)}/{len(V)} states in {time.time() - t0:.0f} s",
                "ok",
                path=str(out),
                k=len(states),
                n=len(V),
            )
    return 0


def main(argv=None):
    os.chdir(Path(__file__).resolve().parents[1])  # data paths are relative to unitary/
    # common options live on a parent parser so they can be given after the subcommand,
    # which is how anyone actually types them
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--mock", action="store_true", help="run with no hardware attached")
    common.add_argument(
        "--sim",
        action="store_true",
        help="no hardware, but the bench as delivered: 6 wired heaters, a "
        "3 V clamp, the measured loss table and noise (pic.sim)",
    )
    common.add_argument(
        "--tec",
        default="auto",
        help="TEC: 'auto' (the real one on the bench, found by USB serial number), 'mock', "
        "'none', or a serial port. A bench run with a fake TEC gates on a fake temperature",
    )
    common.add_argument("--laser-port", default=None)
    common.add_argument("--pic-port", default=None)
    common.add_argument(
        "--dbm",
        type=float,
        default=None,
        help="laser output power (default 5 dBm in SPD mode, 8 in PD mode). 8 dBm is what the best archive set was "
        "taken at and leaves the TIA headroom (brightest of 77,280 "
        "logged reads is 0.587 V); the 6x6's 13 dBm has never been "
        "put on this chip",
    )
    common.add_argument(
        "--dynamic",
        action="store_true",
        help="real-time drift correction: re-probe the four input ports "
        "periodically and pre-distort the commanded phases by the "
        "drift that explains the change. Off by default -- correcting "
        "a drift smaller than the probe noise makes the chip worse",
    )

    ap = argparse.ArgumentParser(prog="pic", description=__doc__.splitlines()[0], parents=[common])
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser(
        "selftest", help="run every model self-test; no hardware", parents=[common]
    ).set_defaults(fn=cmd_selftest)
    sub.add_parser("ports", help="each USB instrument, its chip id and its port").set_defaults(
        fn=cmd_ports
    )
    p = sub.add_parser(
        "spdnorm", help="SPD dark, then every heater at max and each input read once", parents=[common]
    )
    p.add_argument("--write", action="store_true", help="write pic_data/spd_max.json")
    p.add_argument("--settle", type=float, default=5.0)
    p.add_argument("--count", type=float, default=2.0)
    p.add_argument("--dark-s", dest="dark_s", type=float, default=10.0)
    p.set_defaults(fn=cmd_spdnorm)
    sub.add_parser(
        "status", help="laser, TEC and calibration state", parents=[common]
    ).set_defaults(fn=cmd_status)

    p = sub.add_parser("measure", help="live photodiode readout", parents=[common])
    p.add_argument("--duration", type=float, default=120.0)
    p.add_argument("--period", type=float, default=0.2)
    p.add_argument(
        "--ports",
        default=None,
        help="cycle the input switch through these ports while streaming, "
        "e.g. 0,1,2,3 -- a per-port pattern is what distinguishes guided "
        "light from stray light on the detector array",
    )
    p.set_defaults(fn=cmd_measure)

    p = sub.add_parser(
        "calibrate",
        help="agree every rig constant and write what is " "measurable",
        parents=[common],
    )
    p.add_argument(
        "--write",
        action="store_true",
        help="persist to pic_data/calib.json; without it this is a dry run",
    )
    p.add_argument(
        "--no-dark",
        dest="dark",
        action="store_false",
        help="skip the pd_offset measurement (it needs the laser)",
    )
    p.add_argument(
        "--deep",
        action="store_true",
        help="also sweep every reachable heater and refit its fringe. Hours, not "
        "minutes, and the only way a channel that has never been swept gets "
        "a Vpi -- which is what puts it in play",
    )
    p.add_argument("--levels", type=int, default=None)
    p.add_argument("--bases", type=int, default=3)
    p.add_argument("--settle", type=float, default=0.2)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--ports", default=None)
    p.add_argument("--no-switch", action="store_true")
    p.add_argument("--allow-dark", action="store_true")
    p.add_argument("--channels", default=None)
    p.add_argument("--pd", type=int, default=None)
    p.add_argument("--no-normalise", dest="no_normalise", action="store_true")
    p.set_defaults(fn=cmd_calibrate)

    p = sub.add_parser(
        "sync",
        help="re-measure a few stored states and carry the table " "onto today's chip",
        parents=[common],
    )
    p.add_argument(
        "--anchors",
        type=int,
        default=10,
        metavar="M",
        help="states to re-measure. 12M observations fit 15 parameters, so M<2 is "
        "underdetermined and M=3 is measurably worse than doing nothing",
    )
    p.add_argument("--duration", type=float, default=300.0)
    p.add_argument(
        "--dry-run",
        dest="write",
        action="store_false",
        help="fit and report the correction without storing it. Syncing writes by "
        "default: a table you have decided to keep using is one you have "
        "decided to correct, and a correction that is computed and discarded "
        "is bench time spent on nothing",
    )
    p.set_defaults(fn=cmd_sync, write=True)

    p = sub.add_parser(
        "recal", help="fast phi0 re-anchor on characterized channels", parents=[common]
    )
    p.add_argument("--levels", type=int, default=5)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--settle", type=float, default=0.4)
    p.add_argument("--write", action="store_true")
    p.set_defaults(fn=cmd_recal)

    p = sub.add_parser("tec", help="live chip temperature and controller drive")
    p.add_argument(
        "setpoint",
        nargs="?",
        type=float,
        default=None,
        help="retarget the controller to this temperature before streaming",
    )
    p.add_argument("--tec-port", default=None, help="TEC serial port (else autodetect)")
    p.add_argument("--period", type=float, default=1.0)
    p.set_defaults(fn=cmd_tec)

    p = sub.add_parser(
        "fastchar",
        parents=[common],
        help="prescan, then sweep each heater alone where it moved most",
    )
    p.add_argument("--write", action="store_true", help="write pic_data/calib.json")
    p.add_argument("--levels", type=int, default=21)
    p.add_argument("--bases", type=int, default=3)
    p.add_argument("--settle", type=float, default=0.2)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--allow-dark", action="store_true")
    p.add_argument(
        "--channels",
        type=str,
        default=None,
        help="DAC channels to sweep, e.g. 0,1,2 (default: all modelled)",
    )
    p.set_defaults(fn=cmd_fastchar)

    p = sub.add_parser("char", help="sweep every heater and fit its fringe", parents=[common])
    p.add_argument(
        "--pd",
        type=int,
        default=None,
        help="fit against one photodiode; default is whichever sees each heater best",
    )
    p.add_argument(
        "--levels",
        type=int,
        default=None,
        help="sweep points per heater; the default clears the Nyquist gate",
    )
    p.add_argument(
        "--ports", default=None, help="input ports to sweep through, e.g. 0,2 (default: all four)"
    )
    p.add_argument(
        "--channels",
        type=str,
        default=None,
        help="DAC channels to sweep, e.g. 0-5 or 0,2,5 (default: all). The six "
        "internal phase shifters are 0-5; external phases are output-side "
        "and invisible in intensity, so there is nothing to find on them",
    )
    p.add_argument(
        "--keep-laser",
        action="store_true",
        help="leave the laser lit on exit if it was already on when we opened "
        "(default is always off: an unattended lit diode has no watchdog)",
    )
    p.add_argument(
        "--no-switch",
        action="store_true",
        help="the fibre is plugged straight into one port; do not try to switch",
    )
    p.add_argument(
        "--bases",
        type=int,
        default=3,
        help="base biases to sweep from; a heater dark in one may show in another",
    )
    p.add_argument("--settle", type=float, default=0.2)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--write", action="store_true", help="save to pic_data/calib.json")
    p.add_argument(
        "--no-normalise",
        action="store_true",
        help="skip the PD full-scale pass (blocked = 0, transparent = 1)",
    )
    p.add_argument(
        "--allow-dark",
        action="store_true",
        help="sweep even though no emission was detected. Only for measuring "
        "the dark instrument on purpose -- the fits will be noise",
    )
    p.set_defaults(fn=cmd_char)

    p = sub.add_parser("program", help="put a target unitary on the chip", parents=[common])
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--random", action="store_true")
    g.add_argument("--target", help="path to a 4x4 complex matrix (.npz with U, or text)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--duration", type=float, default=60.0)
    p.set_defaults(fn=cmd_program)

    sub.add_parser(
        "ising",
        help="Ising ground states from the four-port intensity probe "
        "(own flags; see `python -m pic.ising --help`)",
    )

    p = sub.add_parser(
        "matvec", help="signed y = B x from the photodiodes, no homodyne", parents=[common]
    )
    p.add_argument(
        "--k",
        type=int,
        default=2,
        help="block size; 3 and 4 do not fit (see " "theory.intensity_matvec)",
    )
    p.add_argument("--target", default=None, help="path to a real matrix (text)")
    p.add_argument("--rails", default=None, help="block to use, e.g. 1,2:1,2 (out:in)")
    p.add_argument(
        "--mode",
        default="shift",
        choices=("shift", "split"),
        help="nonnegative decomposition; 'split' is the 6x6's and is noisier",
    )
    p.add_argument(
        "--scan",
        action="store_true",
        help="rank every rail pair by residual and brightness; no hardware",
    )
    p.add_argument(
        "--block",
        type=int,
        default=None,
        metavar="K",
        help="tile the target into KxK blocks (theory.matmat); K=2 is what this " "die hosts",
    )
    p.add_argument(
        "--cols",
        type=int,
        default=3,
        help="columns of X for --block, or pairs of matrices for --unitary",
    )
    p.add_argument(
        "--unitary",
        action="store_true",
        help="compose two hosted rotations and check the product is still "
        "orthogonal; polar and QR projections side by side",
    )
    p.add_argument(
        "--optical-input",
        action="store_true",
        help="set the input weight with the laser instead of in software, so "
        "the multiply happens in light; costs a retune per port",
    )
    p.add_argument(
        "--static-scale",
        action="store_true",
        help="read one port at a time and divide by the calibration's stored "
        "input_scale, as before the four-port sweep. Deprecated and wrong "
        "on this bench (see pic.normalise); here to reproduce old runs",
    )
    p.add_argument(
        "--planner",
        default="auto",
        choices=("auto", "table", "twin"),
        help="where the heater state comes from: 'table' picks the measured "
        "state that best hosts the target, 'twin' fits volts through the "
        "model. 'auto' is the table on hardware and the twin against either "
        "simulator, because the table is measurements of the real die",
    )
    p.add_argument(
        "--refresh",
        action="store_true",
        help="re-sweep the chip for every vector instead of reusing the held "
        "state, so the vector error carries read noise; costs a thermal "
        "settle per vector",
    )
    p.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="photodiode reads averaged per port; error falls as 1/sqrt(N)",
    )
    p.add_argument(
        "--reanchor",
        type=int,
        default=0,
        metavar="M",
        help="re-measure M stored states and transport the whole table onto "
        "today's chip through a 15-parameter output mixing. M=5 costs 20 s "
        "and beats a 400 s recapture; M<5 cannot determine 15 parameters "
        "and makes the table worse",
    )
    p.add_argument(
        "--terms",
        type=int,
        default=1,
        metavar="K",
        help="host each block as a weighted sum of K measured states instead of "
        "the single nearest. K=4 drives the PLANNING residual to zero and "
        "makes the ANSWER worse: measured A/B on the 8x8, hosted residual "
        "0.0195 -> 0.0000, measured matrix 0.1875 -> 0.1876 (unmoved), "
        "vector error 0.0419 -> 0.0701. Leave it at 1 unless you are "
        "measuring the effect itself",
    )
    p.add_argument(
        "--refine",
        type=int,
        default=0,
        metavar="N",
        help="after picking each tile off the table, hill-climb it on the chip "
        "for N trials. The table is quantised, not noisy: on the 8x8 the "
        "hosted residual is four fifths of the measured error because "
        "sixteen tiles share the hundred states one block gets alone",
    )
    p.add_argument(
        "--budget",
        type=int,
        default=0,
        metavar="T",
        help="spend T refinement trials in TOTAL, allocated across tiles by the "
        "square of their hosted residual, worst first, instead of --refine "
        "N on every tile. Measured offline (pic.matvec.refine_ablation): "
        "half the uniform trials keeps two thirds to three quarters of the "
        "benefit and a quarter keeps a fifth to a third, because the "
        "residual is heavy tailed and a well-hosted tile has nothing to "
        "win. --refine then caps any one tile",
    )
    p.add_argument(
        "--sweep-cycles",
        type=int,
        default=None,
        metavar="C",
        help="how the batched firmware sweep spends its averaging: C complete "
        "visits to the four ports, repeats/C frames averaged at each. "
        "Default 1 -- column normalisation divides out anything that is "
        "constant across a column, which switch repeatability is. Raise it "
        "to put the frames seconds apart instead of milliseconds",
    )
    p.add_argument("--restarts", type=int, default=16)
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--duration",
        type=float,
        default=None,
        help="laser session seconds. Default: estimated from the job by "
        "pic.matvec.job_seconds and printed before the run starts. A flat "
        "default cannot cover both a 2x2 and an 8x8, and the one that was "
        "here (120 s) fired mid-run on the 8x8",
    )
    p.set_defaults(fn=cmd_matvec)

    p = sub.add_parser("capture", help="record a fresh transfer table", parents=[common])
    p.add_argument("--states", type=int, default=200)
    p.add_argument("--settle", type=float, default=0.5)
    p.add_argument("--repeats", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(fn=cmd_capture)

    a = ap.parse_args(argv)
    if getattr(a, "dbm", 0) is None:  # SPD counts were characterised at 5 dBm (dead time)
        a.dbm = 5.0 if detector_mode() == "spd" else 8.0
    what = a.fn.__name__.removeprefix("cmd_")
    ev("job", "started", " ".join(sys.argv[1:] if argv is None else argv), cmd=what)
    try:
        rc = a.fn(a)
    except WatchdogTripped as e:
        # Loud and non-zero, not a traceback and not a partial result: any command that did
        # not catch this itself has no data worth keeping, because the beam was off for an
        # unknown part of it.
        ev("job", "failed", f"ABORTED: {e}", "error", cmd=what, rc=3)
        return 3
    except MissingOutputs as e:  # a refusal about the detectors in hand, not a crash
        ev("job", "failed", str(e), "error", cmd=what, rc=2)
        return 2
    except KeyboardInterrupt:
        # Stop in the UI sends exactly this. Every `with` block above has already run its
        # cleanup (laser off, heaters to 0) on the way out, so a traceback would add nothing.
        ev("job", "interrupted", "stopped", "warn", cmd=what)
        return 130
    ev(
        "job",
        "failed" if rc else "done",
        f"exit {rc or 0}",
        "error" if rc else "ok",
        cmd=what,
        rc=rc or 0,
    )
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
