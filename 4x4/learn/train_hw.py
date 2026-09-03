"""Train the surrogates directly from the chip, one lit round at a time.

    python -m learn.train_hw --sim  --rounds 4 --n-per-round 400
    python -m learn.train_hw --rounds 8 --n-per-round 200 --ports 0,1,2,3

Each round lights the laser, samples random operating points through each input port,
measures the outputs, appends to a growing buffer, refits both surrogates and checkpoints.
The laser is on only while collecting and off during the fit. Ctrl-C saves and exits.

Two models are fitted on the same buffer every round, which is the point: the physics fit
(`learn.unitary_fit`, 52 parameters) and the pruned network (`learn.dpnn`, a couple of
thousand). Watching them side by side on identical data is how you tell a modelling
problem from a measurement problem -- when both stall at the same R^2, the ceiling is the
instrument, not the model.

Operating points are drawn in the uniform **drive** variable (`pic.config.drive_to_volts`)
and converted per channel on the way to the board. That is not cosmetic: the heaters come
in two resistance groups, and a flat 0..3 V draw would put 1.7x its rated current through
every 60-ohm channel.

The input port is a loop index, not a fibre position -- there is a 1x4 switch in front of
the chip. Visit all four: data through one port leaves phase combinations unobservable (see
`learn.unitary_fit`), and the port is a feature of the network (`learn.dpnn`), so a
single-port buffer trains three quarters of a one-hot on nothing.

Prefer `--sim` over `--mock` for a hardware-free rehearsal of *this* board. `--mock` is the
idealised chip, Vpi near 1.5 V, so its heaters sweep three to five fringes over their range
and the network needs tens of thousands of samples to chase that; the delivered chip's Vpi
is 4.24-5.32 V, half a fringe at the ceiling, which the network fits from a few hundred.
A low `--mock` R^2 next to a physics fit at 1.0000 is that gap, not a broken feature set.

Run `./do char --write` first. The physics fit is a refiner, not a search: from a nominal
start it plateaus, and from the Vpi that characterization recovers it converges. This picks
up `pic_data/calib.json` automatically when one exists.
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

from pic import Rig
from pic.acquisition import estimate_seconds, random_vectors, settled_read
from pic.config import (
    DRIVE_MAX_V, DRIVE_SCALE, NUM_OUT, OUT_PDS, SWITCH_SETTLE_S, TEC_SETPOINT_C,
    drive_to_volts,
)
from pic.session import WatchdogTripped
from theory.calib import Calibration
from theory.clements import NMODE
from theory.layout import HEATERS, N_HEATERS

from . import dpnn, unitary_fit

CKPT = "runs/hw"
TABLE_CKPT = "runs/table"

# Every channel the board can drive that the mesh model uses. Aux is outside the model and a
# dark channel reaches the chip as 0 V whatever it is commanded, so neither is worth a
# sample. This is `--channels all`, not the default: see `pick_channels`.
DRIVABLE_DACS = np.array([h.h for h in HEATERS
                          if h.role != "aux" and DRIVE_SCALE[h.h] > 0], int)


def pick_channels(spec):
    """`--channels`: which DAC channels a round randomises.

    The default 'model' is `dpnn.MODEL_DACS` -- the six the bench says move a detector, and
    the six the network has features for. 'all' adds the external channels, which is the
    experiment that would prove one of them live; it costs nothing to take and the physics
    fit does model them, but until one modulates it only adds unexplained variance to the
    network's residual."""
    if spec == "model":
        return np.asarray(dpnn.MODEL_DACS, int)
    if spec == "all":
        return DRIVABLE_DACS
    return np.array(sorted({int(x) for x in spec.split(",")}), int)


def collect_round(rig, n, *, dbm, settle_s, repeats, rng, ports, channels=None):
    """One lit session of `n` random operating points, split evenly over `ports`.

    Blocked by port rather than interleaved: the switch needs about a second to settle
    after a SET, so cycling it per sample would spend most of the round waiting. Returns
    (drive, port index, telemetry, Y)."""
    channels = pick_channels("model") if channels is None else np.asarray(channels, int)
    ports = list(ports)
    per = max(1, n // len(ports))
    dur = estimate_seconds(per * len(ports), settle_s, repeats) + 3 * len(ports) * SWITCH_SETTLE_S
    Ds, Ps, Ts, Ys = [], [], [], []
    with rig.session(duration_s=dur, power_dbm=dbm) as s:
        if not s.emitted:
            raise RuntimeError("laser reports on but no emission detected -- check coupling "
                               "and the front panel before trusting any of this data")
        # A sample is appended only after its read has returned, and `settled_read` raises
        # once the watchdog has hard-offed the laser -- so the buffer this keeps contains
        # only points measured with the beam on, whichever of the two exits it takes.
        try:
            for port in ports:
                rig.select_input(port)
                for d in random_vectors(per, rng=rng, channels=channels, vmax=DRIVE_MAX_V):
                    if s.expired():
                        print("    watchdog deadline reached; keeping the partial round "
                              "(every sample below was taken before it).")
                        return (np.asarray(Ds), np.asarray(Ps, int), np.asarray(Ts),
                                np.asarray(Ys))
                    y = settled_read(rig.board, drive_to_volts(d), settle_s, repeats)
                    bfm, dtemp, mA, chip = s.telemetry()
                    Ds.append(d)
                    Ps.append(port)
                    Ts.append([dbm, bfm, dtemp, mA, chip])
                    Ys.append(y[list(OUT_PDS)])
                    s.keepalive()
        except WatchdogTripped as e:
            print(f"    {e}")
            print(f"    keeping the {len(Ds)} sample(s) taken before the trip.")
    return np.asarray(Ds), np.asarray(Ps, int), np.asarray(Ts), np.asarray(Ys)


def table_buffer(paths):
    """A stored four-port capture, in the shape `collect_round` returns.

    A capture holds one `raw[port][pd]` block per heater state, so every state is four
    samples of the same mesh read through four different launches -- exactly what a round
    that visits all four ports produces, and the encoding `learn.dpnn.port_features` and
    `learn.unitary_fit`'s `X` both already take. Targets stay un-dark-subtracted because
    `InstrumentModel` fits a per-detector offset; handing it a floor already removed would
    make that parameter absorb the error twice.

    `bfm` and `mA` are laser telemetry a capture does not record. They are filled with the
    centre of their fixed scale in `dpnn._SCALES`, so they normalise to exactly 0 and
    contribute nothing past the bias -- an absent input, not an invented one. A capture is
    a single power in a single session, so the three that ARE recorded are constant across
    the buffer anyway and the whole telemetry block is inert here by construction."""
    from pic.config import volts_to_drive
    Ds, Ps, Ts, Ys = [], [], [], []
    for path in paths:
        d = json.loads(open(path).read())
        vmax = np.asarray(d["vmax"], float)
        # The drive variable is volts/ceiling, so a capture taken under a different clamp
        # table would silently rescale every feature. Refuse rather than rescale.
        if not np.allclose(vmax, np.asarray(DRIVE_SCALE, float) * DRIVE_MAX_V, atol=1e-6):
            raise ValueError(f"{path} was taken at ceilings {vmax.tolist()}, but this tree "
                             f"is configured for "
                             f"{(np.asarray(DRIVE_SCALE) * DRIVE_MAX_V).tolist()}. The drive "
                             f"variable is per-channel, so the two are not comparable.")
        tel = [d["dbm"], dpnn._SCALES["bfm"][0], d.get("diode_c", dpnn._SCALES["diode_temp"][0]),
               dpnn._SCALES["mA"][0], d.get("chip_c", TEC_SETPOINT_C)]
        for st in d["states"]:
            raw = np.asarray(st["raw"], float)          # (port, pd)
            drive = volts_to_drive(np.asarray(st["volts"], float))
            for port in range(NMODE):
                Ds.append(drive)
                Ps.append(port)
                Ts.append(tel)
                Ys.append(raw[port][list(OUT_PDS)])
        print(f"    {path}: {len(d['states'])} states x {NMODE} ports at "
              f"+{d['dbm']:g} dBm, chip {d.get('chip_c', float('nan')):.2f} C")
    return np.asarray(Ds), np.asarray(Ps, int), np.asarray(Ts), np.asarray(Ys)


def varying_channels(D, tol: float = 1e-6):
    """The channels a buffer actually moved.

    The online default is `dpnn.MODEL_DACS`, the six theta the bench had shown modulate a
    detector when that list was written. A stored capture may have swept more -- the 25 C
    table moved eight -- and a channel that moved the chip but has no feature column lands
    in the network's residual as unexplained variance. Reading it off the data keeps the
    two in step without either list having to be maintained."""
    D = np.atleast_2d(np.asarray(D, float))
    return np.flatnonzero((D.max(0) - D.min(0) > tol) & (np.asarray(DRIVE_SCALE) > 0))


def main(argv=None):
    ap = argparse.ArgumentParser(prog="learn.train_hw", description=__doc__.splitlines()[0])
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--n-per-round", type=int, default=100)
    ap.add_argument("--ports", default=",".join(str(p) for p in range(NMODE)),
                    help="input ports to visit each round, e.g. 0,2 (default: all four)")
    ap.add_argument("--channels", default="model",
                    help="DAC channels to randomise: 'model' (the six the network has "
                         "features for), 'all' (every drivable mesh channel), or a list")
    ap.add_argument("--dbm", type=float, default=8.0)
    ap.add_argument("--settle", type=float, default=0.2)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--from-table", default=None,
                    help="fit a stored capture instead of lighting the chip: one or more "
                         "raw-transfer JSON paths, comma separated. No hardware, one "
                         "round, and the network gets a feature per channel the capture "
                         "actually moved.")
    ap.add_argument("--out", default=None,
                    help=f"checkpoint directory (default {CKPT}, or {TABLE_CKPT} with "
                         f"--from-table -- a stored capture carries no laser telemetry, so "
                         f"its buffer must not be resumed into a lit one)")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--epochs", type=int, default=200, help="DPNN epochs per round")
    ap.add_argument("--steps", type=int, default=3000, help="physics-fit steps per round")
    ap.add_argument("--restarts", type=int, default=8,
                    help="physics-fit restarts; only the first round needs many")
    ap.add_argument("--calib", default=None,
                    help="bootstrap calibration to start the physics fit from "
                         "(default: pic_data/calib.json if it exists)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mock", action="store_true",
                    help="no hardware: the idealised chip (see the note above)")
    ap.add_argument("--sim", action="store_true",
                    help="no hardware, but the bench as delivered (pic.sim)")
    ap.add_argument("--tec", default="mock")
    ap.add_argument("--laser-port", default=None)
    ap.add_argument("--pic-port", default=None)
    a = ap.parse_args(argv)

    a.out = a.out or (TABLE_CKPT if a.from_table else CKPT)
    offline = net_channels = None
    if a.from_table:
        print(f"fitting stored captures -- no hardware:")
        offline = table_buffer([p for p in a.from_table.split(",") if p])
        net_channels = varying_channels(offline[0])
        a.rounds, a.resume = 1, False

    ports = [int(x) for x in a.ports.split(",")]
    channels = pick_channels(a.channels)
    extra = sorted(set(channels.tolist()) - set(dpnn.MODEL_DACS.tolist()))
    os.makedirs(a.out, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    buf = {"D": np.zeros((0, N_HEATERS)), "ports": np.zeros(0, int),
           "tel": np.zeros((0, len(dpnn.TELEMETRY))), "Y": np.zeros((0, NUM_OUT))}
    calib = Calibration.load(a.calib) if a.calib else Calibration.load_or_nominal()
    if not calib.meta:
        print("no characterization found -- the physics fit will start from the nominal "
              "heater law and is unlikely to converge. Run `./do char --write` first.")
        calib = None
    else:
        print(f"bootstrap calibration: {calib.meta}")
    meta = {"rounds_done": 0}
    bpath = os.path.join(a.out, "buffer.npz")
    if a.resume and os.path.exists(bpath):
        b = np.load(bpath)
        buf = {k: b[k] for k in ("D", "ports", "tel", "Y")}
        meta = json.load(open(os.path.join(a.out, "meta.json")))
        print(f"resumed: {len(buf['D'])} samples, {meta['rounds_done']} rounds done")

    if extra:
        print(f"channels {extra} are being driven but are not network features "
              f"(learn.dpnn.MODEL_DACS); whatever they do lands in the physics fit's "
              f"phases and in the network's residual, so read the two R^2 accordingly.")
    kind = "mock" if (a.mock or a.sim) else "hw"
    rig = None if offline else Rig(
        laser=kind, board="sim" if a.sim else kind, switch=kind,
        tec="mock" if (a.mock or a.sim) else a.tec,
        laser_port=a.laser_port, pic_port=a.pic_port).open()
    r0 = meta["rounds_done"]
    try:
        for r in range(r0, r0 + a.rounds):
            t0 = time.time()
            if offline:
                D, P, T, Y = offline
            else:
                print(f"\n[round {r + 1}] collecting {a.n_per_round} points over ports "
                      f"{ports} ...")
                D, P, T, Y = collect_round(rig, a.n_per_round, dbm=a.dbm, settle_s=a.settle,
                                           repeats=a.repeats, rng=rng, ports=ports,
                                           channels=channels)
            buf = {"D": np.vstack([buf["D"], D]),
                   "ports": np.concatenate([buf["ports"], P]),
                   "tel": np.vstack([buf["tel"], T]), "Y": np.vstack([buf["Y"], Y])}
            print(f"    {len(D)} points in {time.time() - t0:.0f}s; buffer {len(buf['D'])}")

            X = np.eye(NMODE, dtype=complex)[buf["ports"]]
            phys, calib, err, r2p = unitary_fit.fit(
                drive_to_volts(buf["D"]), buf["Y"], X, calib0=calib, steps=a.steps,
                restarts=1 if r > r0 else a.restarts, seed=a.seed)
            np_phys = phys.n_params()   # 52 at 16 heaters; it was 56 when there were 18
            model, norm, r2d, dmeta = dpnn.fit(buf["D"], buf["ports"], buf["tel"], buf["Y"],
                                               channels=net_channels, epochs=a.epochs,
                                               seed=a.seed)

            calib.save(os.path.join(a.out, "calib.json"))
            dpnn.save_ckpt(a.out, model, norm, buf, dmeta)
            np.savez(bpath, **buf)
            meta.update(source=a.from_table or ("sim" if a.sim else kind),
                        rounds_done=r + 1, n_samples=int(len(buf["D"])),
                        r2_physics=float(r2p), r2_dpnn=float(np.mean(r2d)),
                        n_params_physics=int(np_phys), n_params_dpnn=int(dmeta["n_params"]),
                        dpnn_channels=dmeta["channels"],
                        ports=sorted(set(buf["ports"].tolist())))
            json.dump(meta, open(os.path.join(a.out, "meta.json"), "w"), indent=2)
            print(f"    physics ({np_phys}p) R2 {r2p:+.4f} | "
                  f"dpnn ({dmeta['n_params']}p) R2 {float(np.mean(r2d)):+.4f} -> {a.out}/")
    except KeyboardInterrupt:
        print("\ninterrupted -- last checkpoint is saved.")
    finally:
        if rig is not None:
            rig.close()

    if len(meta.get("ports", [])) < 2:
        print("\nNOTE: all data came through one input port. Some phase combinations are "
              "unobservable that way, and three quarters of the network's port one-hot "
              "never varied. Add ports before trusting either surrogate.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
