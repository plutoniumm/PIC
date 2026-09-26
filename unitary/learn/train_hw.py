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
import shutil
import time

import numpy as np

from pic import Rig
from pic.acquisition import estimate_seconds, random_vectors, settled_read
from pic.config import (
    DRIVE_MAX_V,
    DRIVE_SCALE,
    NUM_OUT,
    OUT_PDS,
    SWITCH_SETTLE_S,
    TEC_SETPOINT_C,
    drive_to_volts,
    drive_volts_raw,
)
from pic.__main__ import _find_tec
from pic.config import data_path, port_of
from pic.interface import MissingOutputs, need_outputs
from pic.devices.tec import auto_tec
from pic.log import ev
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
DRIVABLE_DACS = np.array([h.h for h in HEATERS if h.role != "aux" and DRIVE_SCALE[h.h] > 0], int)


def pick_channels(spec):
    """`--channels`: which DAC channels a round randomises.

    The default 'model' is `dpnn.MODEL_DACS`, every theta and phi: what a run drives, and
    what the network has features for. 'all' adds the output phases, which cannot move an
    intensity."""
    if spec == "model":
        return np.asarray(dpnn.MODEL_DACS, int)
    if spec == "all":
        return DRIVABLE_DACS
    return np.array(sorted({int(x) for x in spec.split(",")}), int)


PROGRAM_JITTER = 0.1  # of DRIVE_MAX_V: spread around each program, so a point is not a run


def program_drives(n, rng, calib):
    """`n` drive vectors where runs actually go: a Haar-random 4x4 orthogonal U, programmed
    through the calibration exactly as `pic.apply` does, clipped at each channel's ceiling,
    then jittered. A run pins about 7 of the 12 modelled heaters at their ceiling (bench,
    2026-09-25), a corner uniform sampling never reached (0 of 3189 points), and the model
    was off by 0.11-0.21 on every run there."""
    from scipy.stats import ortho_group

    from pic.config import VOLTAGE_MAX_CH, mirror_pairs, volts_to_drive
    from theory.program import phases_for

    vmax = np.asarray(VOLTAGE_MAX_CH, float)
    out = []
    for _ in range(n):
        U = ortho_group.rvs(NMODE, random_state=int(rng.integers(2**31)))
        v, _ = calib.volts(phases_for(U), vmax=vmax)
        d = volts_to_drive(np.clip(v, 0.0, vmax))
        d = d + rng.normal(0.0, PROGRAM_JITTER * DRIVE_MAX_V, d.size) * (d > 0)
        # a bonded pair is one heater: the jitter must not split it (the board refuses)
        out.append(mirror_pairs(np.clip(d, 0.0, DRIVE_MAX_V)))
    return out


def collect_round(rig, n, *, dbm, settle_s, repeats, rng, ports, channels=None, calib=None,
                  programs=0.5):
    """One lit session of `n` operating points, split evenly over `ports`: a `programs`
    share where runs go (`program_drives`, needs `calib`), the rest uniform over `channels`.

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
            raise RuntimeError(
                "laser reports on but no emission detected -- check coupling "
                "and the front panel before trusting any of this data"
            )
        # A sample is appended only after its read has returned, and `settled_read` raises
        # once the watchdog has hard-offed the laser -- so the buffer this keeps contains
        # only points measured with the beam on, whichever of the two exits it takes.
        try:
            for port in ports:
                rig.select_input(port)
                k = round(per * programs) if calib is not None else 0
                draws = list(random_vectors(per - k, rng=rng, channels=channels, vmax=DRIVE_MAX_V))
                draws += program_drives(k, rng, calib)
                for i in rng.permutation(len(draws)):
                    d = draws[i]
                    if s.expired():
                        ev(
                            "dpnn",
                            "partial",
                            "watchdog deadline reached; keeping the partial round "
                            "(every sample below was taken before it).",
                            "warn",
                            points=len(Ds),
                        )
                        return (np.asarray(Ds), np.asarray(Ps, int), np.asarray(Ts), np.asarray(Ys))
                    rig.hold_band(s)  # out of band: rest at 0 V until it is back
                    y = settled_read(rig.board, drive_to_volts(d), settle_s, repeats)
                    bfm, dtemp, mA, chip = s.telemetry()
                    Ds.append(d)
                    Ps.append(port)
                    Ts.append([dbm, bfm, dtemp, mA, chip])
                    Ys.append(y[list(OUT_PDS)])
                    s.keepalive()
                    if len(Ds) % 10 == 0:  # within-round progress: a round is minutes long
                        ev("dpnn", "sample", k=len(Ds), n=per * len(ports))
        except WatchdogTripped as e:
            ev(
                "dpnn",
                "partial",
                f"{e} Keeping the {len(Ds)} sample(s) taken before the trip.",
                "warn",
                points=len(Ds),
            )
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
            raise ValueError(
                f"{path} was taken at ceilings {vmax.tolist()}, but this tree "
                f"is configured for "
                f"{(np.asarray(DRIVE_SCALE) * DRIVE_MAX_V).tolist()}. The drive "
                f"variable is per-channel, so the two are not comparable."
            )
        tel = [
            d["dbm"],
            dpnn._SCALES["bfm"][0],
            d.get("diode_c", dpnn._SCALES["diode_temp"][0]),
            dpnn._SCALES["mA"][0],
            d.get("chip_c", TEC_SETPOINT_C),
        ]
        for st in d["states"]:
            raw = np.asarray(st["raw"], float)  # (port, pd)
            drive = volts_to_drive(np.asarray(st["volts"], float))
            for port in range(NMODE):
                Ds.append(drive)
                Ps.append(port)
                Ts.append(tel)
                Ys.append(raw[port][list(OUT_PDS)])
        print(
            f"    {path}: {len(d['states'])} states x {NMODE} ports at "
            f"+{d['dbm']:g} dBm, chip {d.get('chip_c', float('nan')):.2f} C"
        )
    return np.asarray(Ds), np.asarray(Ps, int), np.asarray(Ts), np.asarray(Ys)


def sheet_buffer(paths):
    """A bench spreadsheet of (port, DAC volts -> photodiode volts), in the same shape.

    `mrunal/28k_Data_points .xlsx` is 7000 heater combinations read through all four ports
    on 2026-08-26, which is fifty times the 25 C capture. Three things about it are traps.
    The photodiode columns are stored out of order (PD_1, PD_0, PD_3, PD_2), so they are
    selected by name. Ports are numbered from one. And it was swept to a uniform 3.0 V,
    above this tree's present ceiling on the 60-ohm channels -- every row is kept and the
    drive variable carries the volts the chip actually saw, which is why the feature path
    goes through `drive_volts_raw` and not `drive_to_volts`. Those points are real
    measurements of the chip; only the clamp has since moved.

    Nothing about the session was recorded -- no power, no chip temperature -- so the whole
    telemetry block sits at the centre of its fixed scale and normalises to zero. Note the
    date: this predates the 25 C capture and the present clamp table, so it describes the
    chip before that characterization, and a model fitted on it is only as current as the
    drift between the two."""
    import pandas as pd

    hi = np.asarray(DRIVE_SCALE, float) * DRIVE_MAX_V
    tel = [dpnn._SCALES[k][0] for k in dpnn.TELEMETRY]
    Ds, Ps, Ts, Ys = [], [], [], []
    for path in paths:
        df = pd.concat(pd.read_excel(path, sheet_name=None).values(), ignore_index=True)
        dac = [c for c in df.columns if c.startswith("DAC_")]
        chan = np.array([int(c.split("_")[1]) for c in dac], int)
        V = np.zeros((len(df), N_HEATERS))
        V[:, chan] = df[dac].to_numpy(float)
        port = df["port"].to_numpy(int)
        Ds.append(V / np.where(hi > 0, hi, 1.0) * DRIVE_MAX_V)
        Ps.append(port - port.min())  # the sheet numbers ports from one
        Ts.append(np.tile(tel, (len(df), 1)))
        Ys.append(df[[f"PD_{i}" for i in OUT_PDS]].to_numpy(float))
        hot = V > hi + 1e-9
        over = sorted({int(c) for c in np.flatnonzero(hot.any(0))})
        print(
            f"    {path}: {len(df)} rows x {NMODE} ports"
            + (
                f"; {int(hot.any(1).sum())} sit above this tree's ceiling on DAC {over} "
                f"-- kept, and carried as the volts the chip actually saw"
                if over
                else ""
            )
        )
    return (np.vstack(Ds), np.concatenate(Ps), np.vstack(Ts), np.vstack(Ys))


def varying_channels(D, tol: float = 1e-6):
    """The channels a buffer actually moved.

    The online default is `dpnn.MODEL_DACS`, every theta and phi. A stored capture may have swept more -- the 25 C
    table moved eight -- and a channel that moved the chip but has no feature column lands
    in the network's residual as unexplained variance. Reading it off the data keeps the
    two in step without either list having to be maintained."""
    D = np.atleast_2d(np.asarray(D, float))
    return np.flatnonzero((D.max(0) - D.min(0) > tol) & (np.asarray(DRIVE_SCALE) > 0))


def main(argv=None):
    ap = argparse.ArgumentParser(prog="learn.train_hw", description=__doc__.splitlines()[0])
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--n-per-round", type=int, default=100)
    ap.add_argument(
        "--ports",
        default=",".join(str(p) for p in range(NMODE)),
        help="input ports to visit each round, e.g. 0,2 (default: all four)",
    )
    ap.add_argument(
        "--channels",
        default="model",
        help="DAC channels to randomise: 'model' (every theta and phi, the "
        "network's features), 'all' (every drivable mesh channel), or a list",
    )
    ap.add_argument(
        "--programs",
        type=float,
        default=0.5,
        help="share of each round drawn around real Clements programs, where runs go",
    )
    ap.add_argument("--dbm", type=float, default=8.0)
    ap.add_argument("--settle", type=float, default=0.2)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument(
        "--from-table",
        default=None,
        help="fit stored data instead of lighting the chip: raw-transfer "
        "JSON captures and/or bench .xlsx sweeps, comma separated. No "
        "hardware, one round, and the network gets a feature per channel "
        "the data actually moved.",
    )
    ap.add_argument(
        "--out",
        default=None,
        help=f"checkpoint directory (default {CKPT}, or {TABLE_CKPT} with "
        f"--from-table -- a stored capture carries no laser telemetry, so "
        f"its buffer must not be resumed into a lit one)",
    )
    ap.add_argument("--resume", action="store_true")
    ap.add_argument(
        "--fresh",
        action="store_true",
        help="fit the physics from the calibration, not from the last saved model",
    )
    ap.add_argument(
        "--refit",
        action="store_true",
        help="refit the saved buffer in --out with the current code; no hardware",
    )
    ap.add_argument("--epochs", type=int, default=200, help="DPNN epochs per round")
    ap.add_argument(
        "--steps",
        type=int,
        default=3000,
        help="physics-fit steps per round; 0 fits the network only",
    )
    ap.add_argument(
        "--restarts",
        type=int,
        default=8,
        help="physics-fit restarts; only the first round needs many",
    )
    ap.add_argument(
        "--calib",
        default=None,
        help="bootstrap calibration to start the physics fit from "
        "(default: pic_data/calib.json if it exists)",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--mock", action="store_true", help="no hardware: the idealised chip (see the note above)"
    )
    ap.add_argument(
        "--sim", action="store_true", help="no hardware, but the bench as delivered (pic.sim)"
    )
    ap.add_argument("--tec", default="auto", help="'auto' is the real TEC on the bench")
    ap.add_argument("--laser-port", default=None)
    ap.add_argument("--pic-port", default=None)
    a = ap.parse_args(argv)

    # the detector mode's own checkpoint: an SPD-trained net never lands on PD's
    a.out = a.out or str(data_path(TABLE_CKPT if a.from_table else CKPT))
    # Rounds are written to <out>.partial and swapped into <out> only when the whole run
    # finishes: an aborted or failed run leaves the previous model in force, never half of
    # a new one.
    final, a.out = a.out, a.out + ".partial"
    if a.refit:
        # an unfinished run's data counts too, and is read before its directory is cleared
        src = a.out if os.path.exists(os.path.join(a.out, "buffer.npz")) else final
        b = np.load(os.path.join(src, "buffer.npz"))
        refit_buf = (b["D"], b["ports"], b["tel"], b["Y"])
        refit_meta = json.load(open(os.path.join(src, "meta.json")))
        print(f"refitting {len(b['D'])} points from {src}")
    shutil.rmtree(a.out, ignore_errors=True)
    if a.resume and os.path.isdir(final):
        shutil.copytree(final, a.out)
    offline = net_channels = None
    if a.from_table:
        print(f"fitting stored captures -- no hardware:")
        paths = [p for p in a.from_table.split(",") if p]
        parts = [
            (sheet_buffer if p.lower().endswith((".xlsx", ".xls")) else table_buffer)([p])
            for p in paths
        ]
        offline = tuple(np.concatenate(x) for x in zip(*parts))
        net_channels = varying_channels(offline[0])
        a.rounds, a.resume = 1, False
    elif a.refit:
        offline = refit_buf
        a.rounds, a.resume = 1, False

    ports = [int(x) for x in a.ports.split(",")]
    channels = pick_channels(a.channels)
    extra = sorted(set(channels.tolist()) - set(dpnn.MODEL_DACS.tolist()))
    os.makedirs(a.out, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    buf = {
        "D": np.zeros((0, N_HEATERS)),
        "ports": np.zeros(0, int),
        "tel": np.zeros((0, len(dpnn.TELEMETRY))),
        "Y": np.zeros((0, NUM_OUT)),
    }
    calib = Calibration.load(a.calib) if a.calib else Calibration.load_or_nominal()
    if not calib.meta:
        print(
            "no characterization found -- the physics fit will start from the nominal "
            "heater law and is unlikely to converge. Run `./do char --write` first."
        )
        calib = None
    else:
        print(f"bootstrap calibration: {calib.meta}")
    meta = {"rounds_done": 0}
    if a.refit:  # the refit replaces the last round's fit, it is not a round of its own
        meta = refit_meta
        meta["rounds_done"] -= 1
    bpath = os.path.join(a.out, "buffer.npz")
    if a.resume and os.path.exists(bpath):
        b = np.load(bpath)
        buf = {k: b[k] for k in ("D", "ports", "tel", "Y")}
        meta = json.load(open(os.path.join(a.out, "meta.json")))
        print(f"resumed: {len(buf['D'])} samples, {meta['rounds_done']} rounds done")

    if extra:
        print(
            f"channels {extra} are being driven but are not network features "
            f"(learn.dpnn.MODEL_DACS); whatever they do lands in the physics fit's "
            f"phases and in the network's residual, so read the two R^2 accordingly."
        )
    kind = "mock" if (a.mock or a.sim) else "hw"
    rig = (
        None
        if offline
        else Rig(
            laser=kind,
            board="sim" if a.sim else kind,
            switch=kind,
            tec=(
                "mock" if (a.mock or a.sim) else (auto_tec(_find_tec) if a.tec == "auto" else a.tec)
            ),
            laser_port=port_of(a.laser_port),
            pic_port=port_of(a.pic_port),
        ).open()
    )
    r0 = meta["rounds_done"]
    # Warm start from the last saved physics: after drift the chip is near where it was, so
    # one restart from there beats eight from the calibration. A checkpoint from an older
    # model shape does not load and the fit starts cold, as it would with --fresh.
    warm = None
    if not (a.fresh or a.from_table):
        try:
            warm = dpnn.load_physics(final)
        except Exception as e:
            print(f"no warm start ({type(e).__name__}); fitting the physics from the calibration")
    if warm is not None:
        print(f"warm start: physics from {final}")
    try:
        if rig is not None:
            need_outputs(rig, "DPNN training")  # every sample is a four-output vector
        for r in range(r0, r0 + a.rounds):
            t0 = time.time()
            if offline:
                D, P, T, Y = offline
            else:
                ev(
                    "dpnn",
                    "round",
                    f"[round {r + 1}] collecting {a.n_per_round} points over ports {ports} ...",
                    round=r + 1,
                    k=r - r0,
                    n=a.rounds,
                )
                D, P, T, Y = collect_round(
                    rig,
                    a.n_per_round,
                    dbm=a.dbm,
                    settle_s=a.settle,
                    repeats=a.repeats,
                    rng=rng,
                    ports=ports,
                    channels=channels,
                    calib=Calibration.load_or_nominal(),
                    programs=a.programs,
                )
            buf = {
                "D": np.vstack([buf["D"], D]),
                "ports": np.concatenate([buf["ports"], P]),
                "tel": np.vstack([buf["tel"], T]),
                "Y": np.vstack([buf["Y"], Y]),
            }
            ev(
                "dpnn",
                "points",
                f"{len(D)} points in {time.time() - t0:.0f}s; buffer {len(buf['D'])}",
                points=len(D),
                buffer=len(buf["D"]),
                secs=time.time() - t0,
            )

            # The physics fit is minutes where the network is seconds -- restarts x steps
            # of gradient descent through the twin, against one pass of backprop on a
            # couple of thousand weights. `--steps 0` skips it when the network is what
            # you came for, and leaves the bootstrap calibration untouched.
            np_phys, r2p = 0, float("nan")
            if a.steps > 0:
                X = np.eye(NMODE, dtype=complex)[buf["ports"]]
                phys, calib, err, r2p = unitary_fit.fit(
                    drive_volts_raw(buf["D"]),
                    buf["Y"],
                    X,
                    calib0=calib,
                    steps=a.steps,
                    restarts=1 if warm is not None else a.restarts,
                    seed=a.seed,
                    init=warm,
                )
                warm = phys  # the next round starts where this one ended
                np_phys = phys.n_params()  # 52 at 16 heaters; 56 when there were 18
            # the net learns what the physics gets wrong, on top of it
            ref = phys if a.steps > 0 else unitary_fit.InstrumentModel(calib)
            model, norm, r2d, dmeta = dpnn.fit(
                buf["D"],
                buf["ports"],
                buf["tel"],
                buf["Y"],
                channels=net_channels,
                epochs=a.epochs,
                seed=a.seed,
                physics=ref,
            )

            if calib is not None:
                calib.save(os.path.join(a.out, "calib.json"))
            dpnn.save_ckpt(a.out, model, norm, buf, dmeta, physics=ref)
            np.savez(bpath, **buf)
            meta.update(
                source=a.from_table or ("sim" if a.sim else kind),
                rounds_done=r + 1,
                n_samples=int(len(buf["D"])),
                r2_physics=float(r2p),
                r2_dpnn=float(np.mean(r2d)),
                n_params_physics=int(np_phys),
                n_params_dpnn=int(dmeta["n_params"]),
                dpnn_channels=dmeta["channels"],
                ports=sorted(set(buf["ports"].tolist())),
            )
            json.dump(meta, open(os.path.join(a.out, "meta.json"), "w"), indent=2)
            ev(
                "dpnn",
                "fit",
                f"physics ({np_phys}p) R2 {r2p:+.4f} | "
                f"dpnn ({dmeta['n_params']}p{' on physics' if dmeta['residual'] else ', not kept'}) "
                f"R2 {float(np.mean(r2d)):+.4f} -> {a.out}/",
                "ok",
                round=r + 1,
                k=r - r0 + 1,
                n=a.rounds,
                r2_physics=r2p,
                r2_dpnn=float(np.mean(r2d)),
                residual=dmeta["residual"],
                r2_physics_only=dmeta["r2_physics_only"],
                r2_with_net=dmeta["r2_with_net"],
            )
        completed = True
    except MissingOutputs as e:  # a refusal about the detectors in hand, not a crash
        ev("dpnn", "failed", str(e), "error")
        return 2
    except KeyboardInterrupt:
        completed = False
        ev("dpnn", "interrupted", f"interrupted -- the previous model in {final} is kept.", "warn")
    finally:
        if rig is not None:
            rig.close()

    if not completed:
        return 1
    old = final + ".old"
    shutil.rmtree(old, ignore_errors=True)
    if os.path.isdir(final):
        os.replace(final, old)
    os.replace(a.out, final)
    shutil.rmtree(old, ignore_errors=True)
    ev("dpnn", "done", f"all {a.rounds} rounds done -> {final}/", "ok", path=final)

    if len(meta.get("ports", [])) < 2:
        print(
            "\nNOTE: all data came through one input port. Some phase combinations are "
            "unobservable that way, and three quarters of the network's port one-hot "
            "never varied. Add ports before trusting either surrogate."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
