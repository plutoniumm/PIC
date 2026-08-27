"""Train the surrogates directly from the chip, one lit round at a time.

    python -m learn.train_hw --mock --rounds 4 --n-per-round 60
    python -m learn.train_hw --rounds 8 --n-per-round 150 --input-port 0

Each round lights the laser, samples random operating points, measures the outputs,
appends to a growing buffer, refits both surrogates and checkpoints. The laser is on only
while collecting and off during the fit. Ctrl-C saves and exits.

Two models are fitted on the same buffer every round, which is the point: the physics fit
(`learn.unitary_fit`, 56 parameters) and the pruned network (`learn.dpnn`, a couple of
thousand). Watching them side by side on identical data is how you tell a modelling
problem from a measurement problem -- when both stall at the same R^2, the ceiling is the
instrument, not the model.

Run `./do char --write` first. The physics fit is a refiner, not a search: from a nominal
start it plateaus, and from the Vpi that characterization recovers it converges. This picks
up `pic_data/calib.json` automatically when one exists.

The input port is a fibre position, not something the host can switch, so it is recorded
per round from `--input-port`. Data taken through one port under-determines the phases;
see the observability note in `learn.unitary_fit`.
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

from pic import Rig
from pic.acquisition import estimate_seconds, random_vectors, settled_read
from pic.config import NUM_OUT, OUT_PDS, VOLTAGE_MAX
from pic.layout import ACTIVE_DACS, N_HEATERS
from theory.calib import Calibration
from theory.clements import NMODE

from . import dpnn, unitary_fit

CKPT = "runs/hw"


def collect_round(rig, n, *, dbm, settle_s, repeats, rng, input_port):
    """One lit session of `n` random operating points. Returns (V, telemetry, Y)."""
    V = random_vectors(n, rng=rng, channels=ACTIVE_DACS)
    dur = estimate_seconds(n, settle_s, repeats)
    Vs, Ts, Ys = [], [], []
    with rig.session(duration_s=dur, power_dbm=dbm) as s:
        if not s.emitted:
            raise RuntimeError("laser reports on but no emission detected -- check coupling "
                               "and the front panel before trusting any of this data")
        for v in V:
            if s.expired():
                print("    watchdog reached; keeping the partial round.")
                break
            y = settled_read(rig.board, v, settle_s, repeats)
            bfm, dtemp, mA, chip = s.telemetry()
            Vs.append(v)
            Ts.append([dbm, bfm, dtemp, mA, chip])
            Ys.append(y[list(OUT_PDS)])
            s.keepalive()
    return np.asarray(Vs), np.asarray(Ts), np.asarray(Ys)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="learn.train_hw", description=__doc__.splitlines()[0])
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--n-per-round", type=int, default=100)
    ap.add_argument("--input-port", type=int, default=0, choices=range(NMODE),
                    help="which input fibre is lit; recorded with the data")
    ap.add_argument("--dbm", type=float, default=13.0)
    ap.add_argument("--settle", type=float, default=0.2)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--out", default=CKPT)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--epochs", type=int, default=200, help="DPNN epochs per round")
    ap.add_argument("--steps", type=int, default=3000, help="physics-fit steps per round")
    ap.add_argument("--restarts", type=int, default=8,
                    help="physics-fit restarts; only the first round needs many")
    ap.add_argument("--calib", default=None,
                    help="bootstrap calibration to start the physics fit from "
                         "(default: pic_data/calib.json if it exists)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--tec", default="mock")
    ap.add_argument("--laser-port", default=None)
    ap.add_argument("--pic-port", default=None)
    a = ap.parse_args(argv)

    os.makedirs(a.out, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    buf = {"V": np.zeros((0, N_HEATERS)), "tel": np.zeros((0, len(dpnn.TELEMETRY))),
           "Y": np.zeros((0, NUM_OUT)), "port": np.zeros(0, int)}
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
        buf = {k: b[k] for k in ("V", "tel", "Y", "port")}
        meta = json.load(open(os.path.join(a.out, "meta.json")))
        print(f"resumed: {len(buf['V'])} samples, {meta['rounds_done']} rounds done")

    kind = "mock" if a.mock else "hw"
    rig = Rig(laser=kind, board=kind, tec="mock" if a.mock else a.tec,
              laser_port=a.laser_port, pic_port=a.pic_port).open()
    r0 = meta["rounds_done"]
    try:
        for r in range(r0, r0 + a.rounds):
            print(f"\n[round {r + 1}] collecting {a.n_per_round} points, input port "
                  f"{a.input_port} ...")
            t0 = time.time()
            V, T, Y = collect_round(rig, a.n_per_round, dbm=a.dbm, settle_s=a.settle,
                                    repeats=a.repeats, rng=rng, input_port=a.input_port)
            buf = {"V": np.vstack([buf["V"], V]), "tel": np.vstack([buf["tel"], T]),
                   "Y": np.vstack([buf["Y"], Y]),
                   "port": np.concatenate([buf["port"], np.full(len(V), a.input_port)])}
            print(f"    {len(V)} points in {time.time() - t0:.0f}s; buffer {len(buf['V'])}")

            X = np.eye(NMODE, dtype=complex)[buf["port"]]
            _, calib, err, r2p = unitary_fit.fit(
                buf["V"], buf["Y"], X, calib0=calib, steps=a.steps,
                restarts=1 if r > r0 else a.restarts, seed=a.seed)
            model, norm, r2d, dmeta = dpnn.fit(buf["V"], buf["tel"], buf["Y"],
                                               epochs=a.epochs, seed=a.seed)

            calib.save(os.path.join(a.out, "calib.json"))
            dpnn.save_ckpt(a.out, model, norm, {"H": buf["V"], "tel": buf["tel"], "Y": buf["Y"]},
                           dmeta)
            np.savez(bpath, **buf)
            meta.update(rounds_done=r + 1, n_samples=int(len(buf["V"])),
                        r2_physics=float(r2p), r2_dpnn=float(np.mean(r2d)),
                        n_params_physics=56, n_params_dpnn=int(dmeta["n_params"]),
                        ports=sorted(set(buf["port"].tolist())))
            json.dump(meta, open(os.path.join(a.out, "meta.json"), "w"), indent=2)
            print(f"    physics (56p) R2 {r2p:+.4f} | "
                  f"dpnn ({dmeta['n_params']}p) R2 {float(np.mean(r2d)):+.4f} -> {a.out}/")
    except KeyboardInterrupt:
        print("\ninterrupted -- last checkpoint is saved.")
    finally:
        rig.close()

    if len(meta.get("ports", [])) < 2:
        print("\nNOTE: all data came through one input port. Some phase combinations are "
              "unobservable that way; move the input fibre and add rounds before trusting "
              "the fitted calibration for programming.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
