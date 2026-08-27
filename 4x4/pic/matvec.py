"""Run `theory.intensity_matvec` on the rig: signed y = B x from photodiodes alone.

The maths is next door and hardware-free; this is the three things it needs from a bench.

**The box.** `theory` owns the heater law but not the board, so the per-channel voltage
ceiling has to come from here: `pic.config.VOLTAGE_MAX_CH` clamps the 60 ohm heaters to
1.5 V for their current rating and holds three unconfirmed channels dark. That clamp is not
cosmetic -- it takes four of the eight steerable channels from a 0.3 pi span down to 0.08 pi
-- so a validation run against the uniform 3 V ceiling is optimistic and `bench_box` is the
one to believe.

**The probe.** One primitive: set the volts, point the switch at an input port, read the
four output photodiodes, and turn them into intensity with the calibration's own gains and
per-port coupling. Everything above it is arithmetic.

**Where the multiply happens.** With the 1x4 switch only one port is lit at a time, so the
accumulation over input rails is digital no matter what -- that is the hardware, not a
shortcut. The input weight p_j can still be optical, by retuning the laser per port
(`power="laser"`), and then the chip is doing p_j |U_kj|^2 in light. The default is
`power="digital"` because a laser retune per port costs a settle and goes through the safety
scaffolding four times per vector; say `--optical-input` when the claim matters more than
the clock. In digital mode a port read is power-independent, so the honest description of
that run is "the chip multiplies, the host weights and sums".

    python -m pic matvec --mock            # plan a random 2x2 and read it back
    python -m pic matvec --mock --scan     # which rail pair to put the block on
"""

from __future__ import annotations

import time

import numpy as np

from theory.calib import Calibration
from theory.intensity_matvec import (
    HeaterBox, plan_matvec, scan_rails, score, sinkhorn, validate,
)

from .config import DEFAULT_SETTLE_S, VOLTAGE_MAX_CH


def bench_box(calib: Calibration | None = None, vmax=None, known=None) -> HeaterBox:
    """The reachable set as the board actually is: measured Vpi and phi0, per-channel
    ceilings, and only the channels a characterization has fitted."""
    calib = calib or Calibration.load_or_nominal()
    return HeaterBox.from_calibration(
        calib, np.asarray(VOLTAGE_MAX_CH if vmax is None else vmax, float), known=known)


def mock_truth(seed: int = 0):
    """The fabricated instance `pic.interface.twin_forward` builds MockPIC from.

    A `--mock` run plans against it so the loop tests the algorithm rather than the gap
    between a saved calibration and a different simulated chip -- with `pic_data/calib.json`
    against the mock the matvec comes back a factor of four out, which is a correct answer
    to the wrong question. On the bench the calibration file IS the truth and this is
    unused."""
    from theory.twin import MeshError, Twin
    return Calibration.sample(seed=seed), Twin(MeshError.sample(seed=seed))


def rig_probe(rig, calib: Calibration | None = None, power: str = "digital",
              settle_s: float = DEFAULT_SETTLE_S, dbm: float = 8.0):
    """The bench primitive behind every pass: (volts, port, power) -> four intensities.

    Photodiode volts become intensity through the calibration's own gains and the per-port
    input coupling it measured, which is what puts every column of a probe on one scale --
    input port 3 is 6 dB down on this bench and reading it as a chip property is a mistake
    the drift work already made once.

    The thermal settle is charged only when the heater state actually changes, which in
    `shift` mode is once per plan: one program means the volts are written once and every
    later pass only moves the switch. `split` pays it on every pass."""
    calib = calib or rig.calib
    scale = np.asarray(calib.meta.get("normalisation", {}).get("input_scale",
                                                               np.ones(4)), float)
    if power not in ("digital", "laser"):
        raise ValueError(f"power={power!r}; use 'digital' or 'laser'")
    state = {"port": None, "volts": None}

    def probe(volts, port: int, p: float = 1.0):
        port, v = int(port), np.asarray(volts, float)
        if state["port"] != port:
            rig.select_input(port)
            state["port"] = port
        if state["volts"] is None or not np.array_equal(v, state["volts"]):
            rig.measure(v)
            time.sleep(settle_s)
            state["volts"] = v.copy()
        if power == "laser":
            rig.laser.set(dbm + 10 * np.log10(max(float(p), 1e-6)))
        i = calib.to_intensity(np.asarray(rig.outputs(v), float)) / max(scale[port], 1e-9)
        return i if power == "laser" else i * float(p)

    return probe


def run(rig, B, box: HeaterBox | None = None, vectors=None, rails=None, mode: str = "shift",
        power: str = "digital", dbm: float = 8.0, seed: int = 0, calib=None, twin=None,
        **fit_kw) -> dict:
    """Plan B onto the mesh, then measure y = B x for each vector. No laser session is
    opened here -- the caller owns that, because it owns the watchdog."""
    B = np.asarray(B, float)
    calib = calib or rig.calib
    box = box or bench_box(calib)
    plan = plan_matvec(B, box, twin, rails=rails, mode=mode, seed=seed, **fit_kw)
    probe = rig_probe(rig, calib, power=power, dbm=dbm)
    rng = np.random.default_rng(seed)
    xs = [rng.normal(size=B.shape[1]) for _ in range(4)] if vectors is None else list(vectors)

    rows = []
    for x in xs:
        y = plan.matvec(probe, x)
        e, s = score(y, B @ x)
        rows.append({"x": np.asarray(x, float), "y": y, "y_true": B @ x,
                     "err": e, "sign": s})
    m_err, _ = score(plan.predict().ravel(), B.ravel())
    return {"plan": plan, "rows": rows, "matrix_err": m_err,
            "vec_err": float(np.mean([r["err"] for r in rows])),
            "sign_acc": float(np.mean([r["sign"] for r in rows]))}


def digest(res) -> str:
    plan = res["plan"]
    out, inp = plan.rails
    lines = [f"rails out{out} in{inp}, {plan.mode} decomposition, "
             f"{plan.passes} photodiode reads per vector",
             f"hosted residual {plan.err:.4f}   "
             f"realised matrix vs target {res['matrix_err']:.4f}",
             f"{'x':<28} {'B x':<28} {'measured':<28} {'err':>7}"]
    for r in res["rows"]:
        lines.append(f"{np.array2string(r['x'], precision=2):<28} "
                     f"{np.array2string(r['y_true'], precision=2):<28} "
                     f"{np.array2string(r['y'], precision=2):<28} {r['err']:>7.3f}")
    lines.append(f"mean vector error {res['vec_err']:.4f}, "
                 f"sign accuracy {res['sign_acc']:.0%}")
    return "\n".join(lines)


def _selftest(seed: int = 0):
    """The theory validation run against the BENCH box, which is the number that counts.

    Two levels of read noise, both measured: `pic.sim` puts the PD/TIA read noise at
    0.6-4.4 mV against a 173-261 mV full scale."""
    box = bench_box()
    clean = validate(box, k=2, trials=6, restarts=12, steps=300, seed=seed)
    noisy = validate(box, plan=clean["plan"], noise=0.012, trials=12, seed=seed + 1)
    free = int(box.trainable.sum())
    assert clean["vec_err"] < 0.05, clean["vec_err"]
    assert clean["sign_acc"] == 1.0, clean["sign_acc"]
    return {"free": free, "span_max": float(box.span_pi[box.trainable].max(initial=0.0)),
            "fit_err": clean["fit_err"], "vec_err": clean["vec_err"],
            "sign_acc": clean["sign_acc"], "noisy_err": noisy["vec_err"],
            "noisy_sign": noisy["sign_acc"], "passes": clean["passes"],
            "brightness": clean["plan"].terms[0].prog.scale, "rails": clean["rails"]}


def main(rig, a) -> int:
    """`python -m pic matvec`. `rig` is open and inside a laser session, or None for
    `--scan`, which is a fit and needs no light."""
    calib, twin = (rig.calib if rig is not None else Calibration.load_or_nominal()), None
    if getattr(a, "mock", False) and not getattr(a, "sim", False):
        calib, twin = mock_truth()
    box = bench_box(calib)
    rng = np.random.default_rng(a.seed)
    B = np.atleast_2d(np.loadtxt(a.target) if a.target else rng.normal(size=(a.k, a.k)))

    print(f"heater box: {int(box.trainable.sum())} steerable channels, widest span "
          f"{box.span_pi[box.trainable].max(initial=0.0):.2f} pi "
          f"({'measured' if calib.meta else 'NOMINAL'} calibration)")
    if a.scan:
        A = sinkhorn(np.abs(B) + 0.05)[0]
        print(f"\nrail scan for a {B.shape[0]}x{B.shape[0]} block "
              f"(feasible first, brightest among those):")
        print(f"  {'out':<9}{'in':<9}{'residual':>9}{'brightness':>12}")
        for r in scan_rails(A, box, restarts=12, steps=300)[:6]:
            print(f"  {str(r['out']):<9}{str(r['in']):<9}{r['err']:>9.4f}{r['scale']:>12.3f}")
        return 0

    print(f"\ntarget B:\n{np.round(B, 3)}")
    res = run(rig, B, box, rails=None if a.rails is None else _rails(a.rails), mode=a.mode,
              power="laser" if a.optical_input else "digital", dbm=a.dbm, seed=a.seed,
              calib=calib, twin=twin, restarts=a.restarts, steps=a.steps)
    print()
    print(digest(res))
    if res["plan"].err > 0.05:
        print("\nthe mesh cannot host this target: the residual above is in the MATRIX, not "
              "the readout. Try --scan for better rails, or a smaller block.")
    return 0


def _rails(spec: str):
    """`1,2:0,3` -> ((1, 2), (0, 3))."""
    out, inp = spec.split(":")
    return (tuple(int(x) for x in out.split(",")), tuple(int(x) for x in inp.split(",")))


if __name__ == "__main__":
    r = _selftest()
    print(f"bench box: {r['free']} steerable channels, widest span {r['span_max']:.2f} pi")
    print(f"2x2 on rails out{r['rails'][0]} in{r['rails'][1]}, {r['passes']} reads/vector")
    print(f"  hosted residual {r['fit_err']:.4f}, block brightness {r['brightness']:.3f}")
    print(f"  vector error / sign  {r['vec_err']:.4f} / {r['sign_acc']:.0%}  (noiseless)")
    print(f"  vector error / sign  {r['noisy_err']:.4f} / {r['noisy_sign']:.0%}  "
          f"(1.2% read noise, single read)")
