"""Hold the laser diode at a KNOWN temperature, by servoing the setpoint from the outside.

The PDMv5's own TEC loop works: at a fixed setpoint it holds the diode to better than a
millikelvin and is completely indifferent to drive current -- measured 39.907 / 39.906 /
39.906 / 39.907 C at 73 / 89 / 109 / 134 mA. What it does NOT do is hold the temperature the
setpoint asks for, because TECSlope is 0 and the cal table is empty on this unit: asked for
25.0 it has settled at 27.40, at 39.91 and at 47.65 on three separate enables in one session.

That is the worst possible combination for this rig. Wavelength follows diode temperature and
every MZI phase follows wavelength, so a mesh characterized at one hold point has a different
reachable set at another -- stable within a session, irreproducible between them. A transfer
table captured yesterday plans badly today for no reason the table can see.

So close a slow outer loop in software: read `diode_temperature`, nudge the `temperature`
setting, repeat. The inner loop's gain is unknown and its sign is not guaranteed, so this
measures both rather than assuming them, and refuses rather than running away.
"""

from __future__ import annotations

import time

# 35 C, and the absolute value is not the point. The setpoint runs BACKWARDS on this unit --
# measured gain -0.83 to -0.94 C of diode per C of setpoint -- and the diode bottoms out near
# 34.5 C at setpoint 65, so 25 and 30 are simply unreachable. 35 sits inside the range with
# margin instead of pinned against a rail, where the hold is 0.4 C peak-to-peak.
#
# What matters is holding the SAME temperature every session, not a particular one. Wavelength
# follows diode temperature and the mesh's reachable set follows wavelength, so two tables are
# only comparable if both were captured at the same hold point. Left to itself this unit lands
# wherever it likes: 27.40, 39.91 and 47.65 C on three separate enables in one session, which
# is why a table captured an hour ago hosted 2.4x worse than one captured before it.
TARGET_C = 35.0
TOL_C = 0.10             # hold quality is 0.4 C peak-to-peak near the rail, better inside it
SETTLE_S = 20.0          # the inner loop is slow; anything faster measures a transient
MAX_STEPS = 14
SP_MIN, SP_MAX = 5.0, 65.0
TEC_CURRENT_MAX = 500.0  # refuse rather than drive the cooler into whatever its limit is


class DiodeTempError(RuntimeError):
    pass


def read_c(dev) -> float:
    return float(dev.measure("diode_temperature"))


def _settle(dev, s: float = SETTLE_S) -> float:
    time.sleep(s)
    return sum(read_c(dev) for _ in range(6)) / 6.0


def measure_gain(dev, step: float = 3.0) -> tuple[float, float]:
    """dT_diode / dSetpoint, measured. Returns (gain, current setpoint).

    Two points and a difference. The sign matters and is not documented for an uncalibrated
    unit -- a positive setpoint step could plausibly drive either way -- so it is measured
    before anything is trusted to it."""
    sp0 = float(dev.read_setting("temperature"))
    t0 = _settle(dev)
    sp1 = min(SP_MAX, max(SP_MIN, sp0 + step))
    dev.write_setting("temperature", sp1, verify=False)
    t1 = _settle(dev)
    dev.write_setting("temperature", sp0, verify=False)
    if abs(sp1 - sp0) < 1e-6:
        raise DiodeTempError("setpoint could not be stepped inside its limits")
    return (t1 - t0) / (sp1 - sp0), sp0


def hold(dev, target_c: float = TARGET_C, tol_c: float = TOL_C,
         max_steps: int = MAX_STEPS, verbose: bool = True) -> float:
    """Drive the diode to `target_c` and leave it there. Returns the temperature reached.

    Secant on the measured gain rather than a fixed step: the inner loop's transfer is
    unknown, so a fixed step either crawls or overshoots depending on a number nobody has."""
    g, sp = measure_gain(dev)
    if verbose:
        print(f"  diode/setpoint gain {g:+.3f} C/C at setpoint {sp:.2f}")
    if abs(g) < 0.05:
        raise DiodeTempError(
            f"setpoint moves the diode by only {g:+.3f} C per C -- this unit's TEC cannot be "
            f"steered from the setpoint, so the hold point cannot be made reproducible")
    for i in range(max_steps):
        t = _settle(dev)
        ic = abs(float(dev.measure("tec_current")))
        if ic > TEC_CURRENT_MAX:
            dev.write_setting("temperature", 25.0, verify=False)
            raise DiodeTempError(f"tec_current {ic:.0f} over {TEC_CURRENT_MAX:.0f}; "
                                 f"setpoint returned to 25 and the servo stopped")
        if verbose:
            print(f"  step {i}: setpoint {sp:.2f} -> diode {t:.3f} C")
        if abs(t - target_c) <= tol_c:
            return t
        sp = float(min(SP_MAX, max(SP_MIN, sp + (target_c - t) / g)))
        dev.write_setting("temperature", sp, verify=False)
    raise DiodeTempError(f"did not reach {target_c} C in {max_steps} steps; last {t:.3f} C")


class PID:
    """Continuous hold on the diode temperature, running alongside a measurement.

    A one-shot servo is the wrong shape for this loop. Two things defeat it, both measured:
    the gain is strongly nonlinear -- -1.36 C/C around setpoint 55, and effectively -0.03 near
    the floor at 34.2 C where the setpoint has no authority left -- and the diode DRIFTS at a
    fixed setpoint, 0.087 C over four minutes. A single correction leaves that drift running
    for the rest of a capture, which is exactly the interval over which a table has to describe
    one chip.

    So: integrate. The integrator absorbs the drift and the unknown local gain, and clamping it
    to the setpoint range stops it winding up against the floor. Target inside the authority
    band -- 38 C sits mid-band, where 4 C of setpoint moved 4.5 C of diode; 35 and 30 are at or
    under the floor and no controller reaches them.

    Derivative is deliberately absent. The plant is a thermal mass behind a slow inner loop and
    the sensor is quantised; a D term on this would amplify read noise into setpoint dither.
    """

    def __init__(self, dev, target_c: float = 38.0, kp: float = -0.6, ki: float = -0.08,
                 period_s: float = 15.0):
        self.dev, self.target, self.kp, self.ki = dev, float(target_c), kp, ki
        self.period = float(period_s)
        self.i = float(dev.read_setting("temperature"))   # integrator IS the setpoint
        self._last = 0.0

    def step(self) -> tuple[float, float]:
        """One update. Returns (diode_c, setpoint). Call every `period_s`."""
        t = read_c(self.dev)
        e = self.target - t
        self.i = min(SP_MAX, max(SP_MIN, self.i + self.ki * e * self.period))
        sp = min(SP_MAX, max(SP_MIN, self.i + self.kp * e))
        self.dev.write_setting("temperature", sp, verify=False)
        self._last = t
        return t, sp

    def settle(self, tol_c: float = 0.3, timeout_s: float = 420.0, verbose: bool = True):
        """Run until the diode sits within `tol_c` of target for three consecutive steps."""
        t0, ok = time.time(), 0
        while time.time() - t0 < timeout_s:
            t, sp = self.step()
            ok = ok + 1 if abs(t - self.target) <= tol_c else 0
            if verbose:
                print(f"    {time.time()-t0:5.0f}s  diode {t:7.3f}  setpoint {sp:6.2f}"
                      f"  {'ok' if ok else ''}", flush=True)
            if ok >= 3:
                return t
            time.sleep(self.period)
        raise DiodeTempError(f"PID did not hold {self.target} C within {tol_c} in "
                             f"{timeout_s:.0f} s; last {t:.3f} C at setpoint {sp:.2f}")


def _selftest():
    class Fake:
        """An inner loop with an unknown offset and gain, exactly like the real one."""
        def __init__(self, gain=0.8, offset=11.4):
            self.sp, self.g, self.o = 25.0, gain, offset
        def read_setting(self, k): return self.sp
        def write_setting(self, k, v, **kw): self.sp = float(v)
        def measure(self, k): return self.g * self.sp + self.o

    import builtins
    real_sleep, time.sleep = time.sleep, lambda s: None
    try:
        for gain, off in ((0.8, 11.4), (1.0, 2.4), (0.5, 22.0)):
            d = Fake(gain, off)
            got = hold(d, 25.0, verbose=False)
            assert abs(got - 25.0) <= TOL_C, (gain, off, got)
        print(f"servo reaches 25.00 C within {TOL_C} C for three unknown inner loops")
        d = Fake(0.0, 39.9)                      # a unit that cannot be steered
        try:
            hold(d, 25.0, verbose=False)
        except DiodeTempError as e:
            assert "cannot be steered" in str(e)
            print("refuses a unit whose setpoint does not move the diode")
        else:
            raise AssertionError("must refuse when the gain is zero")
        # the PID has to reach a target the one-shot servo cannot, on a plant that drifts
        class Drifting(Fake):
            def __init__(self, gain, offset, drift):
                super().__init__(gain, offset); self.d, self.n = drift, 0
            def measure(self, k):
                self.n += 1
                return self.g * self.sp + self.o + self.d * self.n
        for gain, off, drift in ((-0.9, 62.0, 0.0), (-1.3, 78.0, 0.004), (-0.6, 52.0, -0.003)):
            d = Drifting(gain, off, drift)
            p = PID(d, target_c=38.0, period_s=1.0)
            got = p.settle(tol_c=0.3, timeout_s=1e6, verbose=False)
            assert abs(got - 38.0) <= 0.3, (gain, off, drift, got)
        print("PID reaches 38.00 C on three plants, including two that drift")
        print("diode temperature servo selftest OK")
    finally:
        time.sleep = real_sleep


if __name__ == "__main__":
    _selftest()
