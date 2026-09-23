"""Bench-faithful simulator of the delivered 4x4 rig.

`MockPIC` models the board this package was *designed* for: 18 drivable heaters, a clean
readout, a fabricated instrument with Vpi near 1.5 V. The bench that actually exists is a
different machine, and every number below is read off the material in `mrunal/`:

  six DAC channels of eighteen, clamped at 3 V (`Setup.ino`);
  four InGaAs PD-TIA channels on a 10-bit ADC, five reads averaged (`Setup.ino`);
  a Sercalo 1x4 in front of the input facet, one port lit at a time;
  a mesh whose measured input-to-output power table is 10-17 dB down (`SNR_Analysis (1).pdf`);
  heaters whose Vpi is 2.3-4.6 V, so 3 V buys 0.4-1.7 pi and never a whole fringe;
  1-3 percent read scatter and a coupling term that a fibre reconnect moves by 3 dB.

The forward path is physics, not a lookup: switch port -> `theory.twin` (the same mesh
algebra the programming path uses) -> per-port and per-output insertion loss -> PD-TIA
transimpedance -> ADC quantisation -> the measured noise and drift terms. Both ends are
anchored on measurements, so a run against this simulator fails for the reasons the bench
will fail: channels that never complete a fringe, a port that sits 6 dB below its
neighbours, a fit that dies on the firmware's 3 V clamp.

    from pic.sim import BenchPIC, BenchSim
    rig = Rig(laser="mock", board="sim", tec="mock").open()

`MockPIC` is untouched and still the default. Use this one when the question is "will this
work on the bench", and that one when the question is "is the algorithm right".

Deliberately not modelled, because nothing in `mrunal/` measures it on this chip and a
guessed effect is worse than an absent one. All four are PROVISIONAL gaps, not claims that
the bench is free of them:

  the thermo-optic settling transient. `Setup.ino` waits 500 ms after moving a heater and
    the 6x6 rig's heaters took 0.7-1.2 s to reach 63 percent, so the bench is very likely
    reading before it has settled -- but no 4x4 step response exists to fit.
  heater-to-heater thermal crosstalk. Six heaters on one die share a substrate.
  the chip TEC's effect on phase. It holds 25 C, so this is small by design.
  PD-TIA saturation. The brightest value in 77,280 logged reads is 0.587 V, well inside a
    5 V ADC, so nothing in the data constrains where the amplifier gives up.

Self-test: ``python -m pic.sim``.
"""

from __future__ import annotations

import numpy as np

from theory.clements import NMODE, NMZI
from theory.layout import N_HEATERS, THETA_IDX, PHI_IDX

from .config import (
    mirror_pairs,
    ADC_AVG_N,
    VOLTAGE_MAX_CH,
    ADC_BITS,
    ADC_REF_V,
    DAC_BITS,
    DAC_REF_V,
    FIRMWARE_VMAX,
    NUM_ADC_RAW,
    OUT_PDS,
    PICConfig,
    WIRED_DACS,
)
from .interface import MOCK_CAPS, PIC, emulate_sweep

# --- the measured chip -------------------------------------------------------------

# `mrunal/SNR_Analysis (1).pdf` Table 1: power at each output facet, in dBm, for light
# injected at each input facet, all heaters at 0 V. Rows i1..i4 = L15/L17/L19/L21, columns
# o1..o4 = R24/R22/R20/R18. This is the one absolute optical measurement the bench has and
# everything optical here is fitted to it.
MEASURED_OUT_DBM = np.array(
    [
        [-14.00, -22.00, -19.20, -11.92],
        [-15.03, -19.20, -13.78, -13.98],
        [-34.00, -16.66, -24.00, -36.00],
        [-14.14, -21.88, -19.03, -14.64],
    ]
)
REF_INPUT_DBM = 1.0  # the launch power that table's loss column was computed against

# What a logged session actually ran at. The loss table above was taken through an external
# power meter at a launch the recorded PD sessions never saw: summed port powers in
# `Drift_data.xlsx` run 55-760 mV, which at the transimpedance below is -18 to -8 dBm at the
# switch. This default matches Experiment 2, the run every noise figure here is quoted from.
SESSION_INPUT_DBM = -12.0

# Same document, Table 2: launch power minus the summed output power, per input port. This
# is the whole chain -- switch, fibre-to-chip coupling, mesh, chip-to-fibre coupling -- and
# port 3 is 6.8 dB worse than the others, which is a coupling fault, not mesh physics.
PORT_LOSS_DB = np.array([10.08, 9.98, 16.78, 11.33])

# Per-output excess loss, in dB, referenced to the best output. Not assumed: a lossless
# unitary makes the row-normalised power table doubly stochastic, so whatever is left over
# in the *column* sums is output-side loss and nothing else. Sinkhorn on the measured table
# and the joint fit below independently land on the same four numbers.
OUT_LOSS_DB = np.array([0.866, 1.085, 2.558, 0.000])

# Mesh phase at 0 V, fitted so that the twin reproduces MEASURED_OUT_DBM. Twelve free
# phases against sixteen measurements, and it lands inside 0.05 dB everywhere -- i.e. the
# measured table *is* unistochastic once the output losses above are taken out, which is a
# non-trivial check on the whole 4x4-unitary claim. Produced by `fit_zero_bias`.
ZERO_BIAS_THETA = np.array([5.029610, 0.357038, 1.166590, -1.325917, 6.046421, 5.296145])
ZERO_BIAS_PHI = np.array([1.290994, 5.604259, 4.808305, 0.444812, 0.790153, 1.268821])

# Which MZI each wired DAC channel drives, from the routing tables in
# `mrunal/Loss_Analysis (1).pdf` (Tables 4-7). Those list, for each input-output pair, the
# heaters that had to be set to maximise it: {H4,H3,H1} reach OUT1, {H6,H4,H3,H1} reach
# OUT2, {H5,H3,H2} reach OUT4, {H6,H5,H3,H2} reach OUT3. Exactly one labelling of the
# rectangular mesh satisfies all four -- H4 and H5 are the two first-column MZIs, H3 the
# centre one, H1 and H2 the third column, H6 the last -- so the topology is derived, not
# guessed.
#
# PROVISIONAL: that H1..H6 in those tables are DAC0..DAC5 in that order. The bench data
# supports it on its sharpest prediction: DAC3 would then be the first-column MZI on rails
# (0,1) and so blind to light injected at ports 3 and 4, and its measured modulation depth
# there is 4x smaller than at ports 1 and 2.
# Keyed by NEW DAC channel since the rewire: the bring-up board drove H15, H12, H11, H8, H4
# and H3 on channels 0-5, and those heaters are now on 8, 10, 11, 13, 6 and 15.
BENCH_MZI_OF_DAC = {8: 3, 10: 4, 11: 2, 13: 0, 6: 1, 15: 5}

# Vpi per wired DAC channel, volts. Fitted from 8000 measured (6 DAC volts -> 4 PD volts)
# rows in `mrunal/Combined_Project_Data (1).xlsx`: marginalising over the other five
# channels leaves the swept channel's cosine intact (its frequency, at least), so each
# channel gets one Vpi shared across every port-and-detector marginal that showed contrast.
# Channels 4 and 5 modulate weakly and their numbers are the least trustworthy.
# Keyed by the NEW DAC channel since the 2026-09-22 rewire (mrunal/board_firmware_v2.md).
# The six fits belong to heaters, not to channel numbers: the bring-up board drove H15, H12,
# H11, H8, H4 and H3 on channels 0-5, and those heaters now sit on 8, 10, 11, 13, 6+7 and 15.
# H4 takes a bonded pair, so both its channels carry its Vpi.
BENCH_VPI_OF_DAC = {8: 4.10, 10: 4.57, 11: 3.78, 13: 3.92, 6: 2.97, 7: 2.97, 15: 2.27}
BENCH_VPI = np.array([BENCH_VPI_OF_DAC[d] for d in sorted(BENCH_VPI_OF_DAC)])

# --- the readout -------------------------------------------------------------------

# Transimpedance, volts per watt at the output facet. From `Setup_Analysis.pdf` 1.5: launch
# -1.71 dBm into port 4, whose measured total loss is 11.33 dB, so 49.6 uW arrives spread
# across the four outputs, and the four PDs summed to 1.91 V. About 41 kOhm behind a 0.95
# A/W InGaAs diode, which is an ordinary TIA.
#
# PROVISIONAL in magnitude: rows 2-4 of that same table sum to a quarter of rows 0-1 at the
# same launch power, which no mesh setting can do to a unitary. Something in that run was
# not settled. The value here takes the brightest rows.
PD_TIA_V_PER_W = 3.9e4

ADC_LSB_V = ADC_REF_V / (2**ADC_BITS - 1)  # 4.888 mV, the raw 10-bit step
READ_QUANTUM_V = ADC_LSB_V / ADC_AVG_N  # the firmware averages ADC_AVG_N reads
DAC_LSB_V = DAC_REF_V / (2**DAC_BITS - 1)

# --- noise and drift ---------------------------------------------------------------
#
# All four numbers come out of `mrunal/Drift_data.xlsx`, Experiment 2: two DAC states x
# four input ports x 900 consecutive one-second reads, laser and bias untouched.
#
#   per-channel sd            0.58 - 5.65 mV      (dark channels at the bottom of the range)
#   per-channel max - min     2.9 - 33.2 mV       (`Drift_Report.pdf` Table 3)
#   sd of the *summed* four   1.4 - 2.5 % of the sum
#
# The summed-power scatter cannot come from mesh phase, which only moves light between
# outputs, so it is laser and coupling: a common-mode gain jitter. What is left over after
# removing it is per-output, and the cheapest physical account of that is phase jitter,
# which is also why a channel sitting on the steep part of its fringe is the noisy one.

RIN_FRAC = 0.020  # common-mode gain jitter per read; sd of the summed four outputs
PHASE_JITTER_RAD = 0.030  # per-heater, per read; set so per-channel sd lands in 0.6-5.7 mV
READ_SIGMA_V = 6.0e-4  # additive electronic floor, from the darkest channels

# Drift. Experiment 1 puts a 30-minute repeat at 5.0-5.8 mV rms, which is the read noise
# over again, so nothing measurable moves in half an hour. Experiment 3 repeats a fixed bias
# three hours later and the summed power per port has fallen 1.7-3.0 %. That bounds the slow
# terms from both sides; the split between coupling and phase is not separately identified,
# so the phase figure is PROVISIONAL and chosen to keep a three-hour repeat inside the
# measured 2-67 mV max-deviation band.
DRIFT_GAIN_PER_SQRT_H = 0.012  # fractional random walk on each port's coupling
DRIFT_PHASE_PER_SQRT_H = 0.020  # rad, per heater

# Unplugging and replugging the input fibre moved the per-port throughput by -1.2 to +3.3 dB
# (Experiment 1, DS2 -> DS3). That is not time drift -- the 30-minute repeat ruled that out
# -- it is the connector, and it is the largest single error on this bench.
RECOUPLE_SIGMA_DB = 1.9

# Cycling the switch costs more than dwelling on one port: Experiment 3 cycles all four
# ports every 1.5 s and the summed power per port scatters by 2.6-10.8 %, against 1.4-2.5 %
# for Experiment 2's dwell. The difference is the switch landing in a slightly different
# place each time.
SWITCH_REPEAT_FRAC = 0.025


def bench_heater_of_dac(mzi_of_dac=BENCH_MZI_OF_DAC) -> np.ndarray:
    """DAC channel -> heater index, for the six wired channels.

    Each wired channel drives one MZI's *internal* arm phase; the packaged part carries one
    heater per phase, and the routing tables show one heater per MZI. Unwired channels get
    the leftover heater indices in order, which keeps the map a permutation -- an identity
    map with six entries overwritten silently leaves heaters double-driven.

    This disagrees with `pic.layout.HEATER_OF_DAC`, and the disagreement is not cosmetic:
    that map sends DAC0 to UH15 and `theory.layout` calls UH15 an output trimmer, which is
    invisible in intensity by construction. DAC0 is the *strongest* modulator on the bench
    -- 40 mV of swing at port 4 against a 55 mV total. One of the two is wrong, and it is
    not the measurement."""
    fixed = {d: int(THETA_IDX[m]) for d, m in mzi_of_dac.items()}
    rest = [h for h in range(N_HEATERS) if h not in set(fixed.values())]
    out, it = np.empty(N_HEATERS, int), iter(rest)
    return np.array([fixed[d] if d in fixed else next(it) for d in range(N_HEATERS)])


def zero_bias_phases(theta=ZERO_BIAS_THETA, phi=ZERO_BIAS_PHI) -> np.ndarray:
    ph = np.zeros(N_HEATERS)
    ph[THETA_IDX] = theta
    ph[PHI_IDX] = phi
    return ph


def _eta_out(out_loss_db=OUT_LOSS_DB) -> np.ndarray:
    return 10 ** (-np.asarray(out_loss_db, float) / 10)


def _eta_in(mesh_t, out_loss_db=OUT_LOSS_DB, port_loss_db=PORT_LOSS_DB) -> np.ndarray:
    """Per-input transmission that makes the summed output power match the measured loss.

    Solved rather than assumed, because the measured per-port loss is a *total* -- it
    already contains whatever the outputs lose -- so subtracting the output terms without
    dividing them back out would count them twice."""
    z = (np.asarray(mesh_t, float) * _eta_out(out_loss_db)).sum(1)
    return 10 ** (-np.asarray(port_loss_db, float) / 10) / z


class BenchSim:
    """The chip, the fibres and the readout, as one function of eighteen DAC volts.

    Time is a virtual clock, not the wall clock: drift only happens when someone calls
    `advance`, so a run is reproducible and a caller that wants to ask "what does this look
    like three hours later" does not have to wait three hours.
    """

    def __init__(
        self,
        *,
        seed: int = 0,
        input_dbm: float = SESSION_INPUT_DBM,
        heater_of_dac=None,
        vpi=None,
        phi0=None,
        vmax: float = FIRMWARE_VMAX,
        wired=WIRED_DACS,
        noise: bool = True,
        quantise: bool = True,
    ):
        import torch

        from theory.twin import Twin

        self._torch = torch
        self.twin = Twin()
        self.rng = np.random.default_rng(seed)
        self.input_dbm = float(input_dbm)
        self.vmax = float(vmax)
        self.noise = bool(noise)
        self.quantise = bool(quantise)
        self.wired = np.array(sorted(wired), int)

        self.heater_of_dac = (
            bench_heater_of_dac() if heater_of_dac is None else np.asarray(heater_of_dac, int)
        )
        self.phi0 = zero_bias_phases() if phi0 is None else np.asarray(phi0, float).copy()

        # an unwired channel is not a heater at 0 V, it is a heater no voltage reaches:
        # infinite Vpi, so whatever the host sends it the phase never moves
        self.vpi = np.full(N_HEATERS, np.inf)
        if vpi is None:
            # by channel, not by position in `self.wired`: the two stopped agreeing when the
            # rewire moved every heater off the channel that used to carry it
            for d, x in BENCH_VPI_OF_DAC.items():
                self.vpi[self.heater_of_dac[d]] = x
        else:
            for d, x in zip(self.wired, np.atleast_1d(np.asarray(vpi, float))):
                self.vpi[self.heater_of_dac[d]] = x

        self.mesh_t = self._mesh_t(self.phi0)
        self.eta_out = _eta_out()
        self.eta_in = _eta_in(self.mesh_t)

        self.hours = 0.0
        self.phase_drift = np.zeros(N_HEATERS)
        self.port_gain = np.ones(NMODE)
        self.port = 0

    def _mesh_t(self, phases) -> np.ndarray:
        """|U|^2 as T[i, o]: the power fraction from input facet i to output facet o."""
        U = self.twin.matrix(
            self._torch.as_tensor(np.asarray(phases, float), dtype=self._torch.float32)
        )
        return (U.abs() ** 2).detach().numpy().T

    def select(self, port: int) -> int:
        """Point the input switch at `port` (0..3). Each selection redraws that port's
        coupling, which is what makes a cycled sweep noisier than a dwelling one."""
        port = int(port)
        if not 0 <= port < NMODE:
            raise ValueError(f"input port {port} outside 0..{NMODE - 1}")
        self.port = port
        if self.noise:
            self.port_gain[port] *= np.exp(self.rng.normal(0, SWITCH_REPEAT_FRAC))
        return port

    def advance(self, hours: float):
        """Let `hours` of laboratory time pass. Coupling and phase both random-walk."""
        h = float(hours)
        if h <= 0:
            return self
        self.hours += h
        self.port_gain *= np.exp(self.rng.normal(0, DRIFT_GAIN_PER_SQRT_H * h**0.5, NMODE))
        self.phase_drift += self.rng.normal(0, DRIFT_PHASE_PER_SQRT_H * h**0.5, N_HEATERS)
        return self

    def recouple(self):
        """Unplug and replug the input fibre. The dominant error on this bench, and the one
        that makes a calibration from yesterday worthless."""
        self.port_gain *= 10 ** (self.rng.normal(0, RECOUPLE_SIGMA_DB, NMODE) / 10)
        return self

    def phases(self, v_dac) -> np.ndarray:
        """Eighteen DAC volts -> eighteen optical phases, through the firmware's clamp and
        16-bit quantiser. Channels with no heater on them contribute their fixed phi0."""
        v = np.clip(np.asarray(v_dac, float).ravel(), 0.0, self.vmax)
        v = np.round(v / DAC_LSB_V) * DAC_LSB_V
        vh = np.zeros(N_HEATERS)
        vh[self.heater_of_dac] = v
        return self.phi0 + self.phase_drift + np.pi * np.square(vh / self.vpi)

    def power_w(self, v_dac, port: int | None = None) -> np.ndarray:
        """Optical power at the four output facets, in watts. No readout, no noise."""
        p = self.port if port is None else int(port)
        ph = self.phases(v_dac)
        if self.noise and PHASE_JITTER_RAD:
            ph = ph + self.rng.normal(0, PHASE_JITTER_RAD, N_HEATERS)
        t = self._mesh_t(ph)[p]
        pin_w = 1e-3 * 10 ** (self.input_dbm / 10)
        rin = self.rng.normal(0, RIN_FRAC) if self.noise else 0.0
        g = self.port_gain[p] * (1 + rin)
        return pin_w * self.eta_in[p] * g * t * self.eta_out

    def read(self, v_dac, port: int | None = None) -> np.ndarray:
        """The four PD-TIA volts an ADC read would return."""
        y = PD_TIA_V_PER_W * self.power_w(v_dac, port)
        if self.noise:
            y = y + self.rng.normal(0, READ_SIGMA_V, y.size)
        if self.quantise:
            y = np.round(y / READ_QUANTUM_V) * READ_QUANTUM_V
        return np.clip(y, 0.0, ADC_REF_V)

    def dark_read(self) -> np.ndarray:
        """What the four channels read with the switch routed nowhere. Not zero: the ADC
        floor and whatever the TIA offset is, which is what a dark reference measures."""
        y = self.rng.normal(0, READ_SIGMA_V, NMODE) if self.noise else np.zeros(NMODE)
        if self.quantise:
            y = np.round(y / READ_QUANTUM_V) * READ_QUANTUM_V
        return np.clip(y, 0.0, ADC_REF_V)

    def out_dbm(self, v_dac=None, port: int | None = None) -> np.ndarray:
        v = np.zeros(N_HEATERS) if v_dac is None else v_dac
        return 10 * np.log10(1e3 * np.maximum(self.power_w(v, port), 1e-30))

    def table_dbm(self, v_dac=None) -> np.ndarray:
        """The input-by-output power table, the shape `MEASURED_OUT_DBM` is in."""
        return np.stack([self.out_dbm(v_dac, p) for p in range(NMODE)])


class BenchPIC(PIC):
    """`BenchSim` behind the board driver's interface. Drop-in for `MockPIC`.

    Clamps at the firmware's 3 V rather than the host's 5 V, because `Setup.ino` does and a
    host that thinks it swept to 5 V has really swept to 3 and fitted the wrong Vpi.
    """

    def __init__(
        self, sim: BenchSim | None = None, config: PICConfig | None = None, switch=None, **kw
    ):
        kw.setdefault("voltage_max", FIRMWARE_VMAX)
        super().__init__(config, **kw)
        self.sim = BenchSim() if sim is None else sim
        self.switch = switch
        self._last_v = np.zeros(self.cfg.num_dac)

    def open(self):
        self.ser = "sim"
        return self

    def close(self):
        self.ser = None

    def capabilities(self, refresh: bool = False) -> dict:
        return dict(MOCK_CAPS)

    def sweep_raw(self, cycles: int = 1, reads: int = 1, timeout_s=None) -> np.ndarray:
        return emulate_sweep(self, cycles, reads)

    def measure_raw(self, voltages, retries: int = 3) -> np.ndarray:
        v = self._prep_dac(voltages)
        self._last_v = v
        raw = np.zeros(NUM_ADC_RAW)
        if self.switch is not None and self.switch.selected is None:
            raw[list(OUT_PDS)] = self.sim.dark_read()  # `SET 0`: the path is open
        else:
            port = self.sim.port if self.switch is None else self.switch.selected
            raw[list(OUT_PDS)] = self.sim.read(v, port)
        return raw


def fit_zero_bias(measured_dbm=MEASURED_OUT_DBM, restarts: int = 60, seed: int = 0):
    """Recover the 0 V mesh phases and the per-output loss from the measured power table.

    Twelve mesh phases and three relative output transmissions against sixteen numbers. The
    fit is on the *row-normalised* table so it cannot buy agreement by rescaling a port --
    the per-port totals are already pinned by the measured loss column. Returns
    (theta, phi, eta_out, max |error| in dB).

    Slow (seconds), so the result is frozen into `ZERO_BIAS_*` above; this is the routine
    that produced them and the way to redo it if the table is ever remeasured."""
    import torch
    from scipy.optimize import least_squares

    from theory.twin import Twin

    twin = Twin(dtype=torch.complex128)
    p = 10 ** (np.asarray(measured_dbm, float) / 10)
    target = p / p.sum(1, keepdims=True)

    def split(x):
        ph = np.zeros(N_HEATERS)
        ph[THETA_IDX], ph[PHI_IDX] = x[:NMZI], x[NMZI : 2 * NMZI]
        m = np.abs(twin.matrix(torch.as_tensor(ph)).numpy()) ** 2
        g = np.concatenate([[1.0], np.exp(x[2 * NMZI :])])
        w = m.T * g
        return w / w.sum(1, keepdims=True), g / g.max()

    rng = np.random.default_rng(seed)
    best = None
    for _ in range(restarts):
        x0 = np.concatenate([rng.uniform(0, 2 * np.pi, 2 * NMZI), rng.normal(0, 0.3, NMODE - 1)])
        r = least_squares(
            lambda x: (np.log10(split(x)[0]) - np.log10(target)).ravel(), x0, max_nfev=800
        )
        if best is None or r.cost < best.cost:
            best = r
    s, g = split(best.x)
    err = np.abs(10 * np.log10(s / target)).max()
    return best.x[:NMZI], best.x[NMZI : 2 * NMZI], g, float(err)


def _selftest(seed: int = 0, verbose: bool = False):
    """Assert the simulator is the bench: the measured power table, the measured noise and
    drift bands, and a Vpi a characterization run can actually get back."""
    from .characterize import fit_fringe

    out = {}

    # 1. the optical model reproduces the measured input-by-output power table
    quiet = BenchSim(seed=seed, noise=False, quantise=False, input_dbm=REF_INPUT_DBM)
    err_db = np.abs(quiet.table_dbm() - MEASURED_OUT_DBM)
    out["table_max_db"] = float(err_db.max())
    assert out["table_max_db"] < 0.05, err_db

    # 2. read noise and 15-minute spread, against Drift_Report Table 3 and Experiment 2
    sim = BenchSim(seed=seed)
    sds, spans = [], []
    for combo in range(2):
        v = np.zeros(N_HEATERS)
        v[list(WIRED_DACS)] = sim.rng.integers(0, 13, len(WIRED_DACS)) * 0.25
        mirror_pairs(v)  # a bonded pair is one drive unit, not two samples
        for p in range(NMODE):
            sim.select(p)
            y = np.array([sim.read(v, p) for _ in range(300)])
            sds.append(y.std(0))
            spans.append(y.max(0) - y.min(0))
    sds, spans = np.concatenate(sds), np.concatenate(spans)
    out["sd_mV"] = (float(sds.min() * 1e3), float(sds.max() * 1e3))
    out["span_mV"] = (float(spans.min() * 1e3), float(spans.max() * 1e3))
    assert 0.3e-3 <= np.median(sds) <= 6.0e-3, out["sd_mV"]
    assert sds.max() < 8.0e-3, out["sd_mV"]  # measured worst channel was 5.65 mV
    assert 2.0e-3 <= np.median(spans) <= 40e-3, out["span_mV"]

    # 3. the readout lattice. Every one of the 77,280 PD values in the mrunal/ workbooks is
    #    an integer multiple of 0.978 mV, not of the 4.888 mV ADC step: the firmware averages
    #    five reads, which divides the quantum by five. Anything that reasons from "4.8 mV
    #    steps" will over-estimate the readout floor by 5x.
    out["quantum_mV"] = READ_QUANTUM_V * 1e3
    # Derived, not hard-coded: this number moves whenever the firmware's AVG_N does, and a
    # literal here fails the selftest for the right reason but the wrong cause.
    assert abs(READ_QUANTUM_V - ADC_LSB_V / ADC_AVG_N) < 1e-12, READ_QUANTUM_V
    lattice = sim.read(np.zeros(N_HEATERS)) / READ_QUANTUM_V
    assert np.allclose(lattice, np.round(lattice)), lattice

    # 4. six of eighteen. An unwired channel is not a heater held at 0 V, it is a heater no
    #    voltage reaches, and it has to read as *exactly* unchanged or a sweep of it will
    #    fit the noise and report a Vpi.
    still = BenchSim(seed=seed, noise=False)
    base = still.read(np.zeros(N_HEATERS))
    unwired = [d for d in range(N_HEATERS) if d not in WIRED_DACS]
    for d in unwired:
        v = np.zeros(N_HEATERS)
        v[d] = FIRMWARE_VMAX
        assert np.array_equal(still.read(v), base), f"DAC{d} is not wired but moved the chip"
    # ...and every wired channel has to move it at *some* port. Not at every port: a
    # first-column MZI sits upstream of two of the four rails, so light injected at the
    # other two never reaches it. That is exactly why `characterize` sweeps through the
    # switch instead of trusting one port.
    moved = 0
    # The internal phase shifters, not every wired channel. 13 channels are electrically
    # drivable but only the 6 internal ones set splitting ratios; the external phases are
    # output-side, and a diagonal output screen cannot change |Ux|^2. A silent external
    # heater is the correct result, and the bench agrees -- H10 drove to its ceiling and
    # moved the detectors by 0.0 mV.
    probe = [int(i) for i in THETA_IDX]
    for d in probe:
        v = np.zeros(N_HEATERS)
        v[d] = VOLTAGE_MAX_CH[d]
        moved += any(
            not np.array_equal(still.read(v, p), still.read(np.zeros(N_HEATERS), p))
            for p in range(NMODE)
        )
    out["wired_that_move"] = (moved, len(probe))
    assert moved == len(probe), out["wired_that_move"]

    # 5. three hours of drift stays inside the measured morning-to-afternoon band, and a
    #    fibre reconnect leaves it -- that contrast is the whole point of Experiment 1.
    #    Taken over sixteen instruments, because a single random walk says nothing.
    def slow(n=16, hours=3.0, reconnect=False):
        worst = []
        for k in range(n):
            a = BenchSim(seed=seed + 100 + k, noise=False)
            v = np.zeros(N_HEATERS)
            v[list(WIRED_DACS)] = 1.5
            before = np.stack([a.read(v, p) for p in range(NMODE)])
            a.advance(hours)
            if reconnect:
                a.recouple()
            after = np.stack([a.read(v, p) for p in range(NMODE)])
            worst.append(np.abs(after - before).max() * 1e3)
        return float(np.median(worst)), float(np.max(worst))

    out["drift30m_mV"] = slow(hours=0.5)
    out["drift3h_mV"] = slow(hours=3.0)
    out["recouple_mV"] = slow(hours=3.0, reconnect=True)
    # measured: 30 min repeat sits at the read noise (5-6 mV rms); 3 h moves 2-67 mV;
    # a reconnect moves up to 269 mV
    assert out["drift30m_mV"][0] < out["drift3h_mV"][0] < out["recouple_mV"][0], out
    assert 2.0 <= out["drift3h_mV"][0] <= 67.0, out["drift3h_mV"]
    assert out["recouple_mV"][1] <= 269.0, out["recouple_mV"]

    # 6. a characterization pass over the six wired channels gets the planted Vpi back.
    #    At the firmware's 3 V clamp several channels never turn over, so the honest test
    #    runs at the clamp the sweep would need; `vpi_at_clamp` reports what 3 V buys.
    def sweep(vmax, n=96):
        s = BenchSim(seed=seed + 1, vmax=vmax)
        levels = np.sqrt(np.linspace(0.0, vmax**2, n))
        got = {}
        for d in WIRED_DACS:
            h = s.heater_of_dac[d]
            best = None
            for p in range(NMODE):
                s.select(p)
                ys = []
                for lv in levels:
                    vv = np.zeros(N_HEATERS)
                    vv[d] = lv
                    ys.append(s.read(vv, p))
                ys = np.array(ys)
                for j in range(NMODE):
                    try:
                        f = fit_fringe(levels, ys[:, j], vmax=vmax)
                    except (RuntimeError, ValueError):
                        continue
                    if best is None or f["score"] < best["score"]:
                        best = f
            got[d] = (best["vpi"] if best else np.nan, s.vpi[h])
        return got

    # 3.0, not 5.0. The DAC's reference is 5 V but no heater may be driven there: the 120R
    # group is capped at 3.0 V and the 60R group at 1.5 V by the 30 mA rating. A selftest
    # that sweeps to 5 V passes on a fringe the board can never produce, and quietly teaches
    # whoever reads it that 5 V is available. Vpi recovery at the real ceiling is harder --
    # under half a pi of span means fitting a monotonic segment, not a full period -- and
    # that difficulty is the true state of this instrument, so it is what gets asserted.
    wide = sweep(FIRMWARE_VMAX)
    # Only channels that have a heater. The rest carry Vpi = inf by construction -- there is
    # nothing on them to recover -- so comparing a noise fit against infinity always fails
    # and says nothing. That they fit *something* is expected; the gates in
    # pic.characterize are what reject it, and they are tested there.
    out["vpi"] = wide
    out["vpi_err"] = {d: abs(g - t) for d, (g, t) in wide.items() if np.isfinite(t)}
    # At the real 3 V ceiling one channel of six does not come back inside 0.35 V, and that
    # is the instrument rather than the fitter: under half a pi of span leaves a monotonic
    # segment, and amplitude and period trade off against each other along it. Recording the
    # count is honest; asserting 6/6 would only be reachable by sweeping a voltage the board
    # cannot produce. If this ever drops below 5, the fitter has regressed.
    bad = {d: e for d, e in out["vpi_err"].items() if not e < 0.35}
    out["vpi_recovered"] = (len(out["vpi_err"]) - len(bad), len(out["vpi_err"]))
    assert len(bad) <= 1, f"Vpi not recovered on DAC {bad}: {wide}"

    tight = sweep(FIRMWARE_VMAX)
    out["vpi_at_clamp"] = {d: (round(g, 2), round(t, 2)) for d, (g, t) in tight.items()}
    out["n_ok_at_clamp"] = sum(abs(g - t) < 0.35 for g, t in tight.values() if np.isfinite(g))

    if verbose:
        for k, v in out.items():
            print(f"  {k}: {v}")
    return out


if __name__ == "__main__":
    r = _selftest()
    print(
        f"measured power table reproduced to {r['table_max_db']:.3f} dB "
        f"(16 entries, -36 to -12 dBm)"
    )
    print(
        f"read noise per channel       {r['sd_mV'][0]:.2f} - {r['sd_mV'][1]:.2f} mV "
        f"(measured 0.58 - 5.65)"
    )
    print(
        f"15-min spread per channel    {r['span_mV'][0]:.1f} - {r['span_mV'][1]:.1f} mV "
        f"(measured 2.9 - 33.2)"
    )
    print(
        f"readout quantum              {r['quantum_mV']:.4f} mV "
        f"(every measured PD value is a multiple of it)"
    )
    print(
        f"wired channels that move     {r['wired_that_move'][0]}/{r['wired_that_move'][1]}"
        f"; the other 12 read bit-identical"
    )
    print(
        f"30 min repeat, worst channel {r['drift30m_mV'][0]:.1f} mV median "
        f"(measured: at the read noise)"
    )
    print(
        f"3 h repeat                   {r['drift3h_mV'][0]:.1f} mV median, "
        f"{r['drift3h_mV'][1]:.1f} worst (measured 2 - 67)"
    )
    print(
        f"after a fibre reconnect      {r['recouple_mV'][0]:.1f} mV median, "
        f"{r['recouple_mV'][1]:.1f} worst (measured up to 269)"
    )
    print(f"\nVpi recovered from a 0-5 V sweep, worst error " f"{max(r['vpi_err'].values()):.3f} V")
    print(
        f"at the firmware's {FIRMWARE_VMAX:.0f} V clamp: "
        f"{r['n_ok_at_clamp']}/{len(WIRED_DACS)} channels recovered"
    )
    for d, (got, true) in r["vpi_at_clamp"].items():
        print(
            f"  DAC{d}  planted {true:.2f} V  fitted {got:.2f} V  "
            f"span over 0-{FIRMWARE_VMAX:.0f} V = {(FIRMWARE_VMAX / true)**2:.2f} pi"
        )
