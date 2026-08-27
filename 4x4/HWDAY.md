# Hardware day

Ordered so that each run's output is what the next one needs, and so that a session cut
short still leaves something useful behind. Times assume the six wired channels.

## Before power on

- [ ] **Fix `ILIMN` in `tec_pid_1.ino` and reflash.** `NEG_CURRENT_1A = 0x1B5` is **−5.80 A**,
      not the −1 A it is labelled: the datasheet's two limit registers have *different*
      transfer functions (Eq. 9 `ILIMP = 6.8 − code·13.28 mA`, Eq. 10 `ILIMN = code·−13.28 mA`),
      and 0x1B5 is only correct for the positive one. −5.8 A is past the part's own −4.5 A
      minimum sink rating. Use **`0x0000004B`** (75 → −0.996 A). This is the one item on
      this page that can damage hardware.
- [ ] `UV_CLAMP = 0x0F` limits cooling to −1.26 V despite the name `VOLT_CLAMP_0V`. If the
      TEC cannot pull the die down, that is why, not the PID.
- [ ] Know which sketch is on which board. `Setup.ino` and `tec_pid_1.ino` both use **CS 10**
      and cannot share one Arduino.
- [ ] Four serial devices with overlapping globs: board, laser (FTDI), TEC, switch. Set
      `$PIC4_PORT` or pass `--pic-port` / `--laser-port`. One process per port.
- [ ] Laser at **8 dBm**, now the default. That is what the best archive set used and it
      leaves TIA headroom — the brightest of 77,280 logged reads is 0.587 V. 13 dBm has
      never been on this chip.

## Gate 0 — is anything actually on? (5 min)

```bash
./do status                       # laser, TEC, switch, calibration
./do laser state 1 && ./do laser set 8
./do measure                      # heaters at 0 V, live PDs
```

**`laser_status == 1` does not prove emission.** Every session baselines `bfm_optical_power`
off-versus-on and reports `s.emitted`; if that is False, stop — everything downstream fits
noise. Verify optically before believing anything.

Hold the die at 25 °C and let the TEC settle before the first measurement.

## Run 1 — photodiode full scale (~2 min)

```bash
./do char --write --levels 13 --bases 1     # fast pass; includes the normalisation
```

Blocked (`SET 0` on the switch) is 0, transparent is 1, per PD. Everything downstream is in
fractions after this, and it removes the per-PD half of the gain pair the drift fit would
otherwise carry. Retake it whenever a fibre moves.

The search costs ~350 reads because every read is harvested for all four photodiodes and
each PD is only ascended at the port it responds to best; ascending every (port, PD) pair
would cost ten times that for full-scale values that agree to 2 percent. Each PD's full
scale is stored **with the input port it was measured at** -- the same heater vector at a
different port is a different measurement.

**Normalised readings can exceed 1.0.** Full scale is a measured reference, not a ceiling,
so noise lets a later read come in a few percent above it. Do not clamp.

**Watch for:** a dead PD, and the per-port coupling line. The archive says input 3 sits ~6 dB
below its neighbours; if the normalisation reproduces that independently, the bad launch is
real and not a mesh property.

## Run 2 — which DAC is which heater (20 min) ← **the blocker**

```bash
./do char --write                            # full sweep, all ports, all bases
```

Nothing in any document gives the UH ↔ (θ, φ) mapping, and the one assumption in the code is
already refuted: `theory/layout.py` puts DAC0 on an output trimmer, which is invisible in
intensity by construction, yet DAC0 is the strongest modulator on the bench. Until this run
lands, programming and drift correction are both blocked.

**What to read off the digest:** which PDs each channel modulates, at which ports, and the
fitted Vπ. A channel that only ever moves one output is on a different part of the mesh from
one that moves two.

**Also confirms Vπ.** The simulator's fit says 2.27–4.57 V from 8000 archive rows, which
matters a great deal: at the hard 3 V ceiling a heater covers π·(3/Vπ)², so **no wired
channel reaches 2π** and only the lowest-Vπ one passes π. If the sweep confirms it, arbitrary
unitaries are not programmable on this board and the target set has to be restricted. If it
refutes it, that constraint lifts. Either way it is the single most consequential number of
the day.

## Run 3 — transfer matrices (15 min for 200)

```bash
./do dataset --combos 200                    # -> pic_data/transfer.csv
```

**The gap in the whole archive.** Every existing dataset draws a *fresh* random combination
per port, so no DAC state has ever been measured at all four inputs and no transfer matrix
can be assembled from it. Holding the combination and cycling the switch costs nothing extra
and is what makes `T = |U|²` available — which is what turns unitarity into a usable training
constraint and what the drift probe reads.

Take this even if the day runs short. It is new information; the random-combination sets are
not.

## Run 4 — surrogate training set (20 min)

```bash
./do dataset --combos 1000 --out pic_data/train.csv
```

The DPNN saturates by ~2000 samples (R² 0.958/0.998/0.995/0.995 on the archive at 8 dBm),
so 1000 combinations is already past the knee. Prefer this over more combinations at one
port: the four-port version trains the same model *and* supports the unitarity term.

## Run 5 — drift, if there is time (30 min wall, ~2 min of work)

```bash
./do program --random --dynamic              # anchor, then leave it
# ... wait 30+ min, do something else ...
./do program --random --dynamic              # re-probe, see if the correction earns it
```

Expect it to be **rejected** over 30 minutes — the archive says drift is at the noise floor
on that timescale, and refusing is the correct behaviour. If one does get applied, check its
size: below ~0.02 rad it is the gate's known false-positive rate and is harmless. The interesting run is across a
laser disconnect/reconnect, where 75 % of the change is input coupling and the fit should
report one port moving by several dB.

## If the day is cut short

Priority order: Gate 0 → Run 1 → Run 3 → Run 2 → Run 4 → Run 5. Run 3 is above Run 2 only
because it is short and produces data that does not exist anywhere; Run 2 is what unblocks
the project.

## Things that will look like bugs and are not

- Every drift estimate rejected against an uncharacterized chip. The twin and the chip
  disagree far more than any drift does; the gate is doing its job.
- `char` reporting far fewer than 18 heaters identified. Only 6 are wired, and the amplitude
  gate exists because the visibility gate alone passed pure noise on a dark output.
- The four output trimmers never identified. A diagonal output phase screen cannot change
  |U x|², so intensity can never see them. Correct, not a failure.
- Targets reported UNREACHABLE rather than clipped. That is the 3 V ceiling being honest.
