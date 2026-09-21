# 4x4 unitary PIC

Hardware-in-the-loop control of the 4x4 Clements mesh on a Quanfluence **PIC1A** die:
18 thermo-optic heaters (UH1-UH18), four inputs behind a 1x4 optical switch, four outputs
into a PD-TIA board, the die held at 25 C by a TEC, and the same AeroDiode PDMv5 laser the
6x6 rig uses.

The chip next door (`../6x6`) is a contraction that has to be inverted numerically. This one
is unitary, and everything downstream follows from that: a target matrix maps to heater
phases by a formula, the digital twin is exact rather than approximate, and the surrogate
that used to need 25,063 parameters needs 56.

**Status.** The chip is on a bench and its loss, SNR and drift are measured -- see
`mrunal/README.md`, which is the digest of everything the hardware side has produced.
**Six of the eighteen heaters are wired so far.** The control stack here is written for all
18 and self-tested against a simulated instrument; what is still marked PROVISIONAL is which
UH number plays which role in the mesh, which needs the GDS or a per-heater sweep.

## Quick start

Everything runs with no hardware attached.

```bash
./do selftest                  # every model self-test plus a mock rig pass
./do status --mock             # laser, TEC and calibration state
./do char --mock --write       # sweep all 18 heaters, fit fringes, write pic_data/calib.json
./do program --mock --random   # decompose a random unitary onto the chip, read it back
./do train --mock --rounds 2   # train both surrogates from the (mock) chip
```

On hardware, drop `--mock` and light the laser first:

```bash
./do laser state 1 && ./do laser set 1     # ~39 dB of SNR margin at 1 dBm
./do char --write
./do laser state 0
```

Use the project's `pic` conda env. Run from this directory so `pic`, `theory` and `learn`
resolve.

## What is where

| Path | What it is |
|---|---|
| `theory/clements.py` | Exact 4x4 Clements decomposition and its inverse. Round-trips to 1e-15. |
| `theory/layout.py` | What each of the 18 heaters does. **Edit this when the design lands.** |
| `theory/twin.py` | Differentiable twin, unitary by construction, with a coupler-error model. |
| `theory/calib.py` | The heater law `phi = pi (V/Vpi)^2 + phi0`, and its closed-form inverse. |
| `theory/program.py` | Target unitary to heater voltages, plus the gradient polish for a real chip. |
| `pic/` | The rig: serial driver, laser, TEC, acquisition, characterization, `Rig` facade. |
| `pic/layout.py` | Board wiring. The six wired channels are documented; the rest is a placeholder. |
| `pic/devices/tec.py` | The LT8722 PID loop that holds the die at 25 C. |
| `pic/devices/switch.py` | The 1x4 optical input switch. |
| `mrunal/` | Vendor documents, bench reports, bring-up firmware. Read its README. |
| `theory/drift.py` | Drift inference from a four-port unitarity probe. Rank 9, nullity 6. |
| `pic/drift.py` | The closed loop: anchor, re-probe, gate, pre-distort. `--dynamic`. |
| `theory/intensity_matvec.py` | Signed `y = B x` from the photodiodes alone: Sinkhorn, the switched linearisation, the shift decomposition. |
| `theory/matmat.py` | The rungs above it: `Y = B X` under one program, a big matrix tiled into 2x2 blocks, and the O(k) projection that checks a composed product. |
| `pic/matvec.py` | Both on the rig. `--block K` tiles, `--cols N` walks columns, `--unitary` runs the group check. |
| `learn/unitary_fit.py` | The 52-parameter physics surrogate. |
| `learn/dpnn.py` | The reduced pruned network, carried over from the 6x6. Six squared voltages plus the lit input port and the laser telemetry, four photodiodes out. |
| `learn/train_hw.py` | Online trainer: fits both on the same buffer, every round. |
| `Arduino/pic4x4/` | Firmware. 18 DAC channels, 4 ADC pins, switch passthrough. |
| `figs.py` → `figs/` | The three rig diagrams, drawn from the layout modules and the current calibration. `./do figs`. |
| `twin_vs_hardware.py` | How well the twin predicts the real chip over the 135-state four-port table: Pearson and `||P-M||/||M||` for the characterized calibration, and for `learn.transfer_fit` refitted on a held-out split. Same definitions as `../6x6/scripts/twin_vs_hardware.py`, so the two chips compare. |

## The chip

```
in x4 -> [ MZI (0,1) (2,3) ] -> [ MZI (1,2) ] -> [ MZI (0,1) (2,3) ] -> [ MZI (1,2) ] -> phase screen -> out x4
             column 0              column 1           column 2            column 3
```

`figs/mesh.png` draws the same thing with the measured state on it -- which DAC drives
which MZI, the fitted Vpi, and which channels are unusable. `figs/chain.png` is the control
chain from the host to the die and `figs/channels.png` the per-channel voltage ceilings; all
three come from `./do figs` and are redrawn from `pic_data/calib.json`, so re-run it after a
characterization.

Six interferometers, two heaters each (an internal arm phase `theta` and an external input
phase `phi`) = 12, plus a three-rail output phase screen = 15. The fourth output phase is a
global phase, which no detector sees and no heater needs to make. The die carries 18 physical
pads for that 15-phase mesh, 3 more than it needs; the DAC81416 drives only 16 of the 18, and
of the 4 output-screen pads only 2 have a driver, so the mesh can realise a target unitary
only up to one uncharacterised phase on the undriven, non-reference output rail
(`theory/layout.py UNREACHABLE_RAIL`).

### Heater census

One heater per row, DAC channel alongside. `Vtop` = swept ceiling, `min(3 V, Vmax)`.
`swing = 2B` on the best PD. From `pic_data/sessions/2026-08-27/`.

`n` = how many independent fringe fits of that heater cleared the amplitude and SNR gates, out
of 34 on offer: 4 input ports x 2 base biases x 4 PDs = 32 traces, plus the 30 mA and 40 mA run
fits. `Vpi` is the 40 mA run's value, $\pm$ the sd of those `n`; `[min, max]` their extremes.

Phase is linear in heater power -- `phi = pi * P / Ppi` with `Ppi = Vpi^2/R` -- so `Ppi` is the
mW that buy one pi and is the number to design against; `Vpi` alone is not comparable across the
two resistance groups. The 40 mA ceiling is `Pmax = 1.6*R` mW, so reach in pi is `Pmax/Ppi`.
**Bold `Ppi` = pi is reachable at 40 mA**; 3 of 10.

| H | DAC | R (ohm) | Vtop (V) | Vpi (V) | [min, max] (V) | n | Ppi (mW) | swing (mV) | r2 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| H1 | 15 | 114.1 | -- | -- | -- | -- | -- | -- | -- |
| H2 | 14 | 116.8 | 3.00 | -- | [2.94, 4.75] | 2 | -- | -- | -- |
| H3 | 5 | 113.5 | 3.00 | $4.51 \pm 0.34$ | [3.72, 4.92] | 12 | 179 | 255.1 | 1.000 |
| H4 | 4 | 57.8 | 2.30 | $3.01 \pm 1.00$ | [3.01, 4.43] | 2 | 157 | 36.4 | 0.990 |
| H5 | 13 | 117.1 | 3.00 | $5.94 \pm 0.99$ | [3.66, 5.94] | 5 | 301 | 177.6 | 0.996 |
| H6 | 11 | -- | -- | -- | -- | -- | -- | -- | -- |
| H7 | 12 | 116.1 | 3.00 | $3.94 \pm 0.15$ | [3.94, 4.27] | 4 | **134** | 83.8 | 0.998 |
| H8 | 3 | 114.1 | 3.00 | $4.60 \pm 0.39$ | [3.77, 4.73] | 7 | 185 | 272.1 | 1.000 |
| H9 | 7 | 57.2 | 2.25 | $3.60 \pm 1.07$ | [3.60, 5.12] | 2 | 227 | 39.9 | 0.979 |
| H10 | 10 | 56.4 | 2.25 | -- | [1.94, 5.35] | 2 | -- | -- | -- |
| H11 | 2 | 118.8 | 3.00 | $4.06 \pm 0.68$ | [4.00, 5.80] | 8 | **138** | 256.9 | 0.999 |
| H12 | 1 | 62.3 | 2.45 | $3.60 \pm 0.65$ | [2.67, 5.30] | 10 | 208 | 100.9 | 0.999 |
| H13 | 9 | 114.9 | 3.00 | $3.70 \pm 1.39$ | [3.70, 5.67] | 2 | **119** | 36.5 | 0.993 |
| H14 | 8 | -- | -- | -- | -- | -- | -- | -- | -- |
| H15 | 0 | 114.1 | 3.00 | $4.68 \pm 0.35$ | [4.11, 5.33] | 13 | 192 | 418.2 | 1.000 |
| H18 | 6 | -- | -- | -- | -- | -- | -- | -- | -- |

H16/H17 have no row: the DAC81416 is 16 channels against 18 pads. H18/H14/H6 were never
ohmmetered -> staged at 1.5 V, never swept. H1 has R but was never swept. H2/H10 were fitted
and rejected at SNR 4.0 / 4.5. All four 60-ohm heaters fall short of pi. Ppi median 182 mW
(119-301); all ten fitted at pi at once = 1.84 W into the die. No per-heater drift figure
exists: the only repeats are the 30 mA and 40 mA runs, which differ in swept range, so
`[min, max]` is fit spread, not time.

## What is settled

Everything below was measured against a simulated instrument with a known answer, which is
how the procedure gets settled before bench time is spent on it.

**Programming is a formula.** The Clements decomposition round-trips to 1e-15, and the
ideal twin reproduces it to the same precision. A 2 percent coupler-error mesh realises the
exactly-decomposed phases at fidelity 0.9995; a gradient polish over the phases, warm-started
from that decomposition, recovers 1.0000 in under a second.

**Sweep on a uniform-V-squared grid.** Phase goes as V^2, so the fringe is a pure sinusoid
against V^2, which makes Vpi an FFT peak rather than a search. It also fixes the sampling
rule: a uniform-in-V grid aliases near the top of the range, and an aliased fit comes back
at r2 = 1.0000 with a better residual than the honest ones. `resolvable_span` rejects them
on sample count, which is the only thing that catches it.

**Characterize from several base biases and every input port.** With the mesh at 0 V and one
input port lit, most heaters modulate nothing: an external phase is invisible unless its
interferometer splits light, and on a first-column MZI with one port lit it is a global
phase. Sweeping from a couple of random base biases through all four ports -- which the
optical switch makes a loop index rather than four fibre re-plugs -- identifies 10 of the 12
mesh heaters with Vpi exact. The last two are the first-column external phases, and no amount
of switching finds them: they need light in two ports at once, which a 1x4 switch cannot do
and an external splitter can.

**The output trimmers are unobservable and it does not matter.** A diagonal output phase
screen cannot change `|U x|^2`, so intensity detection can never see them. They come back
rejected, correctly, and stay unidentifiable in the joint fit, which costs nothing because
anything invisible to the readout cannot affect a result read out in intensity.

**Bootstrap before you refine.** A joint fit from a nominal start plateaus at R^2 = 0.42 no
matter how many restarts it gets: 18 unknown Vpi means the forward map oscillates at the
wrong frequency everywhere at once. Hand it the Vpi the fringe sweeps recover and the same
fit reaches 1.0000.

## Still open

- ~~**Which UH is which.**~~ Closed for all 16. `theory/layout.py` reads the role off
  `ananya/4X4MZI_REPORT.pdf`'s labelled mesh diagram, cross-checked against its resistance
  table, the vendor pinout and the wiring notebook; the old placeholder predicted the bench
  data at r = -0.04 held out, the corrected map predicts it (DAC 12, 13 are mid-mesh phi, not
  trimmers, exactly as their clean fringes require). The last two, DAC 7 and 11, are placed
  in column 1's pass-through gap and are exactly redundant with phases the model already
  commands -- not unidentified, and not worth driving. What is still open is the *undriven*
  pad on output rail 3, which nothing on the part can reach.
- **Vpi is unknown on nitride.** The 1.5 V default comes from the 6x6 silicon chip. With the
  board capped at 3 V a heater only reaches a full 2 pi if Vpi <= 2.12 V, which is the first
  thing a sweep has to confirm.
- **Input 3 loses 6 dB more than its neighbours** and splits 82% into one output. That looks
  like coupling on that port rather than a mesh property, and it will skew any
  characterization run through it until it is fixed or accounted for.
- **The paired-port injection** needs an external splitter. Two heaters stay uncharacterized
  without it.
- Whether the laser's own TEC is calibrated on this unit. It is run off, as on the 6x6; the
  working cooler is the separate die TEC.
