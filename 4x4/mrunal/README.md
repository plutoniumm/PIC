# mrunal/ — the delivered 4x4 hardware, as documented

Everything the vendor and the bench have produced for this chip: the Quanfluence tapeout
design review, the packaging IO map, the two Arduino sketches actually flashed, three
component datasheets, three raw datasets, and a series of measurement reports by **Mrunal
Kumavat** (June–August 2026) with heater routing voltages contributed by **Ananya**.

**Read this file instead of the 21 source files.** It is documentation, not runtime — nothing
in `pic/`, `theory/` or `learn/` imports from here — but it is where every PROVISIONAL
constant in `pic/config.py` and every number in `../README.md` comes from. Contradictions
between the sources are collected in the last section rather than smoothed over; several of
them are load-bearing.

Provenance: designed and packaged by **Quanfluence**, fabricated on the **Ligentec AN800**
silicon-nitride C-band process in MPW run `LGT-MPW-AN800-26` (tapeout submitted 29 Mar 2023),
packaging IO released 29 Sep 2025, bench brought up at IIT Madras through 2026. This is a
different vendor, process and material system from the 6x6 chip in `../../6x6` (SJTU-Pinghu
design, AMF silicon-on-insulator) — none of that chip's numbers transfer.


## 1. The chip

`PIC1A` is a **multi-project die**, 10.52 × 4.87 mm, split into two reticle halves: `TC_QC`
(quantum computing) and `TC_PA` (parametric amplifier). The 4x4 unitary is one block of 24.

| Process fact | Value | Source |
|---|---|---|
| Platform | Ligentec AN800, Si₃N₄, C-band | Design review s3 |
| Waveguide cross-section | 1000 nm × 800 nm, single mode | s3 |
| Propagation loss | 0.2 dB/cm (≈0.21 dB across the die) | s3, used by `Loss_Analysis` |
| Minimum bend radius | 50 µm | s3 |
| Modules | X1 (base), **P1+ (high-power, high-stability heaters)**, M1 (heaters), X2, RIB, LoCA-i/s, ExSpot (SMF28 spot-size converters), TCD, AFD (etched facets) | s3 |
| Facet coupling | inverted-taper black box, IL <0.81 dB TE / <1.18 dB TM | s7 |
| 2x2 splitter | 50/50, EL 0.22–0.3 dB, splitting-ratio σ 1.46–2.42 dB TE | s7 |
| Waveguide crossing | EL 0.02 dB TE, crosstalk < −60 dB | s8 |

**Block inventory** (design review s9–s13): 10 micro-ring resonators (MRR_{300..500}_{1800,2300},
r=113 µm) and 6 nanophotonic/concentric molecules as squeezers; a **4x4 MZI unitary**; a single
2x2 MZI (test); an asymmetric-MZI 1D cluster-state generator (14.2 cm, 0.43 dB propagation);
a QRNG (3 MZIs, 8 outputs, LO); coupler loopbacks; a reference straight waveguide; 50 µm
bends; and a spiral parametric amplifier (2.3 µm wide, 10 dB total loss, 2.5 dB/facet).
Only the 4x4 unitary and the single test MZI have wire-bonded heater pads.

### 1.1 The 4x4 mesh topology

The design review's block diagram (s17, reconstructed from the shape coordinates) is an
unambiguous **Clements rectangular mesh**, four modes, four columns:

```
rail 1 ─┬─ MZI 1 ─┬───────────┬─ MZI 4 ─┬───────────┐
rail 2 ─┴──(H1)───┴┬─ MZI 3 ─┬┴──(H4)───┴┬─ MZI 6 ─┬┤
rail 3 ─┬─ MZI 2 ─┬┴──(H3)───┴┬─ MZI 5 ─┬┴──(H6)───┴┤
rail 4 ─┴──(H2)───┴───────────┴──(H5)───┴───────────┘
        col 1: (1,2)(3,4)   col 2: (2,3)   col 3: (1,2)(3,4)   col 4: (2,3)
```

This matches `theory/clements.MESH` exactly. Note the deck names **only six heaters, H1–H6,
one per MZI**, while the packaged part exposes **eighteen** (§1.3) — the deck's H-numbers are
routing labels, not the pad numbering.

**Simulated performance** (Lumerical Interconnect, arcs and waveguides excluded):

| Quantity | Value |
|---|---|
| Single-MZI output power at Φ=π | 68.4 – 76.4 % (IL 1.17 – 1.65 dB) |
| Full-mesh routed-path IL, 16 input→output pairs | 2.34 – 4.72 dB (median ≈4.1) |

The deck's routing table (s21) gives the binary H1–H6 pattern for each of the 16 paths. Two
structural facts fall out of it: **H3 is 0 in all sixteen rows**, and each input reaches its
four outputs by toggling at most two heaters.

### 1.2 Optical port map

Edge-coupled facets, left = inputs, right = outputs. Odd GDS port sequence numbers only —
16/18/20 are skipped on both sides.

| Signal | Facet | GDS port seq | Signal | Facet | GDS port seq |
|---|---|---|---|---|---|
| U_IN1 | L15 | 15 | U_OUT1 | R24 | 15 |
| U_IN2 | L17 | 17 | U_OUT2 | R22 | 17 |
| U_IN3 | L19 | 19 | U_OUT3 | R20 | 19 |
| U_IN4 | L21 | 21 | U_OUT4 | R18 | 21 |

Neighbouring blocks on the same facets, in case a fibre lands one port off: L14/L13 =
M_OUT1/M_OUT2 and L12/L10 = M_IN2/M_IN1 (single MZI), L23/L24 = 1DC_IN1/2, R17/R16/R14/R13 =
1DC_OUT1–4, R15 and R30 = **LO inputs**, R28–R31 and R34/R33/R32 = QRNG outputs, L27–L31 and
L34–L38 = MRR inputs (each MRR is bonded out twice), L3/L4 and L32/L33 = loopbacks.

### 1.3 Heater pad table (the thing everything downstream needs)

Every heater is a pad pair. The GND pads are two common rails — `TP1 TP6 TP11 TP15 TP21 TP26`
(top) and `BP1 BP7 BP15 BP20 BP26` (bottom) — so **the GND pad in a row identifies nothing;
the (+)ve pad is the heater's unique identity.**

| Block | Heater | GND | (+)ve |   | Block | Heater | GND | (+)ve |
|---|---|---|---|---|---|---|---|---|
| MZI | MH1 | TP1 | TP4 | | 4x4 | UH9 | BP15 | BP14 |
| MZI | MH2 | TP1 | TP7 | | 4x4 | UH10 | TP15 | TP13 |
| MZI | MH3 | TP6 | TP5 | | 4x4 | **UH11** | BP20 | **BP16** |
| 4x4 | UH1 | BP1 | TP8 | | 4x4 | **UH12** | BP20 | **BP17** |
| 4x4 | UH2 | BP1 | BP11 | | 4x4 | UH13 | TP26 | TP23 |
| 4x4 | **UH3** | BP1 | **TP9** | | 4x4 | UH14 | TP26 | TP25 |
| 4x4 | **UH4** | BP7 | **BP9** | | 4x4 | **UH15** | TP26 | **TP24** |
| 4x4 | UH5 | BP7 | BP10 | | 4x4 | UH16 | TP26 | TP22 |
| 4x4 | UH6 | TP11 | TP10 | | 4x4 | UH17 | TP26 | TP16 |
| 4x4 | UH7 | BP15 | BP12 | | 4x4 | UH18 | TP26 | BP21 |
| 4x4 | UH8 | BP15 | BP13 | | | | | |

**Bold = the six heaters wired to DACs today** (§2.1). The single test MZI gets 3 pads
(MH1–MH3) and the six-MZI 4x4 gets 18 — consistent with **three heaters per MZI**, but no
document in this directory says which UH is which MZI's θ or φ. That mapping is the single
most important missing datum (§8, item 1).


## 2. The bench as delivered

```
Arduino ──SPI──> 16-ch DAC (6 used) ──> UH heaters
        ──Serial1 9600──> Sercalo 1x4 switch ──> U_IN1..4
laser ──> switch common
U_OUT1..4 ──> InGaAs PD + TIA ──> Arduino A0..A3
second Arduino ──SPI──> LT8722 ──> Peltier;  10k NTC ──> A0
```

### 2.1 DAC → heater wiring

From `Setup_Analysis.pdf` Table 1 and `Drift_Report.pdf` Table 1 (identical tables), joined
against the vendor pad table above:

| DAC ch | (+)ve pad as written | Vendor owner of that pad | Heater |
|---|---|---|---|
| 0 | `BP24` | — (no such heater pad; TP24 = UH15) | **UH15** |
| 1 | BP17 | UH12 | **UH12** |
| 2 | BP16 | UH11 | **UH11** |
| 3 | BP13 | UH8 | **UH8** |
| 4 | BP9 | **UH4** | **UH4** (tables say UH7 — see §8.2) |
| 5 | TP9 | UH3 | **UH3** |

Twelve of eighteen heaters are unbonded to the board. `pic/config.py` exposes this as
`WIRED_DACS`, but **the code's DAC indices are model-heater indices, not these bench indices** —
see §8, item 1.

### 2.2 `Setup.ino` — the data-collection sketch

One 16-channel DAC on `CS = 10`, SPI mode 1 at 10 MHz; `configureDAC` writes `03 00 84` then
`09 00 00`; a channel write is `(0x10|ch), code>>8, code&0xFF` with
`code = (V/5.0)·65535` (16-bit, 5 V reference). Per-channel clamp `maxVolt = 3.0 V`.

Loop: `START` over USB at 115200 → for each port 1..4 { `SET <port>` on `Serial1` @9600,
wait 1000 ms; for each combination { six random levels from `random(0,13)·0.25` (0→3 V in
0.25 V steps), apply, wait **500 ms**, read } } → `DONE`. A read is 5 sweeps of `analogRead`
on A0–A3 spaced 10 ms, averaged, then scaled by `5.0/1023`.

Two consequences that matter:

- **Effective ADC quantisation is 0.978 mV, not 4.888 mV.** The raw LSB is 5/1023 = 4.888 mV,
  but the average of five raw counts is taken *before* scaling, so the output grid has step
  LSB/5. The datasets confirm this exactly (smallest non-zero difference = 0.00098 V).
- **500 ms is the only settling allowance.** The 6x6 rig measured thermo-optic τ ≈ 0.7–1.2 s
  with t99 at 3–5 s. Every dataset here is therefore taken part-way through the thermal
  transient, in a state that depends on the previous combination.

`NUM_COMB` ships as 5; the real runs used 1000, 50 and 900 (§5).

### 2.3 PD-TIA

InGaAs PIN photodiode, 1310–1550 nm, **ℜ ≈ 0.9 A/W**, into an inverting TIA.

| Parameter | Value |
|---|---|
| Feedback resistor `Rf` | 4.4 kΩ (4.3 kΩ effective in the noise maths) |
| Feedback capacitor `Cf` | 1 nF |
| −3 dB bandwidth | 1/(2π·4300·1n) ≈ **37 kHz** |
| Full-scale output | ≈4.44 V at 0.5 dBm (1.01 mA) |
| Dominant noise | `Rf` Johnson noise, 1.96 pA/√Hz — **99.9 %** of the total 1.961 pA/√Hz |
| NEP as derived in that report | **2.18 pW/√Hz** (≠ the 3.08 used elsewhere; §8.4) |

Volts↔power for reading any dataset: `P[W] = V / (4400 · 0.9)`, i.e. 1 mV ≈ 0.253 µW ≈ −36 dBm.

### 2.4 Optical switch

Sercalo **SC/mSC coaxial 1xN MEMS switch with interface**, product 38-1180, rev 1.19. Single
MEMS mirror, bidirectional, supply 4.75–5.25 V. The bench uses the UART interface: **9600 baud
after every power-on or reset**, ASCII commands, device always replies, do not send the next
command before the reply.

| Command | Meaning |
|---|---|
| `SET <P>` | route channel P (1..N) to the common port; `P=0` opens the path. Replies `SET <P>`. This is the one `Setup.ino` sends. |
| `POS` | read back the current channel |
| `BAND <b>` | optical band: 0 = O (1250–1350), **1 = C (1510–1580)**, 2 = L. `DBAND` sets the power-on default, stored in flash |
| `ID`, `RST`, `ERM`, `UART`, `PTY`, `IIC`, `STB`, `BASE` | identity, reset, error verbosity, baud, parity, I²C address, strobe-pin mode, parallel port numbering |

Two operational notes: `Setup.ino` never reads the reply, it just waits 1000 ms; and the
**optical band setting is persistent flash state** that no document here records as having
been set to C-band. The datasheet is the interface manual only — its §14 says "refer to the
datasheet for the optical specifications", so the **3.1 dB switch insertion loss** used in the
link budget has no source in this directory.

### 2.5 TEC — `tec_pid_1.ino` + LT8722

Analog Devices **LT8722**, ±4 A / 15 V full-bridge TEC driver, 3.1–15 V VIN, one 25-bit output
voltage DAC and two 9-bit current-limit DACs, SPI up to 10 MHz. Sketch runs SPI mode 0 at
1 MHz on `CS = 10`.

Frame: `0xF2, addr, D[31:24], D[23:16], D[15:8], D[7:0], CRC8, 0x00`, CRC-8 poly `0x07`, init 0,
over the six preceding bytes. The datasheet's register addresses have `A[0]` always zero, so
the sketch's byte-level addresses are the register numbers shifted left by one — all six check
out:

| Sketch constant | Byte | Register |
|---|---|---|
| `ADDR_COMMAND` | 0x00 | SPIS_COMMAND (0x00) |
| `ADDR_STATUS` | 0x02 | SPIS_STATUS (0x01) |
| `ADDR_ILIMN` | 0x04 | SPIS_DAC_ILIMN (0x02) |
| `ADDR_ILIMP` | 0x06 | SPIS_DAC_ILIMP (0x03) |
| `ADDR_DAC` | 0x08 | SPIS_DAC (0x04) |
| `ADDR_OV_CLAMP` | 0x0A | SPIS_OV_CLAMP (0x05) |
| `ADDR_UV_CLAMP` | 0x0C | SPIS_UV_CLAMP (0x06) |

Output scaling: the sketch computes `code = V·2²⁴/(3.0·16·0.42)`. The apparently arbitrary
`GAIN_ADJUST = 0.42` is not a fudge — `3.0 × 0.42 = 1.26 ≈ 1.25`, recovering the datasheet's
`VOUT = (SPIS_DAC/2²⁴)·1.25·16`. Software clamps the command to ±2 V.

Applying the datasheet's transfer functions to the sketch's written values:

| Sketch write | Datasheet | Actual effect |
|---|---|---|
| `ILIMP = 0x1B5` (437) | `ILIMP = 6.8 A − code·13.28 mA` | **+0.997 A** — correct, as labelled |
| `ILIMN = 0x1B5` (437) | `ILIMN = −code·13.28 mA` | **−5.80 A** — not the labelled 1 A, and beyond the ±4 A part rating. For −1 A the code is 75 (0x4B) |
| `OV_CLAMP = 0x04` | max SPIS_DAC 0x004FFFFF | ≈ **+6.3 V**, as labelled |
| `UV_CLAMP = 0x0F` | min SPIS_DAC 0xFFF00000 | ≈ **−1.26 V**, not the labelled 0 V — and it silently truncates the PID's −2 V cooling authority |

Temperature sense: 10 kΩ NTC on A0 in a divider with a 10 kΩ fixed resistor off 5 V;
β = 3450, R₀ = 10 kΩ at 25 °C; single-pole EMA on resistance with α = 0.08; a hard
`+1.3 °C` calibration offset added to the result. Control: `PID_v1`, REVERSE, setpoint
**25.0 °C**, sample 1000 ms, output limits ±2.0 V, main loop 200 ms. **Kp = 5, Ki = 0, Kd = 0** —
it is a proportional-only loop, so it holds a load-dependent steady-state offset, which the
+1.3 °C constant is presumably absorbing. `t` on the serial line runs a ±1 V polarity test.

The two sketches both claim `CS = 10`, so they are two separate boards or two separate
sessions; they cannot co-exist on one Arduino as written.


## 3. What has been measured

### 3.1 Insertion loss, all 18 heaters at 0 V

External power meter at each facet, `Pin = 0.96 dBm` (1.247 mW). `Week_4_Report.pdf`
(3 Jul 2026), reused verbatim as `SNR_Analysis` §1.1.

Measured output power (dBm):

| in ↓ / out → | o1 (R24) | o2 (R22) | o3 (R20) | o4 (R18) | Σ out | total loss |
|---|---|---|---|---|---|---|
| i1 (L15) | −14.00 | −22.00 | −19.20 | −11.92 | −9.12 dBm | 10.08 dB |
| i2 (L17) | −15.03 | −19.20 | −13.78 | −13.98 | −9.02 dBm | 9.98 dB |
| i3 (L19) | −34.00 | −16.66 | −24.00 | −36.00 | −15.82 dBm | **16.78 dB** |
| i4 (L21) | −14.14 | −21.88 | −19.03 | −14.64 | −10.37 dBm | 11.33 dB |

Subtracting the 0.21 dB of on-chip propagation leaves 9.77–16.57 dB, essentially all of it
fibre-to-facet coupling at the two ends. **Input 3 is ~6 dB worse than the other three**, and
82 % of what does get through i3 exits on o2 alone — a bad launch on L19, not a mesh property.
The same port is the worst in the drift experiments (§3.3), which corroborates it.

Row-normalised split, all heaters cold: i1 → 32.5/5.2/9.8/52.5 %, i2 → 25.1/9.6/33.4/31.9 %,
i3 → 1.5/82.3/15.2/1.0 %, i4 → 41.9/7.1/13.6/37.4 %.

### 3.2 SNR and the link budget

`SNR_Analysis (1).pdf` (3 Jul 2026) and `Loss_Analysis (1).pdf` (17 Jul 2026), both built on
NEP = 3.08 pW/√Hz and Δf = 37 kHz:

| Quantity | Value |
|---|---|
| Integrated noise power `NEP·√Δf` | 5.9245 × 10⁻¹⁰ W = **−62.27 dBm** |
| SNR at −20 dBm in | 16 879 ≈ 84.5 dB |
| SNR at +0.5 dBm in | 1.89 × 10⁶ ≈ 125.5 dB |
| Power for 3 dB SNR | **837 pW** per photodiode |

Budget: −62.27 (floor) +3 (SNR) +12 (MZI IL) +6 (coupling) +3.1 (switch) ⇒ **required laser
−38.17 dBm**; at 1 dBm in and 21.1 dB of loss the receiver sees −20.1 dBm.

**This budget describes the TIA, not the instrument.** 5.92 × 10⁻¹⁰ W at ℜ = 0.9 A/W through
Rf = 4.4 kΩ is **2.3 µV** at the TIA output — 1/2000 of one ADC LSB. The real per-photodiode
floor is set by the 10-bit Arduino ADC and the observed reading-to-reading spread:

| Floor | Value | In optical power |
|---|---|---|
| Raw ADC LSB | 4.888 mV | −29.1 dBm |
| Effective step after the 5× pre-average | 0.978 mV | −36.1 dBm |
| Measured repeat-to-repeat RMS (§3.4) | 4.2 – 8.0 mV | **−29.7 … −26.9 dBm** |

That is ~33 dB above the budgeted floor, and it explains the rest of the record: at
`Pin ≈ 1 dBm` the i3 outputs sit at −34 and −36 dBm, *below* the practical floor, and 2–20 %
of the readings in the low-power dataset are pinned at exactly 0.000 V (PD2 and PD3 worst). The
`MZI Presentation` (31 Jul 2026) states the problem in the field — "find the combinations of
DAC voltages where the output voltage is above the noise floor" — and its two proposed fixes
(more input power, or restrict to bright combinations) are the right ones. The 8 dBm dataset
(§5.3) is the experimental confirmation: median PD reading rises from 9.8 mV to 112 mV.

### 3.3 Drift

`Drift_Report.pdf` (22 Aug 2026), three experiments, all six wired heaters driven, TEC on.
The report's "Variance" is `Σᵢ₌₁..₅₀ Σⱼ₌₁..₄ (PD_old − PD_new)²`, a **sum of 200 squared
differences**, not a variance; per-reading RMS is `√(V/200)`. All three tables below were
recomputed from the raw sheets and reproduce the report exactly.

**Experiment 1 — 50 fixed combinations, three sessions.** DS2 at +30 min; DS3 the next day
at +12 h, after deliberately unplugging and replugging the input fibre.

| Port | DS1−DS2 (30 min) | DS2−DS3 (12 h + reconnect) | DS1−DS3 | per-reading RMS, 12 h |
|---|---|---|---|---|
| 1 | 0.0084 | 0.1122 | 0.1053 | 23 mV |
| 2 | 0.0070 | 0.0133 | 0.0177 | 9 mV |
| 3 | 0.0044 | **1.6070** | **1.5425** | **88 mV** |
| 4 | 0.0029 | 0.0527 | 0.0447 | 15 mV |

30 minutes with nothing touched costs 3.8–6.5 mV RMS — at or below the readout floor. The
12 h comparison is 2.5–350× worse and is dominated by port 3, the same port that was 6 dB lossy
in §3.1. The report attributes it to the fibre reconnection rather than to time, correctly:
nothing separates "12 hours" from "replugged" in this design, but the 30-minute control rules
out ordinary time drift at this magnitude.

**Experiment 2 — 900 samples at 1 Hz, fixed bias, 15 min, ×4 ports ×2 combinations.**
Max−min per channel ranges **2.9 – 33.2 mV** (σ 0.6 – 5.7 mV), broadly uniform across ports.
The near-zero entries (PD2/port3 and PD3/port2, both ≈0.5 mV) are dark channels, not quiet
ones — their means are 0.4–0.6 mV.

**Experiment 3 — same day, morning 11:45 vs afternoon 14:38.** Fixed bias
(DAC0–5 = 2.0, 1.5, 1.0, 2.0, 0.25, 0.0 V), the switch cycling 1→2→3→4 continuously for
45.7 min, 440 cycles per port (≈1.56 s per port visit).

| Port | max\|Δ\| PD0 | PD1 | PD2 | PD3 |
|---|---|---|---|---|
| 1 | 0.0635 | 0.0196 | 0.0166 | 0.0362 |
| 2 | 0.0665 | 0.0235 | 0.0225 | 0.0020 |
| 3 | 0.0655 | 0.0655 | 0.0675 | 0.0029 |
| 4 | 0.0587 | 0.0156 | 0.0156 | 0.0577 |

No ramp within either 46-minute run — the traces are flat with cycle-to-cycle ripple — but a
step of up to 67 mV between morning and afternoon at identical bias. **PD0 is the noisiest
channel throughout** (σ 0.017–0.022 V vs 0.0004–0.010 V for PD1–3), which the report
reasonably reads as PD0 sitting on the steep part of its transfer curve. The report's other
claim — that the afternoon run is noisier — is **not** what the data says (§8.7).

### 3.4 Repeat noise

`Combined_Project_Data (1).xlsx` measures every one of 4000 (port, combination) points
**twice**, which makes it the 4x4 equivalent of the 6x6's repeat-noise set. Recomputed here:

| | PD0 | PD1 | PD2 | PD3 |
|---|---|---|---|---|
| RMS of the repeat difference | **8.0 mV** | 5.9 mV | 4.3 mV | 4.2 mV |
| mean \|difference\| | 6.1 mV | 4.3 mV | 3.0 mV | 3.1 mV |
| max \|difference\| | 32 mV | 29 mV | 19 mV | 19 mV |
| signal mean over the run | 18.2 mV | 34.4 mV | 14.4 mV | 12.1 mV |

So at the input power used for that run the per-reading SNR is between **2 and 6** — one to
two and a half bits. Averaging N repeats buys √N and nothing else; the fix is input power.


## 4. Interpreting the two heater-voltage tables

`SNR_Analysis` §1.4 carries Ananya's measured heater settings that maximise a given path.
They are the only heater→behaviour data on this chip, and they use the **design review's
H1–H6 names**, which nothing joins to the UH pads or the DAC channels.

| Path | H6 | H5 | H4 | H3 | H2 | H1 |
|---|---|---|---|---|---|---|
| IN1–OUT1 | | | 0.0 | 3.0 | | 2.5 |
| IN2–OUT1 | | | 3.0 | 3.0 | | 0.0 |
| IN1–OUT2 | 3.0 | | 3.0 | 3.0 | | 1.5 |
| IN2–OUT2 | 3.0 | | 0.25 | 0.25 | | 1.5 |
| IN3–OUT3 | 3.0 | 1.25 | | 2.25 | 1.5 | |
| IN4–OUT3 | 3.0 | 0.25 | | 0.75 | 0.25 | |
| IN3–OUT4 | | 0.0 | | 2.5 | 0.25 | |
| IN4–OUT4 | | 1.5 | | 3.0 | 1.5 | |

Structurally this is exactly right for the Clements layout of §1.1: OUT1/OUT2 paths use
{H1, H4, H6}, OUT3/OUT4 paths use {H2, H5, H6}, and H3 (the column-2 crossing) appears in all
of them. It disagrees with the deck's simulated routing table, which holds H3 at 0 throughout
(§8.8). Both tables also predate the current bench, and the loss figures in §3.1 were taken
with the heaters *cold*, so no per-path insertion loss has ever been measured — the report
says so explicitly.


## 5. Raw datasets

Three workbooks, ~9 MB, and they are the only measurement data this chip has. All PD columns
are volts at the TIA output; all DAC columns are commanded volts on the 13-level 0→3 V grid.
Every dataset varies **all six wired heaters simultaneously** — there is no single-heater
sweep anywhere.

Common quirks: several sheets carry **side-by-side blocks** at fixed column offsets separated
by a blank column, the last row of a block is a `MIN` / `MAX` / `SUB` spreadsheet summary
rather than data, and `Reading_1` ends with a stray `DONE` row (the sketch's end-of-run
banner). Load with `header=None` and drop non-numeric rows.

### 5.1 `Combined_Project_Data (1).xlsx` — 3 Aug 2026, the repeat-pair set

| Sheet | Shape | Content |
|---|---|---|
| `Reading_1` | 4000 × 12 (+`DONE`) | first pass |
| `Reading_2` | 4000 × 12 | second pass, identical port and DAC columns |
| `DAC_voltages` | 4000 × 7 | duplicate of the port + DAC columns, written **without a header row** (a naive `read_excel` eats row 0 and mangles a column name) |
| `Drift` | 4000 × 6 | `Port, Step, PD0_Error..PD3_Error` = `Reading_2 − Reading_1` |
| `Plots`, `Sheet1` | — | empty |

Columns 0–11 of the reading sheets are unnamed: `port, combination, DAC_0..DAC_5, PD_0..PD_3`.
4 input ports × 1000 random combinations, a **different** random set per port (3996 distinct
of 4000). `Step` restarts per port in `Reading_1` and runs 0–3999 globally in `Reading_2`;
the rows are aligned regardless. Input power is not recorded and the PD levels (mean 12–34 mV)
are low — this is the run that motivated the 8 dBm retake.

### 5.2 `Drift_data.xlsx` — 22 Aug 2026, the source for `Drift_Report.pdf`

| Sheet | Layout | Content |
|---|---|---|
| `Experiment_1` | blocks at col 0 / 13 / 26, then difference and squared-difference blocks from col 39 | DS1, DS2 (+30 min), DS3 (+12 h, fibre replugged); 50 combinations × 4 ports = 200 rows each, DAC columns identical across the three |
| `Experiment_2` | blocks at col 0 / 13 | two fixed DAC combinations, each 900 samples/s × 4 ports = 3600 rows |
| `Experiment_3` | morning at col 0, afternoon at col 13, \|Δ\| at col 26 | 440 switch cycles × 4 ports = 1760 rows each; carries **`elapsed_ms`** instead of `combination` |

`Experiment_3` is the only dataset in the directory with a time axis.

### 5.3 `PD_Drift (1).xlsx` — 14 Aug 2026

`Experiment_1` and `Experiment_2` are **numerically identical** to `Drift_data.xlsx`'s — this
workbook is superseded for those. Its unique content is:

**`Experiment_3(8dBm)`** — 4 ports × 1000 random combinations at **8 dBm** input, single
12-column block. Median PD 112 mV, max 1.005 V, and only 0–34 % of channel readings under
10 mV (vs 38–65 % in `Combined_Project_Data`). This is the dataset to model against.

### 5.4 What these datasets can and cannot support

**Can:** fit a 6-input → 4-output surrogate per input port (4 × 1000 samples at 8 dBm, another
4 × 1000 at low power, both on a clean 13-level grid); establish the readout noise floor and
its per-channel structure; quantify 15-minute, same-day and 12-hour drift; test whether a
model trained on one input port transfers to another.

**Cannot:**

- **Fit Vπ or φ₀ for any heater.** No single-heater sweep exists. Every point moves all six
  heaters at once, so the six phases are only jointly identified, and the mesh is
  under-determined anyway with 12 of 18 heaters unpowered. `Drift_Report` §3.6 asks for
  exactly this dataset and it has not been taken.
- **Recover absolute optical power.** No laser-power column, and the stated input power differs
  across documents (§8.9).
- **Separate thermal settling from noise.** Fixed 500 ms settle, no step response, no
  time-resolved read except `Experiment_3`'s 1.56 s port cycle.
- **Say anything about 12 of the 18 heaters.** They were held at 0 V for every measurement
  ever taken on this chip.


## 6. The report series

One author, weekly, June → August 2026. The through-line is: simulate the mesh → build the
bench → find out the readout is the limit.

| File | Date | What it adds |
|---|---|---|
| `Thesis_Report_2 (1).pdf` | 19 Jun | Neurophox: Reck vs Clements for 4 modes, 6 MZIs, 12 phases. Adam/TensorFlow fit of a 1×4 uniform splitter, converged in 250 epochs on both meshes; the trained (γ, φ, θ) are printed. Clements = 8-parameter array (4 layers × 2 slots), Reck = 10 (5 × 2). Also a cascaded-unitary power-distribution check and a 3-mode photonic classifier on concentric rings. First DAC bring-up over serial. |
| `Thesis_report_week_3.pdf` | 26 Jun | Near-duplicate of the above with tightened wording; adds the next-task statement: *phase→voltage mapping is the missing calibration*. |
| `Week_4_Report.pdf` | 3 Jul | First hardware measurement: the 4×4 input→output power table (§3.1), loss per input, normalised output split. Notes that the split depends on the input port and proposes the fixed-DAC-across-ports comparison and the repeatability test — which become the drift experiments. |
| `PD_TIA_circuit.pdf` | 3 Jul | The receiver design and its noise budget (§2.3), with LTspice AC and noise sweeps. |
| `SNR_Analysis (1).pdf` | 3 Jul | SNR at both ends of the operating range, the 837 pW figure, and Ananya's per-path heater voltages (§4). Explicitly flags uncertainty about how to separate coupling loss from propagation loss. |
| `Loss_Analysis (1).pdf` | 17 Jul | The five-line link budget and the −38.17 dBm requirement (§3.2). Announces the drift campaign. |
| `Setup_Analysis.pdf` | 30 Jul | The full bench description: PD-TIA, DAC→heater table, NTC/β/PID derivation, switch wiring. Five-row port-4 sample dataset. |
| `MZI Presentation.pptx (2).pdf` | 31 Jul | Slide version; PD-TIA calibration curve; states the readout-floor problem and its two fixes; compares 0.84 dBm and 2 dBm input. |
| `Drift_Report.pdf` | 22 Aug | The three drift experiments (§3.3) and the request for a swept-voltage dataset. |
| `PIC_Design_Review.pptx` | 24 Aug (deck dated 2023) | The Quanfluence tapeout review (§1). |

Two things the reports flag as open and never close: the phase→voltage calibration, and a
principled method for separating fibre-coupling loss from on-chip loss.


## 7. External references

- **`Accurate Self-Configuration of Rectangular Multiport Interferometers`** — Hamerly,
  Bandyopadhyay, Englund, arXiv:2106.03249v2. A configuration algorithm for the *Clements*
  geometry (earlier self-configuring methods only worked on triangular meshes), robust to
  fabrication error, requiring **no prior characterisation** and no internal power monitors —
  it zeroes out the target matrix diagonal-by-diagonal from the corners. Directly the method
  this chip wants, and it would bypass the missing per-heater calibration entirely.
  **Caveat before anyone builds on it:** it needs *coherent* external detectors. This bench has
  four intensity-only photodiodes and no LO path into the 4x4 block (the two `LO` facets, R15
  and R30, feed the 1D-cluster and QRNG blocks). Adapting it to intensity-only readout is the
  work, not a detail.
- **`Photonic matrix multiplication lights up photonic accelerator and beyond`** — Zhou et al.,
  *Light: Science & Applications* 11:30 (2022). Review, not a method: the three families of
  optical matrix multiplication (plane-light-conversion, MZI mesh, WDM), their milestones and
  their applications. Background and citation source for the mesh-method framing.


## 8. Open questions and contradictions

1. **Which UH is which mesh phase — unknown, and it blocks everything.** The design review
   names six routing heaters H1–H6, one per MZI; the packaging exposes eighteen pads UH1–UH18;
   the bench wires six of them, {UH3, UH4, UH8, UH11, UH12, UH15}. The arithmetic (18 = 6 MZIs
   × 3, and the single test MZI gets MH1–MH3) supports three heaters per MZI, but nothing here
   assigns a UH to an (MZI, θ/φ) role, and the GND-pad grouping in the pad table does not
   partition cleanly into six triples. Consequence: `pic_data/heater_map.csv` and
   `theory/layout.py` number heaters H0–H17 by role, and `pic/config.py`'s
   `WIRED_DACS = (0..5)` means *model heaters* 0–5, which is **not** the bench's DAC 0–5. The
   two numberings must not be joined until the GDS or a per-heater sweep settles it.
2. **UH4 vs UH7 on DAC channel 4 — resolved to UH4, on pin evidence.** `Setup_Analysis`
   Table 1 and `Drift_Report` Table 1 (the same table, printed twice) name the heater `UH7` but
   give pad `BP9`; the vendor table assigns `BP9` to **UH4** and `BP12` to UH7.
   `Setup_Analysis` §1.5 independently lists the active heaters as "UH3, **UH4**, UH8, UH11,
   UH12, UH15". The pin column of that table is otherwise exact (TP9→UH3, BP13→UH8,
   BP16→UH11, BP17→UH12) and its one other error is a transcription slip in the same
   direction — `BP24` for UH15, where no BP24 heater pad exists and `TP24` is UH15. Reading
   the pads as authoritative and the names as looked-up gives **UH4**. Confirm with a
   continuity check before anything depends on it.
3. **The bench table's GND column disagrees with the vendor's per-heater GND pads** (it lists
   BP1 for UH8/UH11/UH12 where the vendor says BP15/BP20/BP20). Harmless: those are all
   commoned bottom-rail pads. Less clearly harmless is that it puts UH3 and UH15 across the
   *top* rail (TP26) where the vendor puts UH3 on the bottom rail (BP1) — that is a
   rail-crossing, not a relabelling, and it is worth a meter.
4. **NEP is stated two different ways, √2 apart.** `PD_TIA_circuit.pdf` derives
   **2.18 pW/√Hz** from its own noise budget (1.961 pA/√Hz ÷ 0.9 A/W). `SNR_Analysis` and
   `Loss_Analysis` both use **3.08 pW/√Hz** with no derivation. 2.179 × √2 = 3.08, so one of
   them has an unexplained √2. Using the TIA report's own number moves the budgeted floor from
   −62.27 to −63.5 dBm. Immaterial in practice (see item 6) but it should not stay unexplained.
5. **"4.8 mV ADC step" vs the 0.98 mV grid in the data.** `Drift_Report` §2.3 compares the
   measured spreads to a 4.8 mV step. That is the raw LSB; `Setup.ino` averages five raw counts
   before scaling, so the effective step is 0.978 mV, which is what the spreadsheets show.
   Not a contradiction once the sketch is read, but it changes the report's conclusion: the
   15-minute spreads are 3–34 LSB, not 0.6–7.
6. **The link budget is ~33 dB optimistic because it ignores the ADC.** The budgeted floor of
   −62.27 dBm corresponds to 2.3 µV at the TIA output. The real floor is the 10-bit Arduino ADC
   plus reading-to-reading spread, ≈ **−29 dBm per photodiode** (§3.2, §3.4). Every conclusion
   drawn from "837 pW is enough" needs re-deriving. A higher-resolution ADC, a larger `Rf`, or
   both, moves the floor much more cheaply than laser power does — the TIA's 37 kHz bandwidth
   is three orders of magnitude wider than the 100 Hz sampling actually used, so there is a lot
   of gain-bandwidth to trade away. The TIA report notices this and stops short of the
   conclusion.
7. **"The afternoon run is noisier" is contradicted by its own data.** `Drift_Report` §3.5
   attributes a higher afternoon noise floor to higher ambient temperature. Recomputing σ per
   (port, PD) from `Drift_data.xlsx` `Experiment_3`: the afternoon run is **quieter on 15 of
   16 channels** (mean σ 7.37 mV vs 8.44 mV), with port 3 the most improved. The morning↔
   afternoon *offset* in Table 4 is real; the noise-floor claim is not.
8. **The simulated routing table and the measured heater settings disagree on H3.** The design
   review holds H3 at 0 across all sixteen paths; Ananya's measured max-power settings put H3
   between 0.25 V and 3.0 V in all eight paths she measured. Possibly just a units mismatch
   (binary "cross/bar" vs volts, with a real Vπ nobody has measured), possibly the simulation
   used a different MZI convention. Unresolved, and it matters because the deck's table is the
   only thing that says the mesh routes as designed.
9. **Input power is stated six different ways and appears in no dataset.** −1.71 dBm
   (`Setup_Analysis` §1.5), 0.84 dBm and 2 dBm (`MZI Presentation`), 0.96 dBm (`Week_4`),
   1 dBm assumed (`Loss_Analysis`), 8 dBm (`PD_Drift` Exp 3). None of the spreadsheets carries
   a power column. Worse, `Setup_Analysis` Table 2's PD readings (0.06–0.49 V, summing to
   1.9 V ≈ 480 µW ≈ −3.2 dBm collected) are irreconcilable with a −1.71 dBm input and the
   ~10 dB loss measured in §3.1 — one of the two numbers in that report is wrong. Treat every
   dataset as relative-only until a run is taken with the power logged.
10. **The budget's 12 dB "MZI insertion loss" exceeds the measured chip loss.** §3.1 measures
    10.08 / 9.98 / 11.33 dB *total* for inputs 1, 2 and 4 — and that already includes both
    fibre couplings, which the budget adds a further 6 dB for. Only input 3 (16.78 dB, the bad
    facet) approaches the budgeted 12 + 6. The budget is conservative by ~8 dB on three of the
    four inputs.
11. **500 ms of settling is probably not enough.** The 6x6 rig measured τ ≈ 0.7–1.2 s and
    t99 at 3–5 s for its thermo-optic heaters, with a slow substrate tail on top. Every
    dataset here uses `HEATER_DELAY_MS = 500`, so readings retain memory of the previous
    combination. Different process and different heater module (Ligentec P1+), so the number
    does not transfer directly — but it has never been measured on this chip, and a step
    response is cheap.
12. **The switch's optical band is persistent flash state that nobody has recorded setting.**
    `BAND`/`DBAND` select O, C or L; the wrong setting is a silent insertion-loss penalty that
    would land in the "coupling loss" bucket. Read it back with `BAND` before the next
    calibration run.
13. **`ILIMN` is set to −5.8 A where the comment says 1 A**, and `UV_CLAMP` limits cooling to
    −1.26 V where the constant is named `VOLT_CLAMP_0V` and the PID is allowed to ask for
    −2 V (§2.5). Neither has caused a visible problem — the TEC evidently holds 25 °C — but the
    negative-direction protection is not doing what the sketch says it is.
14. **The TEC loop is proportional-only** (`Kp=5, Ki=0, Kd=0`) with a hard `+1.3 °C` offset on
    the thermistor reading. It will hold a load-dependent steady-state error, which is exactly
    the kind of thing that shows up later as a "time-of-day drift" (§3.3, Experiment 3). No
    independent temperature log exists to check it against.
