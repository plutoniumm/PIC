#include <SPI.h>

// 4x4 unitary mesh firmware. One line in ("v0,...,v17\n", short lines padded with 0), one
// line out (NUM_PINS mean ADC voltages). Same protocol as the 6x6 board, so the host driver
// is the same shape; only the widths differ.
//
// Optical input switch: the 1x4 unit sits on Serial1 at 9600 and takes "SET <1..4>". A host
// line beginning with "P" (e.g. "P3\n") selects an input port and replies "PORT 3"; anything
// else is treated as a DAC vector. Selecting a port waits SWITCH_SETTLE_MS for the mirror.
//
// "S<cycles>,<reads>" runs a whole four-port sweep here and answers once -- see readSweep.
// "C" reports what this firmware can do, so a host can find out without guessing.
//
// These numbers must match pic/config.py: NUM_DAC == NUM_DAC there, NUM_PINS == NUM_ADC_RAW,
// VMAX == FIRMWARE_VMAX. A host expecting 4 values will hang forever against a firmware
// sending 8.
//
// PROVISIONAL: the bring-up board (mrunal/Setup.ino) wires one DAC chip and drives only 6
// of the 18 heaters. NUM_DAC is written for the full mesh; channels beyond what is physically
// connected are set and go nowhere, which is harmless and keeps the host width fixed.
//
// The value-write-plus-config sequence in setDAC runs on every write, not once at boot. The
// one-time init on the first 6x6 sketch does not bring these chips up: a whole
// characterization run came back with no real DAC drive before that was found. Do not
// "optimise" it back into setup().

const int NUM_CHIPS = 2;
// CS is pin 7 on this board, confirmed at the bench 2026-09-22. It is NOT the 10 that
// `mrunal/Setup.ino` and `Arduino/pin_check/pin_check.ino` both use -- those describe the
// original bring-up wiring. Driving 10 leaves the DAC unselected, so it ignores every write
// and holds every output at 0 V: SPI looks perfectly healthy (there is no readback), the
// clamp table reads back correctly, and the analog rail draws 0.00 A because nothing is
// sourcing. That is a whole afternoon of "the heaters do not modulate light".
const int CS[NUM_CHIPS] = {7, 9};    // chip k drives channels 16k .. 16k+15
const int NUM_DAC = 16;              // DAC81416 channels, ch0..ch15

// Per-channel ceiling: V = I*n*R from each heater's own measured resistance at the current
// limit in pic/config.py (HEATER_MAX_MA = 40), where n is how many DAC channels are bonded
// onto that heater. Keep this table and pic.config.VOLTAGE_MAX_CH in step element for
// element -- `pic.rig.assert_firmware_vmax` reads it back over the `V` query on every open
// and refuses on any disagreement. A single global 3.0 V is WRONG and unsafe: the 57R group
// draws 52 mA at 3 V on one channel. Index is DAC channel.
//
// Rewired 2026-09-22 (mrunal/board_firmware_v2.md): sixteen channels, THIRTEEN heaters.
// ch2+3, ch4+5 and ch6+7 are each shorted onto one heater, which is why those six entries
// are equal in pairs and why they are the high ones -- a bonded pair sources 80 mA into
// 57 ohm, and that is what finally takes H4 past 2 pi. H18, H14 and H6 no longer have a
// driver and are gone from this table entirely.
//
// 50 mA since 2026-09-22, but every 114-119 ohm channel lands on 4.95 and not the 5.7-5.9
// that 50 mA would want: the DAC cannot exceed DAC_REF, and `setDAC` would not clamp it --
// code = V * 65535 / DAC_REF overflows a 16-bit unsigned and wraps to a LOW voltage with no
// error. So those sit near 44 mA. The bonded pairs do reach 50 mA per channel, half the
// resistance needing half the volts.
//
// ch10 (H12) is the one conservative entry. Its resistance is disputed -- 62.3 ohm here and
// in every prior bench table, 116.8 in the v2 sweep firmware, which is also that firmware's
// value for H2. The low reading stands: guessing high would put 75 mA through it.
//        ch:   0    1    2    3    4    5    6    7    8    9   10   11   12   13   14   15
// heater:     H1   H2  H10  H10   H9   H9   H4   H4  H15  H13  H12  H11   H7   H8   H5   H3
const float VMAX[NUM_DAC] = {
           4.95, 4.95, 4.95, 4.95, 4.95, 4.95, 4.95, 4.95, 4.95, 4.95, 3.10, 4.95, 4.95, 4.95, 4.95, 4.95
};

// Channels shorted together on the board. Held here as well as on the host because this is
// the backstop: two DAC81416 output stages commanded to different voltages drive current
// into each other, and the host is not the only thing that can write this board.
const int NUM_PAIRS = 3;
const int PAIRS[NUM_PAIRS][2] = { {2, 3}, {4, 5}, {6, 7} };

const float DAC_REF = 5.0;           // DAC full-scale reference

// The ADC ran against AVCC (5.0 V) and the photodiodes never come near it: over the
// 155-state table at 25 C the brightest read of any detector is 0.468 V and the MEDIAN entry
// is 0.056 V, so 9.4% of full scale was in use and 6.6 of 10 bits. Quantisation is
// LSB/sqrt(12) = 1.41 mV, which after AVG_N averaging is negligible against a BRIGHT entry --
// which is how it came to be dismissed -- but a transfer matrix is mostly dim entries, and at
// the median one it is 0.0063 relative against a 0.0035 structured noise floor. 64% of the
// table's entries were quantisation-limited.
//
// 2.56 V and not 1.1 V because the laser is going to +12 dBm: 4 dB is 2.5x, which takes the
// brightest read from 0.468 V to about 1.17 V and would clip a 1.1 V reference outright. At
// 2.56 V that is 46% of full scale against 9.4% at AVCC. Note the arithmetic: raising the
// power and raising the reference cancel in quantisation terms (signal/LSB goes 437 -> 468),
// so +12 dBm is worth doing only if the dominant noise is ADDITIVE -- detector and TIA --
// rather than multiplicative laser RIN, which does not improve with power. The nominal 1.1 V bandgap is
// only +-10% part to part, so ADC_REF_V is a scale that must be trimmed against a known
// voltage before absolute readings mean anything -- every number this rig uses is a RATIO
// within one sweep (to_transfer divides each column by its own sum), so a scale error cancels.
// Saturation would not: `sat` reports it rather than silently clipping.
const float ADC_REF_V = 2.56;
const int NUM_PINS = 4;              // one PD-TIA per mesh output, A0..A3

// Autoranging. The photodiodes use a few percent of a 2.56 V span -- the brightest output
// measured 158 mV at +5 dBm, which is 63 of 1023 counts, so six of ten bits were being
// thrown away. The Mega's 1.1 V bandgap is the only lower reference it has and buys 2.33x,
// a bit and a bit. The frame still goes out in VOLTS, converted against whichever reference
// was actually used, so the host protocol does not change and nothing upstream has to know.
//
// Slack on both edges: step up well before clipping, step down only when the reading would
// still sit comfortably inside the smaller span. Both references are +-10% part to part, so
// these are nominal scales and every quantity this rig uses is a ratio within one sweep.
const float ADC_REFS[] = {1.1, 2.56, 5.0};
const uint8_t ADC_MODES[] = {INTERNAL1V1, INTERNAL2V56, DEFAULT};
const int N_ADC_REFS = 3;
const int RANGE_UP = 950;     // any channel above this: the span is too small
const int RANGE_DOWN = 330;   // every channel below this: the next span down still fits
int adcRef = 1;               // start where the old fixed build sat, at 2.56 V

void setAdcRef(int idx) {
  if (idx < 0) idx = 0;
  if (idx >= N_ADC_REFS) idx = N_ADC_REFS - 1;
  if (idx == adcRef) return;
  adcRef = idx;
  analogReference(ADC_MODES[adcRef]);
  for (int n = 0; n < 8; n++) analogRead(0);   // the reference needs settling reads
}

// Pick a range from one cheap look, before the averaged read that is actually reported.
void autorange() {
  for (int pass = 0; pass < N_ADC_REFS; pass++) {
    int peak = 0;
    for (int i = 0; i < NUM_PINS; i++) {
      int v = analogRead(i);
      if (v > peak) peak = v;
    }
    if (peak > RANGE_UP && adcRef < N_ADC_REFS - 1) { setAdcRef(adcRef + 1); continue; }
    if (peak < RANGE_DOWN && adcRef > 0) { setAdcRef(adcRef - 1); continue; }
    return;
  }
}

const int AVG_N = 16;                // full ADC sweeps averaged per reply. 16 not 5: PD0's
                                     // read noise is ~170x the other detectors', and averaging
                                     // is the only lever that costs nothing but time.
// 1000 ms was inherited from mrunal/Setup.ino, which sent "SET n" and never read the
// reply -- so the delay was standing in for a handshake rather than for mirror settling.
// The Sercalo answers every command (38-1180 rev 1.19 SS2.4) and must not be sent another
// before it does. Reading the reply IS the handshake; what remains is a real settling
// margin for a MEMS mirror, which moves in milliseconds. At 8 switch moves per state the
// old value was 62% of a 21-minute table capture and it made deep averaging unaffordable.
// 150 ms and not 20: at 20 ms the first read after a port change came back LOW on two of
// four rails, by 0.6% and 1.4%, consistently signed -- light still rising, not scatter. The
// transient is real, the second it was given was not.
const int SWITCH_SETTLE_MS = 150;
const unsigned long SWITCH_REPLY_MS = 250;   // give up waiting; never hang the board

// A four-port sweep is the host's unit of measurement (pic/normalise.py:sweep), and it used
// to cost 4*repeats round trips: 24 at repeats=6. The board's share of one of those is
// AVG_N*NUM_PINS conversions, ~6.7 ms at the default prescaler, against a measured 0.12 s
// per read on the host side -- so 95 percent of a sweep was USB latency and mirror settling
// paid one command at a time. Doing the whole sweep here answers once instead of 24 times.
//
// The two counts are NOT interchangeable and the host must not collapse them.
//   `reads`  frames averaged back to back at one mirror position. ~7 ms apart, so they
//            average detector and ADC noise and nothing slower. Nearly free.
//   `cycles` complete visits to all four ports. Frames at one port are then a full cycle
//            (~0.7 s) apart, which is the spread the host's `repeats` loop used to buy, and
//            it costs a mirror settle per port per cycle.
// Total frames per port is cycles*reads. Measured Allan deviation on this rig falls as
// tau^-1/2 from 6 to 400 s (talk/progress.md), so averaging is valid over the `cycles`
// spread; below ~1 s nothing has been measured, which is why `cycles` stays available.
//
// Caps are wall clock, not arithmetic: the accumulator is 32-bit and 1023*AVG_N per frame
// overflows only past 262,000 frames. SWEEP_MAX_FRAMES bounds the ADC work at 4*64*6.7 ms
// = 1.7 s and SWEEP_MAX_CYCLES the mirror work at 64*0.16 s = 10 s, and a host asking for
// either must extend its own serial timeout to match (pic.acquisition.batch_sweep_seconds
// sizes it).
const int NUM_PORTS = 4;
const int SWEEP_MAX_CYCLES = 16;
const int SWEEP_MAX_READS = 64;
const int SWEEP_MAX_FRAMES = 64;      // cycles*reads, per port

float values[NUM_DAC];
unsigned int lastCode[NUM_DAC];      // what setDAC last wrote, for the `R` readback check

void setup() {
  Serial.begin(115200);
  Serial1.begin(9600);               // 1x4 optical input switch

  for (int k = 0; k < NUM_CHIPS; k++) {
    pinMode(CS[k], OUTPUT);
    digitalWrite(CS[k], HIGH);
  }
  SPI.begin();
  SPI.beginTransaction(SPISettings(10000000, MSBFIRST, SPI_MODE1));

  analogReference(ADC_MODES[adcRef]);
  for (int n = 0; n < 8; n++) analogRead(0);          // the mux/reference needs settling reads
  for (int i = 0; i < NUM_DAC; i++) setDAC(i, 0.0);   // known state on reset
  Serial.println("pic4x4 ready");
}

void loop() {
  if (!Serial.available()) return;

  static char buf[400];
  int n = Serial.readBytesUntil('\n', buf, sizeof(buf) - 1);
  buf[n] = '\0';

  if (buf[0] == 'P' || buf[0] == 'p') {              // "P<1..4>": select an input port
    int port = atoi(buf + 1);
    // 0 is the Sercalo's open channel: the common port routes to nothing, which is the
    // optical dark the photodiode normalisation measures its zero against. Without it the
    // only way to go dark is to turn the laser off, which also moves the PD baselines.
    if (port < 0 || port > 4) {
      Serial.println("ERR port");
      return;
    }
    selectPort(port);
    Serial.print("PORT ");
    Serial.println(port);
    return;
  }

  if (buf[0] == 'Q' || buf[0] == 'q') {              // "Q": ask the switch where it is
    // The mirror's own answer, passed through verbatim. `selectPort` deliberately discards
    // replies, so a switch that is unpowered, unplugged or parked open is indistinguishable
    // from a working one -- and downstream that reads as a dead optical path with no way to
    // tell which half is at fault. POS is the Sercalo's position query (`mrunal/Sercalo
    // Optical Switch.pdf` 10.9); silence here is itself the diagnosis.
    while (Serial1.available()) Serial1.read();
    Serial1.println("POS");
    unsigned long t0 = millis();
    Serial.print("SWITCH ");
    bool any = false;
    while (millis() - t0 < 1000) {
      while (Serial1.available()) { Serial.write(Serial1.read()); any = true; }
    }
    if (!any) Serial.print("(no reply)");
    Serial.println();
    return;
  }

  if (buf[0] == 'C' || buf[0] == 'c') {              // "C": what this firmware can do
    // A board can carry the `V` table and still predate the batched sweep, so the host
    // cannot infer one capability from the other. Asked before anything is driven, so a
    // firmware without this command parses it as a DAC line, zeroes every channel and
    // answers with an ADC frame -- which is the shape the host detects and is harmless
    // exactly there and nowhere else.
    Serial.print("CAP sweep=1 ports="); Serial.print(NUM_PORTS);
    Serial.print(" pins="); Serial.print(NUM_PINS);
    Serial.print(" dac="); Serial.print(NUM_DAC);
    Serial.print(" avg="); Serial.print(AVG_N);
    Serial.print(" maxcycles="); Serial.print(SWEEP_MAX_CYCLES);
    Serial.print(" maxreads="); Serial.print(SWEEP_MAX_READS);
    Serial.print(" maxframes="); Serial.print(SWEEP_MAX_FRAMES);
    Serial.print(" switchms="); Serial.print(SWITCH_SETTLE_MS);
    Serial.print(" rb=1");
    Serial.print(" adcref="); Serial.println(ADC_REFS[adcRef], 2);
    return;
  }

  if (buf[0] == 'S' || buf[0] == 's') {              // "S<cycles>,<reads>": a whole sweep
    int cycles = atoi(buf + 1), reads = 1;
    char* comma = strchr(buf, ',');
    if (comma) reads = atoi(comma + 1);
    if (cycles < 1) cycles = 1;
    if (reads < 1) reads = 1;
    if (cycles > SWEEP_MAX_CYCLES || reads > SWEEP_MAX_READS
        || (long)cycles * reads > SWEEP_MAX_FRAMES) {
      Serial.println("ERR sweep");                   // refuse rather than block for minutes
      return;
    }
    readSweep(cycles, reads);
    return;
  }

  if ((buf[0] == 'R' || buf[0] == 'r') && buf[1] && buf[1] != '\r') {  // "R<hex>": one register
    byte addr = (byte)strtol(buf + 1, NULL, 16);
    Serial.print("REG 0x"); Serial.print(addr, HEX);
    Serial.print(" = 0x"); Serial.println(readReg(CS[0], addr), HEX);
    return;
  }

  if (buf[0] == 'R' || buf[0] == 'r') {              // "R": read the DACs back
    // Writes alone prove nothing: the CS-on-pin-10 fault accepted every write for an
    // afternoon. Reading each channel's data register back over SDO shows the chip is
    // selected, alive and holding what was sent. It cannot see a broken wire past the DAC
    // pin -- an open heater still holds its code -- which needs current sensing.
    for (int k = 0; k < NUM_CHIPS && k * 16 < NUM_DAC; k++) {
      Serial.print("RB chip="); Serial.print(k);
      Serial.print(" id=0x"); Serial.print(readReg(CS[k], 0x01), HEX);
      int ok = 0;
      String bad = "";
      for (int ch = k * 16; ch < NUM_DAC && ch < (k + 1) * 16; ch++) {
        unsigned int got = readReg(CS[k], 0x10 | (ch % 16));
        if (got == lastCode[ch]) ok++;
        else { if (bad.length()) bad += ","; bad += ch; bad += ":"; bad += lastCode[ch]; bad += "/"; bad += got; }
      }
      Serial.print(" ok="); Serial.print(ok); Serial.print("/"); Serial.print(min(16, NUM_DAC - k * 16));
      Serial.print(" bad="); Serial.println(bad);
    }
    return;
  }

  if (buf[0] == 'V' || buf[0] == 'v') {              // "V": report the clamp table
    // The host has its own copy in pic.config.VOLTAGE_MAX_CH and the two silently diverging
    // is the documented failure of this rig -- a host asking for volts the firmware clamps
    // to something else produces a measurement of a state nobody chose. One line makes the
    // duplication checkable instead of a convention.
    Serial.print("VMAX");
    for (int i = 0; i < NUM_DAC; i++) { Serial.print(' '); Serial.print(VMAX[i], 2); }
    Serial.println();
    return;
  }

  int idx = 0;
  for (char* tok = strtok(buf, ","); tok && idx < NUM_DAC; tok = strtok(NULL, ","))
    values[idx++] = atof(tok);
  for (; idx < NUM_DAC; idx++) values[idx] = 0.0;    // pad a short line

  // Mirror before driving: a bonded pair must reach the DACs as one voltage. The host
  // refuses a mismatch outright; here the lower channel wins, because a firmware that
  // rejects a line has no way to tell the caller what it did with the heater.
  for (int p = 0; p < NUM_PAIRS; p++) values[PAIRS[p][1]] = values[PAIRS[p][0]];

  for (int i = 0; i < NUM_DAC; i++) setDAC(i, values[i]);
  readAndSendADC();
}

void setDAC(int ch, float voltage) {
  if (ch < 0 || ch >= NUM_DAC) return;
  if (voltage < 0.0) voltage = 0.0;
  if (voltage > VMAX[ch]) voltage = VMAX[ch];
  unsigned int code = (unsigned int)((voltage * 65535.0) / DAC_REF);
  int cs = CS[ch / 16];
  int channel = ch % 16;

  digitalWrite(cs, LOW);
  // Config BEFORE the value, and no 0x05. Both matter, and the old order is why this board
  // accepted every write and drove nothing.
  //
  // 0x05 is SYNCCONFIG: a set bit puts that channel in synchronous update mode, where DAC
  // data lands in the register and does NOT reach the pin until an LDAC trigger. This wrote
  // 0xFFFF -- all sixteen channels -- after every single value, and nothing here ever
  // issues LDAC. The vendor's `mrunal/Setup.ino` never touches 0x05 at all.
  //
  // The remaining two stay per-write rather than once in setup(), which is this project's
  // documented invariant, but they now precede the value the way `configureDAC` does.
  digitalWrite(cs, LOW);                   // power-up / config
  SPI.transfer(0x03); SPI.transfer(0x00); SPI.transfer(0x84);
  digitalWrite(cs, HIGH);

  digitalWrite(cs, LOW);
  SPI.transfer(0x09); SPI.transfer(0x00); SPI.transfer(0x00);
  digitalWrite(cs, HIGH);

  digitalWrite(cs, LOW);
  SPI.transfer(0x10 | channel);            // write + update channel
  SPI.transfer((code >> 8) & 0xFF);
  SPI.transfer(code & 0xFF);
  digitalWrite(cs, HIGH);
  lastCode[ch] = code;
}

unsigned int readReg(int cs, byte addr) {
  // DAC81416 read: frame 1 sends the address with the read bit, frame 2 (a NOP) clocks
  // the register out on SDO. SDO is enabled by SPICONFIG 0x0084, which setDAC writes.
  digitalWrite(cs, LOW);
  SPI.transfer(0x80 | addr); SPI.transfer(0x00); SPI.transfer(0x00);
  digitalWrite(cs, HIGH);
  digitalWrite(cs, LOW);
  SPI.transfer(0x00);
  unsigned int hi = SPI.transfer(0x00);
  unsigned int lo = SPI.transfer(0x00);
  digitalWrite(cs, HIGH);
  return (hi << 8) | lo;
}

void selectPort(int port) {
  while (Serial1.available()) Serial1.read();        // stale reply from a previous command
  Serial1.print("SET ");
  Serial1.println(port);
  unsigned long t0 = millis();
  while (!Serial1.available() && millis() - t0 < SWITCH_REPLY_MS) { }
  while (Serial1.available()) Serial1.read();
  delay(SWITCH_SETTLE_MS);
}

void accumulate(unsigned long* acc, int frames) {
  for (int n = 0; n < frames * AVG_N; n++)
    for (int i = 0; i < NUM_PINS; i++) acc[i] += analogRead(i);
}

void readAndSendADC() {
  autorange();
  unsigned long acc[NUM_PINS] = {0};
  accumulate(acc, 1);
  for (int i = 0; i < NUM_PINS; i++) {
    float mean = ((float)acc[i] / AVG_N / 1023.0) * ADC_REFS[adcRef];
    Serial.print(mean, 5);
    if (i < NUM_PINS - 1) Serial.print(",");
  }
  Serial.println();
}

void readSweep(int cycles, int reads) {
  // The DACs are NOT touched: the host owns the heater state and its thermal settle,
  // because only the host knows whether the state actually changed. Same contract as
  // pic/normalise.py:sweep, which this replaces round trip for round trip.
  unsigned long acc[NUM_PORTS][NUM_PINS];
  for (int k = 0; k < NUM_PORTS; k++)
    for (int i = 0; i < NUM_PINS; i++) acc[k][i] = 0;

  for (int c = 0; c < cycles; c++)
    for (int k = 0; k < NUM_PORTS; k++) {
      selectPort(k + 1);                             // the Sercalo numbers its channels from 1
      accumulate(acc[k], reads);
    }

  // Port-major, matching how a session file stores a sweep (raw[port][pd]) and the order
  // the mirror visits. One line so the host reads it with one readline().
  long frames = (long)cycles * reads;
  Serial.print("SWEEP");
  for (int k = 0; k < NUM_PORTS; k++)
    for (int i = 0; i < NUM_PINS; i++) {
      Serial.print(k == 0 && i == 0 ? ' ' : ',');
      Serial.print(((float)acc[k][i] / (frames * AVG_N) / 1023.0) * ADC_REFS[adcRef], 5);
    }
  Serial.println();
}
