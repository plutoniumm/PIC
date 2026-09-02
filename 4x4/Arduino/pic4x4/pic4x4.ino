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
const int CS[NUM_CHIPS] = {10, 9};   // chip k drives channels 16k .. 16k+15
const int NUM_DAC = 16;              // DAC81416 channels, ch0..ch15

// Per-channel ceiling: V = I*R from each heater's own measured resistance at the current
// limit in pic/config.py (HEATER_MAX_MA). Raised from 30 mA to 40 mA deliberately --
// span goes as V^2, so this is 1.78x more phase everywhere, and no document gives an
// absolute maximum to weigh it against. Keep this table and pic.config in step. A single
// global 3.0 V is WRONG and unsafe: the 60R group draws 52 mA at 3 V. Index is DAC channel.
// H18/H14/H6 (ch 6, 8, 11) have no confirmed resistance, so they take the SMALLEST
// staged 1.5 V, not 0.0 and not their group's ceiling. 1.5 V draws 26.6 mA even against the
// smallest resistance on this board, so it is safe under every hypothesis, and it is enough
// for pic.resistance to weigh the channel against a known one using the TEC as a
// calorimeter. Raise to the real I*R ceiling only after that measurement -- 3 V would draw
// 53 mA if the channel turned out to be 56 ohm. Holding them at 0.0 was self-defeating: a
// dark channel is never swept, so never characterized, so it stays dark.
//        ch:   0    1    2    3    4    5    6    7    8    9   10   11   12   13   14   15
// heater:    H15  H12  H11   H8   H4   H3  H18   H9  H14  H13  H10   H6   H7   H5   H2   H1
const float VMAX[NUM_DAC] = {
           4.55, 2.45, 4.75, 4.55, 2.30, 4.50, 1.50, 2.25, 1.50, 4.55, 2.25, 1.50, 4.60, 4.65, 4.65, 4.55
};
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

void setup() {
  Serial.begin(115200);
  Serial1.begin(9600);               // 1x4 optical input switch

  for (int k = 0; k < NUM_CHIPS; k++) {
    pinMode(CS[k], OUTPUT);
    digitalWrite(CS[k], HIGH);
  }
  SPI.begin();
  SPI.beginTransaction(SPISettings(10000000, MSBFIRST, SPI_MODE1));

  analogReference(INTERNAL2V56);
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
    Serial.print(" switchms="); Serial.println(SWITCH_SETTLE_MS);
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
  SPI.transfer(0x10 | channel);            // write + update channel
  SPI.transfer((code >> 8) & 0xFF);
  SPI.transfer(code & 0xFF);
  digitalWrite(cs, HIGH);

  digitalWrite(cs, LOW);                   // power-up / config, every write
  SPI.transfer(0x03); SPI.transfer(0x00); SPI.transfer(0x84);
  digitalWrite(cs, HIGH);

  digitalWrite(cs, LOW);
  SPI.transfer(0x09); SPI.transfer(0x00); SPI.transfer(0x00);
  digitalWrite(cs, HIGH);

  digitalWrite(cs, LOW);
  SPI.transfer(0x05); SPI.transfer(0xFF); SPI.transfer(0xFF);
  digitalWrite(cs, HIGH);
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
  unsigned long acc[NUM_PINS] = {0};
  accumulate(acc, 1);
  for (int i = 0; i < NUM_PINS; i++) {
    float mean = ((float)acc[i] / AVG_N / 1023.0) * ADC_REF_V;
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
      Serial.print(((float)acc[k][i] / (frames * AVG_N) / 1023.0) * ADC_REF_V, 5);
    }
  Serial.println();
}
