#include <SPI.h>

// 4x4 unitary mesh firmware. One line in ("v0,...,v17\n", short lines padded with 0), one
// line out (NUM_PINS mean ADC voltages). Same protocol as the 6x6 board, so the host driver
// is the same shape; only the widths differ.
//
// Optical input switch: the 1x4 unit sits on Serial1 at 9600 and takes "SET <1..4>". A host
// line beginning with "P" (e.g. "P3\n") selects an input port and replies "PORT 3"; anything
// else is treated as a DAC vector. Selecting a port waits SWITCH_SETTLE_MS for the mirror.
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
const int NUM_DAC = 18;              // heaters UH1..UH18
const float VMAX = 3.0;              // per-channel ceiling
const float DAC_REF = 5.0;           // DAC full-scale reference

const int NUM_PINS = 4;              // one PD-TIA per mesh output, A0..A3
const int AVG_N = 5;                 // full ADC sweeps averaged per reply
const int SWITCH_SETTLE_MS = 1000;

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
    if (port < 1 || port > 4) {
      Serial.println("ERR port");
      return;
    }
    Serial1.print("SET ");
    Serial1.println(port);
    delay(SWITCH_SETTLE_MS);
    Serial.print("PORT ");
    Serial.println(port);
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
  if (voltage < 0.0) voltage = 0.0;
  if (voltage > VMAX) voltage = VMAX;
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

void readAndSendADC() {
  long sum[NUM_PINS] = {0};
  for (int n = 0; n < AVG_N; n++)
    for (int i = 0; i < NUM_PINS; i++) sum[i] += analogRead(i);
  for (int i = 0; i < NUM_PINS; i++) {
    float mean = ((float)sum[i] / AVG_N / 1023.0) * 5.0;
    Serial.print(mean, 5);
    if (i < NUM_PINS - 1) Serial.print(",");
  }
  Serial.println();
}
