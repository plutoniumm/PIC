#include <SPI.h>

// 128-channel protocol firmware: one line in ("v0,v1,...\n", up to 128 values,
// short lines padded with 0), one line out (14 mean ADC voltages).
// DAC programming = drive.ino's PROVEN per-write sequence (value write followed by the
// 0x03/0x09/0x05 config commands every time -- the one-time init from
// Fully_automated_PIC.ino does NOT bring these chips up; verified 2026-07-23 when a
// census on that init produced no real DAC drive).
// CS pins per Anagha's 128-ch wiring (input_sequence_3_anagha_128.ino).

const int NUM_CHIPS = 8;
const int CS[NUM_CHIPS] = {10, 9, 8, 7, 6, 5, 4, 3};  // chip k drives ch 16k..16k+15
const int numDAC = 128;

const int numPins = 14;  // ADC pins A0..A13
const int AVG_N = 10;    // full ADC sweeps averaged per reply (on-chip noise floor)

float values[numDAC];

void setup() {
  Serial.begin(115200);
  for (int k = 0; k < NUM_CHIPS; k++) {
    pinMode(CS[k], OUTPUT);
    digitalWrite(CS[k], HIGH);
  }
  SPI.begin();
  SPI.beginTransaction(SPISettings(10000000, MSBFIRST, SPI_MODE1));
  Serial.println("pic128 ready");
}

void loop() {
  if (Serial.available()) {
    static char buf[1200];
    int n = Serial.readBytesUntil('\n', buf, sizeof(buf) - 1);
    buf[n] = '\0';
    int idx = 0;
    for (char* tok = strtok(buf, ","); tok && idx < numDAC; tok = strtok(NULL, ","))
      values[idx++] = atof(tok);
    for (; idx < numDAC; idx++) values[idx] = 0.0;  // pad short input

    for (int i = 0; i < numDAC; i++) setDAC(i, values[i]);
    readAndSendADCValues();
  }
}

// Value write + config/power-up sequence, every write (drive.ino / pic.ino, proven).
// 0-2 V safety clamp on a 0-5 V reference scale.
void setDAC(int ch, float voltage) {
  if (voltage < 0.0) voltage = 0.0;
  if (voltage > 2.0) voltage = 2.0;
  unsigned int decimal = (unsigned int)((voltage * 65535.0) / 5.0);
  int cs = CS[ch / 16];
  int channel = ch % 16;

  digitalWrite(cs, LOW);
  SPI.transfer(0x10 | channel);  // write + update channel
  SPI.transfer((decimal >> 8) & 0xFF);
  SPI.transfer(decimal & 0xFF);
  digitalWrite(cs, HIGH);

  digitalWrite(cs, LOW);
  SPI.transfer(0x03); SPI.transfer(0x00); SPI.transfer(0x84);
  digitalWrite(cs, HIGH);

  digitalWrite(cs, LOW);
  SPI.transfer(0x09); SPI.transfer(0x00); SPI.transfer(0x00);
  digitalWrite(cs, HIGH);

  digitalWrite(cs, LOW);
  SPI.transfer(0x05); SPI.transfer(0xFF); SPI.transfer(0xFF);
  digitalWrite(cs, HIGH);
}

// Average AVG_N full sweeps of the 14 ADC pins, reply comma-separated volts.
void readAndSendADCValues() {
  long adcSum[numPins] = {0};
  for (int n = 0; n < AVG_N; n++)
    for (int i = 0; i < numPins; i++) adcSum[i] += analogRead(i);
  for (int i = 0; i < numPins; i++) {
    float meanVoltage = ((float)adcSum[i] / AVG_N / 1023.0) * 5.0;
    Serial.print(meanVoltage, 4);
    if (i < numPins - 1) Serial.print(",");
  }
  Serial.println();
}
