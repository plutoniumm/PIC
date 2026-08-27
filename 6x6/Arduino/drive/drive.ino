#include <SPI.h>

// Thin protocol layer: one line in, one line out. No menus, no averaging.
//   in : comma-separated DAC voltages "v0,v1,...\n"  (sets channels 0..n-1)
//   out: comma-separated raw ADC voltages "a0,...,a13\n"  (single read, no averaging)
// Build the loops / averaging / library on top in Python.

const int CS[4] = {5, 4, 3, 2};   // DAC chip-selects: chip k drives channels 16k..16k+15
const int NUM_DAC = 64;           // 4 chips x 16 channels
const int NUM_ADC = 14;           // A0..A13

void setup() {
  SPI.beginTransaction(SPISettings(10000000, MSBFIRST, SPI_MODE1));
  SPI.begin();
  Serial.begin(115200);
  for (int i = 0; i < 4; i++) {
    pinMode(CS[i], OUTPUT);
    digitalWrite(CS[i], HIGH);
  }
  Serial.println("drive ready");
}

void loop() {
  if (!Serial.available()) return;
  String line = Serial.readStringUntil('\n');
  line.trim();
  if (line.length() == 0) return;

  // Set DAC channels from the values, in order. Extra values past NUM_DAC are ignored.
  int ch = 0, start = 0;
  while (ch < NUM_DAC) {
    int comma = line.indexOf(',', start);
    String tok = (comma == -1) ? line.substring(start) : line.substring(start, comma);
    setDAC(ch++, tok.toFloat());
    if (comma == -1) break;
    start = comma + 1;
  }

  // Read every ADC once and reply.
  for (int i = 0; i < NUM_ADC; i++) {
    float v = analogRead(i) / 1023.0 * 5.0;
    Serial.print(v, 4);
    if (i < NUM_ADC - 1) Serial.print(",");
  }
  Serial.println();
}

void setDAC(int ch, float voltage) {
  if (voltage < 0) voltage = 0;
  if (voltage > 5) voltage = 5;
  unsigned int decimal = (unsigned int)((voltage * 65535.0) / 5.0);
  unsigned int MSB = (decimal >> 8) & 0xFF;
  unsigned int LSB = decimal & 0xFF;
  int cs = CS[ch / 16];
  int channel = ch % 16;

  digitalWrite(cs, LOW);
  SPI.transfer(0x10 | channel);   // write + update channel
  SPI.transfer(MSB);
  SPI.transfer(LSB);
  digitalWrite(cs, HIGH);

  digitalWrite(cs, LOW);          // config / power-up sequence (from pic.ino, proven)
  SPI.transfer(0x03);
  SPI.transfer(0x00);
  SPI.transfer(0x84);
  digitalWrite(cs, HIGH);

  digitalWrite(cs, LOW);
  SPI.transfer(0x09);
  SPI.transfer(0x00);
  SPI.transfer(0x00);
  digitalWrite(cs, HIGH);

  digitalWrite(cs, LOW);
  SPI.transfer(0x05);
  SPI.transfer(0xFF);
  SPI.transfer(0xFF);
  digitalWrite(cs, HIGH);
}
