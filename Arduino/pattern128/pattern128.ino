#include <SPI.h>

// Standalone ELECTRICAL map-check (laser OFF): a UNIQUE linear ramp on one 64-channel
// half, the other half held at 0. Channel n in the active half reads
// 0.05 * (n - RAMP_START + 1) V, so a multimeter reading directly identifies the
// channel: n = RAMP_START + V/0.05 - 1. Re-written every second (self-healing, no host).
// Up to 3.2 V HERE ONLY (electrical test, no light) -- the runtime firmware pic128.ino
// keeps its 0-2 V clamp; REFLASH pic128 before any optical work.

const int NUM_CHIPS = 8;
const int CS[NUM_CHIPS] = {10, 9, 8, 7, 6, 5, 4, 3};  // chip k drives ch 16k..16k+15
const int numDAC = 128;
const int RAMP_START = 64;  // active half: RAMP_START .. RAMP_START+63 (0 or 64)
const float STEP = 0.05;

void setup() {
  Serial.begin(115200);
  for (int k = 0; k < NUM_CHIPS; k++) {
    pinMode(CS[k], OUTPUT);
    digitalWrite(CS[k], HIGH);
  }
  SPI.begin();
  SPI.beginTransaction(SPISettings(10000000, MSBFIRST, SPI_MODE1));
  Serial.print("pattern128 ramp: ch ");
  Serial.print(RAMP_START);
  Serial.print("..");
  Serial.print(RAMP_START + 63);
  Serial.println(" = 0.05*(n-start+1) V, others 0; rewritten every 1 s");
}

void loop() {
  for (int i = 0; i < numDAC; i++) {
    int k = i - RAMP_START;
    setDAC(i, (k >= 0 && k < 64) ? STEP * (k + 1) : 0.0);
  }
  delay(1000);
}

// Value write + config/power-up sequence, every write (drive.ino / pic.ino, proven).
void setDAC(int ch, float voltage) {
  if (voltage < 0.0) voltage = 0.0;
  if (voltage > 3.25) voltage = 3.25;  // test-sketch ceiling; runtime firmware stays 0-2 V
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
