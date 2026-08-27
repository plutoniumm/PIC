#include <SPI.h>

/* =====================================================================
   13-HEATER RANDOM COMBINATION EXPERIMENT
   ---------------------------------------------------------------------
   - Only run this AFTER 1_pin_connection_check.ino has confirmed every
     active channel drives the pin you expect.
   - H6, H14, H18 (DAC channels 11, 8, 6) are EXCLUDED - resistance was
     never confirmed for them. They are forced to 0V once at startup
     and never touched by the sweep. Only the 13 heaters with known
     resistance are driven/randomized/logged.
   ===================================================================== */

/* ================= USER PARAMETERS ================= */
#define NUM_DAC_CH        16   // physical channel count on the DAC81416
#define NUM_COMB          5
#define NUM_PD            4

#define SWITCH_DELAY_MS   1000
#define HEATER_DELAY_MS   500
#define PD_SAMPLE_DELAY   10
#define PD_AVG_COUNT      5

/* ================= HARDWARE ================= */
#define CS_DAC            10
#define VREF_DAC          5.0

/* ================= GLOBALS ================= */
float dacVoltages[NUM_DAC_CH];
bool runFlag = false;

/* ================= PER-CHANNEL VOLTAGE LIMITS =================
   Index = DAC channel. Value = max safe voltage for that channel,
   set from the heater's resistance category (~120R -> 3.0V,
   ~60R -> 1.5V, keeping current at ~25mA vs the 30mA rating).
   Order/heater identity matches the mapping decoded from your notes.
   Channels 6, 8, 11 (H18, H14, H6) are unused - value irrelevant,
   they are never driven above 0V (see activeChannels[] below).
   ================================================================= */
const char* heaterName[NUM_DAC_CH] = {
  "H15","H12","H11","H8","H4","H3","H18","H9",
  "H14","H13","H10","H6","H7","H5","H2","H1"
};

float maxVolt[NUM_DAC_CH] = {
  3.0,  // ch0  H15  114.1R
  1.5,  // ch1  H12  62.3R
  3.0,  // ch2  H11  118.8R
  3.0,  // ch3  H8   114.1R
  1.5,  // ch4  H4   57.8R
  3.0,  // ch5  H3   113.5R
  0.0,  // ch6  H18  EXCLUDED
  1.5,  // ch7  H9   57.2R
  0.0,  // ch8  H14  EXCLUDED
  3.0,  // ch9  H13  114.9R
  1.5,  // ch10 H10  56.4R
  0.0,  // ch11 H6   EXCLUDED
  3.0,  // ch12 H7   116.1R
  3.0,  // ch13 H5   117.1R
  3.0,  // ch14 H2   116.8R
  3.0,  // ch15 H1   114.1R
};

/* Only these 13 channels are driven, randomized, and logged.
   H6 (11), H14 (8), H18 (6) are left out entirely. */
const int activeChannels[] = {0, 1, 2, 3, 4, 5, 7, 9, 10, 12, 13, 14, 15};
const int NUM_ACTIVE = sizeof(activeChannels) / sizeof(activeChannels[0]); // 13

/* ================= SETUP ================= */
void setup() {

  Serial.begin(115200);
  Serial1.begin(9600);

  pinMode(CS_DAC, OUTPUT);
  digitalWrite(CS_DAC, HIGH);

  SPI.begin();
  SPI.beginTransaction(SPISettings(10000000, MSBFIRST, SPI_MODE1));

  configureDAC();

  // Force every channel to 0V at boot, including the 3 excluded ones,
  // then leave the excluded ones alone for the rest of the run.
  for (int i = 0; i < NUM_DAC_CH; i++) setDAC(i, 0.0);

  randomSeed(analogRead(A7));

  Serial.println(F("READY"));
  Serial.println(F("Running 13 heaters. H6, H14, H18 excluded (held at 0V)."));
  Serial.println(F("Send START"));

  // Print the limits being used so a bad table edit is obvious immediately
  for (int i = 0; i < NUM_ACTIVE; i++) {
    int ch = activeChannels[i];
    Serial.print(F("ch")); Serial.print(ch);
    Serial.print(F(" -> ")); Serial.print(heaterName[ch]);
    Serial.print(F(" max=")); Serial.print(maxVolt[ch]);
    Serial.println(F("V"));
  }
}

/* ================= LOOP ================= */
void loop() {

  if (Serial.available()) {
    String cmd = Serial.readStringUntil('\n');
    cmd.trim();

    if (cmd == "START") {
      runFlag = true;
      Serial.println(F("RUNNING"));
    }
  }

  if (runFlag) {
    runFlag = false;
    runFullExperiment();
    Serial.println(F("DONE"));
  }
}

/* ================= MAIN EXPERIMENT ================= */
void runFullExperiment() {

  for (int port = 1; port <= 4; port++) {

    Serial1.print("SET ");
    Serial1.println(port);
    delay(SWITCH_DELAY_MS);

    for (int c = 0; c < NUM_COMB; c++) {

      generateCombination();
      applyDACs();

      delay(HEATER_DELAY_MS);

      float pd[NUM_PD];
      readPDs(pd);

      Serial.print(port);
      Serial.print(",");
      Serial.print(c);

      for (int i = 0; i < NUM_ACTIVE; i++) {
        int ch = activeChannels[i];
        Serial.print(",");
        Serial.print(dacVoltages[ch], 3);
      }

      for (int i = 0; i < NUM_PD; i++) {
        Serial.print(",");
        Serial.print(pd[i], 5);
      }

      Serial.println();
    }
  }

  // active heaters off between paths (excluded channels are already 0)
  for (int i = 0; i < NUM_ACTIVE; i++) setDAC(activeChannels[i], 0.0);
}

/* ================= COMBINATION GENERATOR =================
   Generates each channel's random voltage within ITS OWN max
   (0.25V steps), instead of drawing 0-3V then clamping. Clamping
   a 0-3V draw down to a 1.5V ceiling would bunch a large fraction
   of samples exactly at 1.5V (biased distribution) - this avoids
   that by sizing the step count per channel.
   ============================================================ */
void generateCombination() {

  // Only randomize the 13 active channels; excluded channels (H6/H14/H18)
  // simply keep whatever is in dacVoltages[] for them, which is always
  // driven to 0V in applyDACs()/setDAC() regardless.
  for (int i = 0; i < NUM_ACTIVE; i++) {
    int ch = activeChannels[i];
    int levels = (int)(maxVolt[ch] / 0.25) + 1;   // e.g. 1.5V -> 7 levels, 3V -> 13 levels
    int r = random(0, levels);
    dacVoltages[ch] = r * 0.25;

    if (dacVoltages[ch] > maxVolt[ch])
      dacVoltages[ch] = maxVolt[ch];
  }
}

/* ================= PD READ ================= */

void readPDs(float *pd) {

  for (int i = 0; i < NUM_PD; i++) pd[i] = 0;

  for (int n = 0; n < PD_AVG_COUNT; n++) {

    pd[0] += analogRead(A0);
    pd[1] += analogRead(A1);
    pd[2] += analogRead(A2);
    pd[3] += analogRead(A3);

    delay(PD_SAMPLE_DELAY);
  }

  for (int i = 0; i < NUM_PD; i++) {
    pd[i] = (pd[i] / PD_AVG_COUNT) * (5.0 / 1023.0);
  }
}

/* ================= DAC ================= */

void configureDAC() {

  digitalWrite(CS_DAC, LOW);
  SPI.transfer(0x03);
  SPI.transfer(0x00);
  SPI.transfer(0x84);
  digitalWrite(CS_DAC, HIGH);
  delayMicroseconds(5);

  digitalWrite(CS_DAC, LOW);
  SPI.transfer(0x09);
  SPI.transfer(0x00);
  SPI.transfer(0x00);
  digitalWrite(CS_DAC, HIGH);
}

void applyDACs() {
  // Drive only the 13 active channels. Excluded channels (H6/H14/H18)
  // were already forced to 0V in setup() and are never touched again.
  for (int i = 0; i < NUM_ACTIVE; i++) {
    int ch = activeChannels[i];
    setDAC(ch, dacVoltages[ch]);
  }
}

void setDAC(uint8_t ch, float voltage) {

  voltage = constrain(voltage, 0, maxVolt[ch]);

  uint16_t code = (uint16_t)((voltage / VREF_DAC) * 65535.0);

  digitalWrite(CS_DAC, LOW);

  SPI.transfer(0x10 | (ch & 0x0F));
  SPI.transfer(code >> 8);
  SPI.transfer(code & 0xFF);

  digitalWrite(CS_DAC, HIGH);
}
