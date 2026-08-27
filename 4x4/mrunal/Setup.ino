#include <SPI.h>

/* ================= USER PARAMETERS ================= */
#define NUM_DAC_CH        6
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

/* ================= VOLTAGE LIMITS ================= */
float maxVolt[NUM_DAC_CH] = {3,3,3,3,3,3};

/* ================= SETUP ================= */
void setup() {

  Serial.begin(115200);
  Serial1.begin(9600);

  pinMode(CS_DAC, OUTPUT);
  digitalWrite(CS_DAC, HIGH);

  SPI.begin();
  SPI.beginTransaction(SPISettings(10000000, MSBFIRST, SPI_MODE1));

  configureDAC();

  randomSeed(analogRead(A7));

  Serial.println("READY");
  Serial.println("Send START");
}

/* ================= LOOP ================= */
void loop() {

  if (Serial.available()) {

    String cmd = Serial.readStringUntil('\n');
    cmd.trim();

    if (cmd == "START") {
      runFlag = true;
      Serial.println("RUNNING");
    }
  }

  if (runFlag) {

    runFlag = false;
    runFullExperiment();
    Serial.println("DONE");
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

      for (int i = 0; i < NUM_DAC_CH; i++) {
        Serial.print(",");
        Serial.print(dacVoltages[i],3);
      }

      for (int i = 0; i < NUM_PD; i++) {
        Serial.print(",");
        Serial.print(pd[i],5);
      }

      Serial.println();
    }
  }
}

/* ================= COMBINATION GENERATOR ================= */

void generateCombination() {

  int levels = 13;   // 0 → 3V in steps of 0.25

  for (int i = 0; i < NUM_DAC_CH; i++) {

    int r = random(0, levels);
    dacVoltages[i] = r * 0.25;

    if (dacVoltages[i] > maxVolt[i])
      dacVoltages[i] = maxVolt[i];
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

  for (int i = 0; i < NUM_DAC_CH; i++)
    setDAC(i, dacVoltages[i]);
}

void setDAC(uint8_t ch, float voltage) {

  voltage = constrain(voltage,0,maxVolt[ch]);

  uint16_t code = (uint16_t)((voltage / VREF_DAC) * 65535.0);

  digitalWrite(CS_DAC, LOW);

  SPI.transfer(0x10 | (ch & 0x0F));
  SPI.transfer(code >> 8);
  SPI.transfer(code & 0xFF);

  digitalWrite(CS_DAC, HIGH);
}