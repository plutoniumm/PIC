#include <SPI.h>

/* =====================================================================
   PIN / WIRING VERIFICATION SKETCH  (13 heaters — H6/H14/H18 excluded)
   ---------------------------------------------------------------------
   Purpose: confirm that DAC channel N is actually wired to the heater/
   test-point you *think* it is, BEFORE running any full sweep.

   H6, H14, H18 (DAC channels 11, 8, 6) are NOT in the resistance table
   you have, so they are excluded entirely: forced to 0 V once at
   startup and never driven or checked by this sketch.

   How it works:
     - Only ONE (active) channel is ever driven at a time; every other
       channel, including the 3 excluded ones, is held at 0 V.
     - The driven channel is held at a small, safe TEST_VOLTAGE
       (default 0.4 V) regardless of that heater's category, so even
       an unexpectedly low-resistance (60 ohm) heater sees only:
           I = 0.4 / 60 = 6.7 mA   (well under the 30 mA rating)
     - You probe the expected TP/BP pin with a multimeter (or watch
       the corresponding PD channel move) and confirm it matches the
       table below before moving on.
     - Advance to the next channel only when YOU send 'n' over Serial.
       Nothing auto-advances, so there's no risk of skipping a check.

   Usage:
     1. Open Serial Monitor at 115200 baud, line ending = Newline.
     2. Send: START
     3. For each channel it will print which heater/pin it expects,
        and hold that voltage. Probe the pin, confirm, then send: n
     4. Send 'r' at any time to repeat the current channel's test.
     5. Send 'q' to stop and force all channels back to 0 V.
   ===================================================================== */

#define NUM_DAC_CH   16   // physical channel count on the DAC81416
#define CS_DAC       10
#define VREF_DAC     5.0
#define TEST_VOLTAGE 0.4     // safe probe voltage, same for all channels

/* Mapping decoded from your notebook — EDIT if you find a mismatch */
struct HeaterInfo {
  const char* heaterName;
  const char* plusPin;
  const char* gndPin;
  const char* resistanceNote;
};

HeaterInfo heaterMap[NUM_DAC_CH] = {
  /*DAC0 */ {"H15", "TP24", "TP26", "114.1 ohm (120R grp)"},
  /*DAC1 */ {"H12", "BP17", "BP20", "62.3 ohm (60R grp)"},
  /*DAC2 */ {"H11", "BP16", "BP20", "118.8 ohm (120R grp)"},
  /*DAC3 */ {"H8",  "BP13", "BP15", "114.1 ohm (120R grp)"},
  /*DAC4 */ {"H4",  "BP9",  "BP7",  "57.8 ohm (60R grp)"},
  /*DAC5 */ {"H3",  "TP9",  "BP1",  "113.5 ohm (120R grp)"},
  /*DAC6 */ {"H18", "BP21", "?",    "EXCLUDED - resistance unknown"},
  /*DAC7 */ {"H9",  "BP14", "BP15", "57.2 ohm (60R grp)"},
  /*DAC8 */ {"H14", "TP25", "?",    "EXCLUDED - resistance unknown"},
  /*DAC9 */ {"H13", "TP23", "TP26", "114.9 ohm (120R grp)"},
  /*DAC10*/ {"H10", "TP13", "TP15", "56.4 ohm (60R grp)"},
  /*DAC11*/ {"H6",  "TP10", "?",    "EXCLUDED - resistance unknown"},
  /*DAC12*/ {"H7",  "BP12", "BP15", "116.1 ohm (120R grp)"},
  /*DAC13*/ {"H5",  "BP10", "BP7",  "117.1 ohm (120R grp)"},
  /*DAC14*/ {"H2",  "BP11", "BP1",  "116.8 ohm (120R grp)"},
  /*DAC15*/ {"H1",  "TP8",  "BP1",  "114.1 ohm (120R grp)"},
};

/* Channels to actually check. H6(11), H14(8), H18(6) are left out. */
const int activeChannels[] = {0, 1, 2, 3, 4, 5, 7, 9, 10, 12, 13, 14, 15};
const int NUM_ACTIVE = sizeof(activeChannels) / sizeof(activeChannels[0]); // 13

bool runFlag = false;

void setup() {
  Serial.begin(115200);

  pinMode(CS_DAC, OUTPUT);
  digitalWrite(CS_DAC, HIGH);

  SPI.begin();
  SPI.beginTransaction(SPISettings(10000000, MSBFIRST, SPI_MODE1));

  configureDAC();
  allChannelsZero();

  Serial.println(F("READY - wiring check sketch"));
  Serial.println(F("Checking 13 heaters only. H6, H14, H18 excluded (held at 0V)."));
  Serial.println(F("Send START to begin"));
}

void loop() {
  if (Serial.available()) {
    String cmd = Serial.readStringUntil('\n');
    cmd.trim();
    if (cmd == "START") {
      runFlag = true;
    }
  }

  if (runFlag) {
    runFlag = false;
    runChannelCheck();
  }
}

void runChannelCheck() {
  for (int idx = 0; idx < NUM_ACTIVE; idx++) {
    int ch = activeChannels[idx];

    bool repeat = true;
    while (repeat) {
      repeat = false;

      allChannelsZero();
      setDAC(ch, TEST_VOLTAGE);

      Serial.println(F("----------------------------------------"));
      Serial.print(F("Driving DAC channel: ")); Serial.println(ch);
      Serial.print(F("Expected heater:     ")); Serial.println(heaterMap[ch].heaterName);
      Serial.print(F("Expected (+) pin:    ")); Serial.println(heaterMap[ch].plusPin);
      Serial.print(F("Expected GND pin:    ")); Serial.println(heaterMap[ch].gndPin);
      Serial.print(F("Resistance note:     ")); Serial.println(heaterMap[ch].resistanceNote);
      Serial.print(F("Applied test voltage: ")); Serial.print(TEST_VOLTAGE); Serial.println(F(" V"));
      Serial.println(F("Probe the expected pin now."));
      Serial.println(F("Send 'n' = next channel, 'r' = repeat this channel, 'q' = stop"));

      // Wait for user response
      while (true) {
        if (Serial.available()) {
          String resp = Serial.readStringUntil('\n');
          resp.trim();
          if (resp == "n") {
            break;
          } else if (resp == "r") {
            repeat = true;
            break;
          } else if (resp == "q") {
            allChannelsZero();
            Serial.println(F("STOPPED - all channels set to 0V"));
            return;
          }
        }
      }
    }
  }

  allChannelsZero();
  Serial.println(F("CHECK COMPLETE - all channels back to 0V"));
}

void allChannelsZero() {
  for (int i = 0; i < NUM_DAC_CH; i++) setDAC(i, 0.0);
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

void setDAC(uint8_t ch, float voltage) {
  voltage = constrain(voltage, 0, VREF_DAC);
  uint16_t code = (uint16_t)((voltage / VREF_DAC) * 65535.0);

  digitalWrite(CS_DAC, LOW);
  SPI.transfer(0x10 | (ch & 0x0F));
  SPI.transfer(code >> 8);
  SPI.transfer(code & 0xFF);
  digitalWrite(CS_DAC, HIGH);
}
