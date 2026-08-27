#include <SPI.h>

// Define the chip select pins
int CS_DAC0 = 5;
int CS_DAC1 = 4;
int CS_DAC2 = 3;
int CS_DAC3 = 2;

const int numPins = 14;  // Number of ADC pins (A0 to A13)
// ADC averaging window. One sweep of 14 pins is ~1.46 ms (~104 us/analogRead on
// the Mega), so this many ms fits ~interval/1.46 sweeps: 75 ms -> ~50 sweeps.
const unsigned long interval = 75; // ~50 averaged sweeps (was 31 ms / ~20)
const unsigned long dacSettlingTime = 10; // DAC settling time in microseconds

void setup() {
  Serial.begin(115200);

  // Set up CS pins
  pinMode(CS_DAC0, OUTPUT);
  pinMode(CS_DAC1, OUTPUT);
  pinMode(CS_DAC2, OUTPUT);
  pinMode(CS_DAC3, OUTPUT);

  // Ensure CS pins are inactive (HIGH)
  digitalWrite(CS_DAC0, HIGH);
  digitalWrite(CS_DAC1, HIGH);
  digitalWrite(CS_DAC2, HIGH);
  digitalWrite(CS_DAC3, HIGH);

  // Initialize SPI - CORRECT ORDER
  SPI.begin();
  SPI.beginTransaction(SPISettings(10000000, MSBFIRST, SPI_MODE1));

  // Initialize all DAC chips
  initializeDAC(CS_DAC0);
  initializeDAC(CS_DAC1);
  initializeDAC(CS_DAC2);
  initializeDAC(CS_DAC3);

  Serial.println("DACs initialized and ready");
}

void loop() {
  if (Serial.available()) {
    // Read 64 values from Python
    String input = Serial.readStringUntil('\n');
    float values[64];
    int startIdx = 0;

    for (int i = 0; i < 64; i++) {
      int commaIdx = input.indexOf(',', startIdx);
      if (commaIdx == -1) {
        values[i] = input.substring(startIdx).toFloat();
        break;
      }
      values[i] = input.substring(startIdx, commaIdx).toFloat();
      startIdx = commaIdx + 1;
    }

    // Send values to DACs
    for (int i = 0; i < 64; i++) {
      int dacChannel = i % 16;  // Each DAC chip can handle 16 channels
      int csPin;
      if (i < 16) {
        csPin = CS_DAC0;
      } else if (i < 32) {
        csPin = CS_DAC1;
      } else if (i < 48) {
        csPin = CS_DAC2;
      } else {
        csPin = CS_DAC3;
      }

      sendDACvalue(values[i], csPin, dacChannel);
      delayMicroseconds(dacSettlingTime);  // Allow DAC to settle
    }

    // Read ADC values and send them back to Python
    readAndSendADCValues();
  }
}

// Initialize a DAC chip (power up all channels and enable broadcast)
void initializeDAC(int csPin) {
  // Power up all channels (command 0x03)
  digitalWrite(csPin, LOW);
  SPI.transfer(0x03);  // Power control command
  SPI.transfer(0x00);
  SPI.transfer(0xFF);  // Power up all 16 channels (0xFF = all bits set)
  digitalWrite(csPin, HIGH);
  delay(1);

  // Enable internal reference (command 0x04) - adjust based on your DAC model
  digitalWrite(csPin, LOW);
  SPI.transfer(0x04);
  SPI.transfer(0x00);
  SPI.transfer(0x01);  // Enable internal reference
  digitalWrite(csPin, HIGH);
  delay(1);
}

// Send a single voltage value to a DAC channel
void sendDACvalue(float voltage, int csPin, int channel) {
  // Clamp voltage to valid range
  if (voltage < 0.0)
    voltage = 0.0;
  if (voltage > 2.0)
    voltage = 2.0;

  // Convert voltage to 16-bit DAC value
  unsigned int dacValue = (voltage * 65535) / 5.0;
  unsigned int MSB = (dacValue >> 8) & 0xFF;
  unsigned int LSB = dacValue & 0xFF;

  // Write to DAC channel (command 0x1n where n is the channel)
  digitalWrite(csPin, LOW);
  SPI.transfer(0x10 | channel);  // Write and update channel
  SPI.transfer(MSB);
  SPI.transfer(LSB);
  digitalWrite(csPin, HIGH);
}

// Read ADC values for each pin and send them to Python
void readAndSendADCValues() {
  unsigned long startTime = millis();
  long adcSum[numPins] = {0};
  int count = 0;

  while (millis() - startTime < interval) {
    for (int i = 0; i < numPins; i++) {
      adcSum[i] += analogRead(i);
    }
    count++;
  }

  for (int i = 0; i < numPins; i++) {
    float meanVoltage = ((float)adcSum[i] / count / 1023.0) * 5.0;
    Serial.print(meanVoltage, 3);
    if (i < numPins - 1) {
      Serial.print(",");
    }
  }
  Serial.println();
}