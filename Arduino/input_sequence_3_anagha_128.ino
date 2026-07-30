#include <SPI.h>

// ------------------------------
// DAC and SPI Configuration
// ------------------------------
const int CS_DAC0 = 10;
const int CS_DAC1 = 9;
const int CS_DAC2 = 8;
const int CS_DAC3 = 7;
// NEW CHIP SELECT PINS ADDED for DACs 4 through 7 (Channels 64-127)
const int CS_DAC4 = 6;
const int CS_DAC5 = 5;
const int CS_DAC6 = 4;
const int CS_DAC7 = 3;

const int numDAC = 128;  // Total number of DAC channels (8 chips x 16 channels)
float voltages[numDAC] = {0};  // Array to hold DAC voltage values (in volts)

// ------------------------------
// ADC Configuration
// ------------------------------
const int numPins = 14;  // Number of ADC channels to sample (Mega has A0–A15)
const unsigned long sampleInterval = 10; // Sampling interval in milliseconds

// ------------------------------
// Setup
// ------------------------------
void setup() {
  Serial.begin(115200);
  while (!Serial) {
    ; // Wait for serial port to connect (needed for some boards)
  }

  SPI.begin();
  // Ensure SPI settings are appropriate for your DAC chip
  SPI.beginTransaction(SPISettings(10000000, MSBFIRST, SPI_MODE1));

  // Initialize CS pins
  pinMode(CS_DAC0, OUTPUT);
  pinMode(CS_DAC1, OUTPUT);
  pinMode(CS_DAC2, OUTPUT);
  pinMode(CS_DAC3, OUTPUT);
  pinMode(CS_DAC4, OUTPUT);
  pinMode(CS_DAC5, OUTPUT);
  pinMode(CS_DAC6, OUTPUT);
  pinMode(CS_DAC7, OUTPUT);

  // Deactivate all CS pins
  digitalWrite(CS_DAC0, HIGH);
  digitalWrite(CS_DAC1, HIGH);
  digitalWrite(CS_DAC2, HIGH);
  digitalWrite(CS_DAC3, HIGH);
  digitalWrite(CS_DAC4, HIGH);
  digitalWrite(CS_DAC5, HIGH);
  digitalWrite(CS_DAC6, HIGH);
  digitalWrite(CS_DAC7, HIGH);

  Serial.println("Arduino ready. Send 128 comma-separated DAC voltages.");
}

// ------------------------------
// Main Loop
// ------------------------------
void loop() {
  if (Serial.available() > 0) {
    String inputLine = Serial.readStringUntil('\n');
    inputLine.trim();

    if (parseVoltages(inputLine, voltages)) {
      sendtoDAC();
      delay(100);   // Wait for DAC outputs to settle
      readAndSendADCValues(); // Capture ADC values and time stamp
    } else {
      Serial.println("Error: Invalid number of voltages received.");
    }
  }
}

// ------------------------------
// Function: Parse DAC Voltage Values
// ------------------------------
bool parseVoltages(const String &input, float dacVoltages[]) {
  // Increased buffer size to handle 128 comma-separated floats
  char inputBuffer[1024];
  input.toCharArray(inputBuffer, sizeof(inputBuffer));  // Convert to char array
  char* token = strtok(inputBuffer, ",");
  int index = 0;

  while (token != NULL && index < numDAC) {
    dacVoltages[index] = atof(token);  // Convert token to float
    index++;
    token = strtok(NULL, ",");  // Get next token
  }

  return (index == numDAC);
}

// ------------------------------
// Function: Update DAC Values
// ------------------------------
void sendtoDAC() {
  for (int DAC_SEL = 0; DAC_SEL < numDAC; DAC_SEL++) {
    int dacChannel = DAC_SEL % 16;
    int CSdac;

    // Map DAC_SEL (0–127) to the correct Chip Select pin (8 chips total)
    if (DAC_SEL < 16) {
      CSdac = CS_DAC0;
    } else if (DAC_SEL < 32) {
      CSdac = CS_DAC1;
    } else if (DAC_SEL < 48) {
      CSdac = CS_DAC2;
    } else if (DAC_SEL < 64) {
      CSdac = CS_DAC3;
    } else if (DAC_SEL < 80) {
      CSdac = CS_DAC4;
    } else if (DAC_SEL < 96) {
      CSdac = CS_DAC5;
    } else if (DAC_SEL < 112) {
      CSdac = CS_DAC6;
    } else {
      CSdac = CS_DAC7;
    }

    // Calculate 16-bit decimal value (assuming 5V reference)
    unsigned int decimalValue = (unsigned int)((voltages[DAC_SEL] * 65535.0) / 5.0);
    unsigned int MSB = (decimalValue >> 8) & 0xFF;
    unsigned int LSB = decimalValue & 0xFF;

    // Command 1: Write to DAC channel
    digitalWrite(CSdac, LOW);
    SPI.transfer(0x10 + dacChannel); // 0x10 = write data to input register n
    SPI.transfer(MSB);
    SPI.transfer(LSB);
    digitalWrite(CSdac, HIGH);

    // Additional configuration commands
    digitalWrite(CSdac, LOW);
    SPI.transfer(0x03);
    SPI.transfer(0x00);
    SPI.transfer(0x84);
    digitalWrite(CSdac, HIGH);

    digitalWrite(CSdac, LOW);
    SPI.transfer(0x09);
    SPI.transfer(0x00);
    SPI.transfer(0x00);
    digitalWrite(CSdac, HIGH);

    digitalWrite(CSdac, LOW);
    SPI.transfer(0x05);
    SPI.transfer(0xFF);
    SPI.transfer(0xFF);
    digitalWrite(CSdac, HIGH);
  }
}

// ------------------------------
// Function: Sample ADCs and Send Data
// ------------------------------
void readAndSendADCValues() {
  unsigned long startTime = millis();
  long adcSum[numPins] = {0};
  int count = 0;

  while (millis() - startTime < sampleInterval) {
    for (int i = 0; i < numPins; i++) {
      int val = analogRead(i);
      adcSum[i] += val;
    }
    count++;
  }

  // --- Output Elapsed Time ---
  Serial.print(millis());
  Serial.print(",");

  for (int i = 0; i < numPins; i++) {
    float mean = (float)adcSum[i] / count;
    float meanVoltage = (mean / 1023.0) * 5.0; // Assuming 10-bit ADC, 5V ref
    Serial.print(meanVoltage, 4);

    if (i < numPins - 1) {
      Serial.print(",");
    }
  }
  Serial.println();
}
