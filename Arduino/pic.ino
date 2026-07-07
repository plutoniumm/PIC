#include <SPI.h> // working code for manual and auto char

// DEFINE THE CHIP SELECT PINS
int CS_DAC0 = 5;
int CS_DAC1 = 4;
int CS_DAC2 = 3;
int CS_DAC3 = 2;

const unsigned long duration = 10;  // Duration to read in milliseconds (1 second)
unsigned long startTime;
bool startReading = false;
const int numPins = 14;  // Number of ADC pins (A0 to A15)

void setup() {
  SPI.beginTransaction(SPISettings(10000000, MSBFIRST, SPI_MODE1));
  SPI.begin();                  // SPI Config
  Serial.begin(115200);         // Serial monitor config with baudrate

  // PIN MODE
  pinMode(CS_DAC0, OUTPUT);
  pinMode(CS_DAC1, OUTPUT);
  pinMode(CS_DAC2, OUTPUT);
  pinMode(CS_DAC3, OUTPUT);

  // Ensure CS pins are inactive (HIGH)
  digitalWrite(CS_DAC0, HIGH);
  digitalWrite(CS_DAC1, HIGH);
  digitalWrite(CS_DAC2, HIGH);
  digitalWrite(CS_DAC3, HIGH);
}

void loop() {
  // Select mode: Manual or Automated
  Serial.println("Select mode: 0 for Manual, 1 for Automated");
  while (!Serial.available());
  int mode = Serial.readString().toInt();

  if (mode == 0) {
    manualMode();
  } else if (mode == 1) {
    automatedMode();
  } else {
    Serial.println("Invalid mode selected. Please choose 0 or 1.");
  }
}

void manualMode() {
  // Get DAC input from user (0-63)
  Serial.println("Select DAC (0 to 63): ");
  while (!Serial.available());
  int DAC_SEL = Serial.readString().toInt();

  if (DAC_SEL >= 0 && DAC_SEL <= 63) {
    Serial.print("Selected DAC: ");
    Serial.println(DAC_SEL);

    // Get voltage input from user (0-5V)
    Serial.println("Set voltage for DAC (0V to 5V): ");
    while (!Serial.available());
    float Voltage = Serial.readString().toFloat();
    Serial.print("Set voltage: ");
    Serial.println(Voltage);

    if (Voltage >= 0 && Voltage <= 5) {
      // Convert voltage to DAC value (16-bit)
      unsigned int decimal = (Voltage * 65535) / 5;
      unsigned int MSB = (decimal >> 8) & 0xFF;  // MSB shift
      unsigned int LSB = decimal & 0xFF;         // LSB shift

      // Determine which DAC chip to use based on DAC_SEL
      int CSdac;
      if (DAC_SEL < 16) {
        CSdac = CS_DAC0;
      } else if (DAC_SEL < 32) {
        CSdac = CS_DAC1;
        DAC_SEL -= 16;  // Adjust DAC channel for this chip
      } else if (DAC_SEL < 48) {
        CSdac = CS_DAC2;
        DAC_SEL -= 32;  // Adjust DAC channel for this chip
      } else {
        CSdac = CS_DAC3;
        DAC_SEL -= 48;  // Adjust DAC channel for this chip
      }

      // Send the DAC value to the selected channel
      sendToDAC(CSdac, DAC_SEL, MSB, LSB);
      Serial.println("DAC configured successfully.");
    } else {
      Serial.println("Invalid voltage input. Must be between 0 and 5V.");
    }
  } else {
    Serial.println("Invalid DAC input. Must be between 0 and 63.");
  }
  // Collect ADC readings for the duration
    startReading = true;
    startTime = millis();
    long adcSum[numPins] = {0};
    int count = 0;
    long adcSumSq[numPins] = {0};

    while (millis() - startTime < duration) {
      for (int i = 0; i < numPins; i++) {
        int val = analogRead(i);
        adcSum[i] += val;
        adcSumSq[i] += (long)val * val;
      }
      count++;
      //delay(1);
    }

    // Print mean ADC values
    for (int i = 0; i < numPins; i++) {
      float meanADC = (float)adcSum[i] / count;
  float meanVoltage = (meanADC / 1023.0) * 5.0;

  float meanSq = (float)adcSumSq[i] / count;
  float stddev = sqrt(meanSq - (meanADC * meanADC));
  float stddevVoltage = (stddev / 1023.0) * 5.0;

  Serial.print(meanVoltage, 3);
  Serial.print("±");
  Serial.print(stddevVoltage, 3);

  if (i < numPins - 1) Serial.print(", ");
}
    Serial.println();
  }


void automatedMode() {
  Serial.println("Enter DAC numbers to automate (comma-separated, e.g., 3,4,6,8): ");
  while (!Serial.available());
  String input = Serial.readString();
  input.trim();

  // Parse DAC numbers from input
  int dacList[64];
  int dacCount = 0;

  char *token = strtok(input.c_str(), ",");
  while (token != NULL) {
    int dac = atoi(token);
    if (dac >= 0 && dac < 64) {
      dacList[dacCount++] = dac;
    } else {
      Serial.print("Invalid DAC number ignored: ");
      Serial.println(dac);
    }
    token = strtok(NULL, ",");
  }

  if (dacCount == 0) {
    Serial.println("No valid DAC numbers provided. Exiting automated mode.");
    return;
  }

  float step = 0.25;  // Voltage step increment
  for (float voltage = 0; voltage <= 5; voltage += step) {
    for (int i = 0; i < dacCount; i++) {
      int dac = dacList[i];
      int CSdac;
      int adjustedDAC = dac;

      // Determine which chip to use
      if (dac < 16) {
        CSdac = CS_DAC0;
      } else if (dac < 32) {
        CSdac = CS_DAC1;
        adjustedDAC -= 16;
      } else if (dac < 48) {
        CSdac = CS_DAC2;
        adjustedDAC -= 32;
      } else {
        CSdac = CS_DAC3;
        adjustedDAC -= 48;
      }

      // Convert voltage to DAC value (16-bit)
      unsigned int decimal = (voltage * 65535) / 5;
      unsigned int MSB = (decimal >> 8) & 0xFF;  // MSB shift
      unsigned int LSB = decimal & 0xFF;         // LSB shift

      // Send the DAC value to the selected channel
      sendToDAC(CSdac, adjustedDAC, MSB, LSB);

//      Serial.print("DAC ");
//      Serial.print(dac);
//      Serial.print(",");
      Serial.print(voltage);
      Serial.print(",");
    }

    // Collect ADC readings for the duration
    startReading = true;
    startTime = millis();
    long adcSum[numPins] = {0};
    int count = 0;
    long adcSumSq[numPins] = {0};

    while (millis() - startTime < duration) {
      for (int i = 0; i < numPins; i++) {
        int val = analogRead(i);
        adcSum[i] += val;
        adcSumSq[i] += (long)val * val;
      }
      count++;
      //delay(1);
    }

    // Print mean ADC values
    for (int i = 0; i < numPins; i++) {
      float meanADC = (float)adcSum[i] / count;
  float meanVoltage = (meanADC / 1023.0) * 5.0;

  float meanSq = (float)adcSumSq[i] / count;
  float stddev = sqrt(meanSq - (meanADC * meanADC));
  float stddevVoltage = (stddev / 1023.0) * 5.0;

  Serial.print(meanVoltage, 3);
  Serial.print("±");
  Serial.print(stddevVoltage, 3);

  if (i < numPins - 1) Serial.print(", ");
}
    Serial.println();
  }

  // Reset selected DACs to 0V
  for (int i = 0; i < dacCount; i++) {
    int dac = dacList[i];
    int CSdac;
    int adjustedDAC = dac;

    if (dac < 16) {
      CSdac = CS_DAC0;
    } else if (dac < 32) {
      CSdac = CS_DAC1;
      adjustedDAC -= 16;
    } else if (dac < 48) {
      CSdac = CS_DAC2;
      adjustedDAC -= 32;
    } else {
      CSdac = CS_DAC3;
      adjustedDAC -= 48;
    }

    sendToDAC(CSdac, adjustedDAC, 0x00, 0x00);
  }

  Serial.println("DAC automated mode completed.");
  startReading = false;  // Stop ADC readings
}

// Function to send data to the selected DAC channel
void sendToDAC(int CSdac, int DACChannel, unsigned int MSB, unsigned int LSB) {
  digitalWrite(CSdac, LOW);
  SPI.transfer(0x10 | DACChannel);  // Command for channel 0-15
  SPI.transfer(MSB);                // Send MSB
  SPI.transfer(LSB);                // Send LSB
  digitalWrite(CSdac, HIGH);


  digitalWrite(CSdac, LOW );
  SPI.transfer(0x03);
  SPI.transfer(0x00);
  SPI.transfer(0x84);
  digitalWrite(CSdac, HIGH);

  // Power up all channels
  digitalWrite(CSdac, LOW);
  SPI.transfer(0x09);
  SPI.transfer(0x00);
  SPI.transfer(0x00);
  digitalWrite(CSdac, HIGH);

  // Enable broadcast config for all channels
  digitalWrite(CSdac, LOW);
  SPI.transfer(0x05);
  SPI.transfer(0xFF);
  SPI.transfer(0xFF);
  digitalWrite(CSdac, HIGH);
}
