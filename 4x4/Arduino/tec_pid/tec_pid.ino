#include <SPI.h>
#include <PID_v1.h>
#include <math.h>

// =======================================================
// 1. HARDWARE PINS
// =======================================================

const int CS_PIN = 10;
const int NTC_PIN = A0;

// =======================================================
// 2. LT8722 REGISTER ADDRESSES
// =======================================================

const uint8_t ADDR_COMMAND  = 0x00;
const uint8_t ADDR_STATUS   = 0x02;
const uint8_t ADDR_ILIMN    = 0x04;
const uint8_t ADDR_ILIMP    = 0x06;
const uint8_t ADDR_DAC      = 0x08;
const uint8_t ADDR_OV_CLAMP = 0x0A;
const uint8_t ADDR_UV_CLAMP = 0x0C;

// =======================================================
// 3. TEC DRIVER LIMITS
// =======================================================

// ILIMP = 6.8 A - code * 13.28 mA (datasheet Eq. 9). 0x1B5 was +0.997 A; raised to +2.006 A
// on 2026-09-23 (the loop sat in current limit, STATUS bit 1). LT8722 rating is 4 A.
const uint32_t POS_CURRENT_2A  = 0x00000169;
// Eq. 10 is ILIMN = code * -13.28 mA, NOT Eq. 9's downward scale, so 0x1B5 is -5.80 A --
// past the part's own -4.5 A sink rating. 0x4B (75) is the -1 A this constant claims.
const uint32_t NEG_CURRENT_1A  = 0x0000004B;

const uint32_t VOLT_CLAMP_6V   = 0x00000004;
const uint32_t VOLT_CLAMP_0V   = 0x0000000F;

// SPIS_COMMAND is 22 bits and was written as a bare 0x17, which also cleared VCC_VREG (VCC
// 3.1 V, only valid with an externally overdriven VCC), SW_VC_INT (minimum inductor current)
// and PWR_LIM (2 W). Keep the datasheet default 0x08A214 and set only the enable bits.
const uint32_t CMD_DEFAULT     = 0x0008A214;
const uint32_t CMD_ENABLE      = CMD_DEFAULT | 0x1;  // ENABLE_REQ: VCC LDO + linear stage
const uint32_t CMD_STARTUP_PWM = CMD_DEFAULT | 0x3;  // + SWEN_REQ: PWM switching

// =======================================================
// 4. DAC SCALING
// =======================================================

const double DAC_MAX = 5.0;  // was 2.0: railed under any heater load (2026-09-23)
const double REF_VOLTAGE = 3.0;
const double INTERNAL_GAIN = 16.0;
const double GAIN_ADJUST = 0.42;   // calibrated

// =======================================================
// 5. THERMISTOR PARAMETERS
// =======================================================

const float Vin = 5.0;
const float R_fixed = 10000;

const float R_nominal = 10000.0;
const float T_nominal = 25.0;
const float Beta = 3450.0;

const float calibrationOffset = 1.3;

float alpha = 0.08;
float filteredR = 10000.0;
bool firstRun = true;

// =======================================================
// 6. PID VARIABLES
// =======================================================

double Setpoint = 25.0;
double Input;
double Output;

// Kp alone left a steady error of drive/Kp: holding the die took +1.63 V, so it sat 0.33 C
// warm at 25.31 and never reached 25.00. Ki trims that away; PID_v1 clamps the integral to
// the output limits, so it cannot wind up while the drive is at its cap.
PID tecPID(&Input, &Output, &Setpoint, 5, 0.1, 0, REVERSE);

// =======================================================
// 7. CRC FUNCTION
// =======================================================

uint8_t calculateCRC(uint8_t cmd, uint8_t addr, uint32_t data)
{
    uint8_t bytes[6] =
    {
        cmd,
        addr,
        (uint8_t)(data >> 24),
        (uint8_t)(data >> 16),
        (uint8_t)(data >> 8),
        (uint8_t)data
    };

    uint8_t crc = 0;

    for (int i = 0; i < 6; i++)
    {
        crc ^= bytes[i];

        for (int j = 0; j < 8; j++)
        {
            if (crc & 0x80)
                crc = (crc << 1) ^ 0x07;
            else
                crc <<= 1;
        }
    }

    return crc;
}

// =======================================================
// 8. SPI WRITE
// =======================================================

uint32_t lt8722_read(uint8_t addr)
{
    // Same 8-byte frame as a write, opcode 0xF4, data bytes zeroed and the CRC computed over
    // them. A read without a valid CRC is rejected and the part answers with a fixed error
    // response -- identical for every address, which is what made an earlier attempt look
    // like the registers all held the same value.
    uint8_t cmd = 0xF4;
    uint8_t crc = calculateCRC(cmd, addr, 0);
    uint8_t r[8];

    digitalWrite(CS_PIN, LOW);
    r[0] = SPI.transfer(cmd);
    r[1] = SPI.transfer(addr);
    for (int i = 2; i < 6; i++) r[i] = SPI.transfer(0x00);
    r[6] = SPI.transfer(crc);
    r[7] = SPI.transfer(0x00);
    digitalWrite(CS_PIN, HIGH);

    return ((uint32_t)r[2] << 24) | ((uint32_t)r[3] << 16)
         | ((uint32_t)r[4] << 8)  |  (uint32_t)r[5];
}

void lt8722_write(uint8_t addr, uint32_t data)
{
    uint8_t cmd = 0xF2;

    uint8_t crc = calculateCRC(cmd, addr, data);

    digitalWrite(CS_PIN, LOW);

    SPI.transfer(cmd);
    SPI.transfer(addr);

    SPI.transfer(data >> 24);
    SPI.transfer(data >> 16);
    SPI.transfer(data >> 8);
    SPI.transfer(data);

    SPI.transfer(crc);
    SPI.transfer(0x00);

    digitalWrite(CS_PIN, HIGH);
}

// =======================================================
// 9. DAC UPDATE
// =======================================================

void updateLT8722_DAC(double voltage)
{
    if (voltage > DAC_MAX) voltage = DAC_MAX;
    if (voltage < -2.0) voltage = -2.0;

    int32_t signedCode = (int32_t)round(
        (voltage * 16777216.0) / (REF_VOLTAGE * INTERNAL_GAIN * GAIN_ADJUST));

    uint32_t code = (uint32_t)signedCode;

    lt8722_write(ADDR_DAC, code);
}

// =======================================================
// 10. TEMPERATURE READING
// =======================================================

double readTemperature()
{
    int rawADC = analogRead(NTC_PIN);

    float V_out = rawADC * (Vin / 1023.0);

    if (V_out >= Vin || V_out <= 0.1) return Input;

    float R_raw = R_fixed * (V_out / (Vin - V_out));

    if (firstRun)
    {
        filteredR = R_raw;
        firstRun = false;
    }

    filteredR = alpha * R_raw + (1 - alpha) * filteredR;

    float steinhart =
        log(filteredR / R_nominal) / Beta +
        (1.0 / (T_nominal + 273.15));

    float tempC = (1.0 / steinhart) - 273.15;

    return tempC + calibrationOffset;
}

// =======================================================
// 11. SETUP
// =======================================================

void setup()
{
    Serial.begin(115200);

    pinMode(CS_PIN, OUTPUT);
    digitalWrite(CS_PIN, HIGH);

    SPI.begin();
    SPI.beginTransaction(SPISettings(1000000, MSBFIRST, SPI_MODE0));

    Serial.println("Starting LT8722...");

    // Datasheet start-up sequence (LT8722 Rev. B, "Enable and startup sequence")
    lt8722_write(ADDR_COMMAND, CMD_ENABLE);
    delay(10);
    lt8722_write(ADDR_DAC, 0xFF000000);  // LDR to GND when the linear stage turns on
    lt8722_write(ADDR_ILIMP, POS_CURRENT_2A);
    lt8722_write(ADDR_ILIMN, NEG_CURRENT_1A);
    lt8722_write(ADDR_OV_CLAMP, VOLT_CLAMP_6V);
    lt8722_write(ADDR_UV_CLAMP, VOLT_CLAMP_0V);
    lt8722_write(ADDR_STATUS, 0x00000000);  // clear latched faults: enables the linear stage
    delay(2);
    // ramp 0xFF000000 -> 0 over >= 5 ms so both outputs rise together to VIN/2
    for (int32_t c = (int32_t)0xFF000000; c < 0; c += 0x40000) {
        lt8722_write(ADDR_DAC, (uint32_t)c);
        delayMicroseconds(100);
    }
    lt8722_write(ADDR_DAC, 0x00000000);
    lt8722_write(ADDR_COMMAND, CMD_STARTUP_PWM);
    delay(10);
    Serial.print("STATUS  0x"); Serial.println(lt8722_read(ADDR_STATUS), HEX);
    Serial.print("COMMAND 0x"); Serial.println(lt8722_read(ADDR_COMMAND), HEX);
    Serial.print("ILIMN   0x"); Serial.println(lt8722_read(ADDR_ILIMN), HEX);
    Serial.print("ILIMP   0x"); Serial.println(lt8722_read(ADDR_ILIMP), HEX);

    // start at zero TEC current
    updateLT8722_DAC(0.0); //was 2.5

    // PID configuration
    tecPID.SetOutputLimits(-DAC_MAX, DAC_MAX);
    tecPID.SetSampleTime(1000);
    tecPID.SetMode(AUTOMATIC);

    Serial.println("Enter target temperature:");
}
void runPolarityTest()
{
    Serial.println("=== Polarity test start ===");

    Serial.println("Commanding +1.0V");
    updateLT8722_DAC(1.0);
    delay(5000);
    Serial.print("Temp @ +1.0V: "); Serial.println(readTemperature(), 2);

    updateLT8722_DAC(0.0);
    delay(2000);

    Serial.println("Commanding -1.0V");
    updateLT8722_DAC(-1.0);
    delay(5000);
    Serial.print("Temp @ -1.0V: "); Serial.println(readTemperature(), 2);

    updateLT8722_DAC(0.0);
    Serial.println("=== Polarity test complete ===");
}
// =======================================================
// 12. MAIN LOOP
// =======================================================

void loop()
{
    Input = readTemperature();

    if (Serial.available() > 0)
    {
        String inString = Serial.readStringUntil('\n');

        if (inString == "t")
        {
            runPolarityTest();
            return;
        }

        float val = inString.toFloat();

        if (val > 5 && val < 80)   // sanity range
        {
            Setpoint = val;

            Serial.print("New Target: ");
            Serial.println(Setpoint);
        }
    }
    // STATUS every 5 s: a latched fault silently disables the output stage, and nothing
    // else on this line would show it. Fault bits are 4..10; bit 0 is PWM switching.
    static unsigned long lastStatus = 0;
    if (millis() - lastStatus > 5000)
    {
        lastStatus = millis();
        Serial.print("STATUS 0x");
        Serial.println(lt8722_read(ADDR_STATUS), HEX);
    }
    tecPID.Compute();

    double dacVoltage = Output;

    updateLT8722_DAC(dacVoltage);

    Serial.print(Input,2);
    Serial.print(",");

    Serial.print(Setpoint,2);
    Serial.print(",");

    Serial.println(dacVoltage,3);

    delay(200);
}