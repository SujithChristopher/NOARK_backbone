#include <SPI.h>

#define CS0_PIN     D1
#define DRDY0_PIN   D2
#define RESET0_PIN  D0
// ================= LOAD CELLS CALIBRATION =================
#define CAL_FACTOR_1  -5628.5f
#define CAL_FACTOR_2  -5548.1f

float weight1 = 0.0f;
float weight2 = 0.0f;
float F_measured_x = 0.0f;
float F_measured_y = 0.0f;

long tare1 = 0;
long tare2 = 0;
// -----------------------
// Register writes (transaction already active during acquisition)
// -----------------------
static inline void writeRegister0_fast(uint8_t reg, uint8_t val) {
  digitalWrite(CS0_PIN, LOW);
  SPI.transfer((uint8_t)(0x40 | reg));
  SPI.transfer(val);
  digitalWrite(CS0_PIN, HIGH);
}

static inline void selectDiff0(uint8_t ainp, uint8_t ainn) {
  uint8_t mux = (uint8_t)(((ainp & 0x0F) << 4) | (ainn & 0x0F));
  writeRegister0_fast(0x11, mux);
}

static inline void ads0_start() { digitalWrite(CS0_PIN, LOW); SPI.transfer((uint8_t)0x08); digitalWrite(CS0_PIN, HIGH); }
static inline void ads0_stop()  { digitalWrite(CS0_PIN, LOW); SPI.transfer((uint8_t)0x0A); digitalWrite(CS0_PIN, HIGH); }

static inline int32_t readConv0_24b_signext(int pin1, int pin2) {
  selectDiff0(pin1, pin2);
  digitalWrite(CS0_PIN, LOW);
  unsigned long t0 = millis();
  while (digitalRead(DRDY0_PIN) == LOW)  { if (millis() - t0 > 50) { digitalWrite(CS0_PIN, HIGH); return 0; } }
  t0 = millis();
  while (digitalRead(DRDY0_PIN) != LOW)  { if (millis() - t0 > 50) { digitalWrite(CS0_PIN, HIGH); return 0; } }
  SPI.transfer((uint8_t)0x12);
  (void)SPI.transfer((uint8_t)0x00);
  uint8_t b2 = SPI.transfer((uint8_t)0x00);
  uint8_t b1 = SPI.transfer((uint8_t)0x00);
  uint8_t b0 = SPI.transfer((uint8_t)0x00);
  digitalWrite(CS0_PIN, HIGH);
  int32_t raw = ((int32_t)b2 << 16) | ((int32_t)b1 << 8) | (int32_t)b0;
  if (raw & 0x800000) raw |= 0xFF000000;
  return raw;
}

void doTare() {
  delay(200);
  int64_t sum1 = 0, sum2 = 0;
  unsigned long sampleCount = 0;
  unsigned long startTime = millis();

  while (millis() - startTime < 1000) {
    sum1 += readConv0_24b_signext(9, 10);
    sum2 += readConv0_24b_signext(1, 2);
    sampleCount++;
  }

  if (sampleCount > 0) {
    tare1 = (long)(sum1 / sampleCount);
    tare2 = (long)(sum2 / sampleCount);
  }
  Serial.println("TARE DONE");
}

void setup() {
  Serial.begin(115200);
  pinMode(CS0_PIN, OUTPUT);
  pinMode(DRDY0_PIN, INPUT);
  pinMode(RESET0_PIN, OUTPUT);
  digitalWrite(CS0_PIN, HIGH);
  digitalWrite(RESET0_PIN, HIGH);
  digitalWrite(RESET0_PIN, LOW); delayMicroseconds(10); digitalWrite(RESET0_PIN, HIGH);
  SPI.begin();
  SPI.beginTransaction(SPISettings(5000000, MSBFIRST, SPI_MODE1));
  writeRegister0_fast(0x02, 0x48);
  writeRegister0_fast(0x03, 0x0A);  // MODE1: sinc1 filter, 2400 SPS
  writeRegister0_fast(0x05, 0x00);
  writeRegister0_fast(0x06, 0x10);
  writeRegister0_fast(0x10, 0x05);
  ads0_start();
  delay(300);  // allow ADS1256 to settle before loop starts
}

void loop() {
  if (Serial.available() > 0) {
    char cmd = Serial.read();
    while (Serial.available()) Serial.read();
    if (cmd == 't' || cmd == 'T') doTare();
  }

  long adc1 = readConv0_24b_signext(9, 10);
  long adc2 = readConv0_24b_signext(1, 2);

  weight1 = ((float)adc1 - (float)tare1) / CAL_FACTOR_1;
  weight2 = ((float)adc2 - (float)tare2) / CAL_FACTOR_2;

  F_measured_x = weight1;
  F_measured_y = weight2;

  Serial.print(F_measured_x, 3);
  Serial.print(",");
  Serial.println(F_measured_y, 3);
}
