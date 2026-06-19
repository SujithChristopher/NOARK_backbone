#include <Adafruit_TinyUSB.h>
#include <SPI.h>

#define CS0_PIN     D1
#define DRDY0_PIN   D2
#define RESET0_PIN  D0
//#define sample_pin  D3

// ================= LOAD CELLS CALIBRATION =================
#define CAL_FACTOR_1  -5628.5f  // mg per ADC unit for LoadCell1 (CALIBRATE THIS)
#define CAL_FACTOR_2  -5548.1f  // mg per ADC unit for LoadCell2 (CALIBRATE THIS)

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

// Global state
int count = 0;
static uint32_t lastPacket = 0;

static inline void selectDiff0(uint8_t ainp, uint8_t ainn) {
  uint8_t mux = (uint8_t)(((ainp & 0x0F) << 4) | (ainn & 0x0F));
  writeRegister0_fast(0x11, mux);
}


void doTare() {
  delay(200);
  int64_t sum1 = 0, sum2 = 0;
  long count1 = 0, count2 = 0;
  unsigned long startTime = millis();

  while (millis() - startTime < 1000) {
    long v1 = readConv0_24b_signext(9, 10);
    if (v1 != 0) { sum1 += v1; count1++; }
    long v2 = readConv0_24b_signext(1, 2);
    if (v2 != 0) { sum2 += v2; count2++; }
  }

  if (count1 > 0) tare1 = (long)(sum1 / count1);
  if (count2 > 0) tare2 = (long)(sum2 / count2);

  Serial.println("TARE DONE");
}
// ======================================================================================
static inline void ads0_start() { digitalWrite(CS0_PIN, LOW); SPI.transfer((uint8_t)0x08);  digitalWrite(CS0_PIN, HIGH); }
static inline void ads0_stop()  { digitalWrite(CS0_PIN, LOW); SPI.transfer((uint8_t)0x0A);  digitalWrite(CS0_PIN, HIGH); }

// =============================== Fast conversion reads ===============================
static inline int32_t readConv0_24b_signext(int pin1, int pin2) {
  selectDiff0(pin1, pin2);
  digitalWrite(CS0_PIN, LOW);
  while (digitalRead(DRDY0_PIN) == LOW){}
  while (digitalRead(DRDY0_PIN) != LOW){}
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



static inline void startAcqHard(int pin1, int pin2) {
  SPI.beginTransaction(SPISettings(5000000, MSBFIRST, SPI_MODE1));
  selectDiff0(pin1, pin2);
  ads0_start();
}

// ---------------------------------------------------------------------------
// Setup
// ---------------------------------------------------------------------------
void setup() {
  Serial.begin(115200);
  while(!Serial) delay(10); // Wait for USB Serial monitor to open

  pinMode(CS0_PIN, OUTPUT);
  pinMode(DRDY0_PIN, INPUT);
  pinMode(RESET0_PIN, OUTPUT);
//  pinMode(sample_pin, OUTPUT);

  digitalWrite(CS0_PIN, HIGH);
  digitalWrite(RESET0_PIN, HIGH);
  digitalWrite(RESET0_PIN, LOW); delayMicroseconds(10); digitalWrite(RESET0_PIN, HIGH);
//  digitalWrite(sample_pin, LOW);
  SPI.begin();
  
  SPI.beginTransaction(SPISettings(5000000, MSBFIRST, SPI_MODE1));
  writeRegister0_fast(0x02, 0x48);  // DOR 40kSPS
  writeRegister0_fast(0x03, 0x01);  // set to 21 for CHOP mode
  writeRegister0_fast(0x05, 0x00);  // set to 40 for status byte
  writeRegister0_fast(0x06, 0x10);  // Internal Reference
  writeRegister0_fast(0x10, 0x05);  // Gain 32 
  ads0_start();
}

// ---------------------------------------------------------------------------
// Main Loop - Every 20ms, prints 1 line of text containing metadata + 4 samples
// ---------------------------------------------------------------------------
void loop() {
// digitalWrite(sample_pin, HIGH);
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
  Serial.flush();   // force USB packet sent immediately, no batching
//  digitalWrite(sample_pin, LOW);
  
}
