#include <HX711.h>

#define CALIBRATION_FACTOR_X  11224
#define CALIBRATION_FACTOR_Y  11395

#define LOADCELL_DOUT_X  2
#define LOADCELL_SCK_X   3
#define LOADCELL_DOUT_Y  6
#define LOADCELL_SCK_Y   7

HX711 scale_X;
HX711 scale_Y;

String inputBuffer = "";

void setup() {
  Serial.begin(115200);

  scale_X.begin(LOADCELL_DOUT_X, LOADCELL_SCK_X);
  scale_Y.begin(LOADCELL_DOUT_Y, LOADCELL_SCK_Y);
  scale_X.set_scale(CALIBRATION_FACTOR_X);
  scale_Y.set_scale(CALIBRATION_FACTOR_Y);

  if (!scale_X.wait_ready_timeout(2000)) Serial.println("X not found!");
  if (!scale_Y.wait_ready_timeout(2000)) Serial.println("Y not found!");

  Serial.println("Taring... remove all load.");
  delay(1000);
  scale_X.tare();
  scale_Y.tare();
  Serial.println("Tare done.");
  Serial.println("Type force value in N and press Enter.");
  Serial.println("Type t + Enter to re-tare.");
  Serial.println("force_N,lc_x_N,lc_y_N");
}

void loop() {
  while (Serial.available() > 0) {
    char c = Serial.read();

    if (c == '\n' || c == '\r') {
      inputBuffer.trim();

      if (inputBuffer == "t" || inputBuffer == "T") {
        Serial.println("Taring...");
        scale_X.tare();
        scale_Y.tare();
        Serial.println("Tare done.");

      } else if (inputBuffer.length() > 0) {
        float entered_force = inputBuffer.toFloat();
        float lc_x = scale_X.get_units(1);
        float lc_y = scale_Y.get_units(1);

        Serial.print(entered_force, 3);
        Serial.print(",");
        Serial.print(lc_x, 3);
        Serial.print(",");
        Serial.println(lc_y, 3);
      }

      inputBuffer = "";

    } else {
      inputBuffer += c;
    }
  }
}
