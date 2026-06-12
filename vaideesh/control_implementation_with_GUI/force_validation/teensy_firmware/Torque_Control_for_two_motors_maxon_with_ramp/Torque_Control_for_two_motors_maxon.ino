#include <Encoder.h>
#include "Variables.h"

void setup() {
  Serial.begin(115200);
  Serial.setTimeout(5);
  analogReadResolution(12);
  analogWriteResolution(12);
  encoderSetup();
  motorDataSetup();

  // Auto-tare on startup
  encOffsetCount_1 = MotorEnc1.read();
  encOffsetCount_2 = MotorEnc2.read();
  theta1 = 0.0; theta2 = 0.0;
  Serial.println("ENC_RESET");
}
void loop() {
  time_ellapsed = millis();
  updateEncoders();
  controller();
   Serial.printf("%.2f,%.2f\n", theta1,theta2);
   Serial.send_now();
//  delay(2);
}
