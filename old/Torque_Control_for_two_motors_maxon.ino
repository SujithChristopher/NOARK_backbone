#include <Encoder.h>
#include "Variables.h"

void setup() {
  Serial.begin(115200);
  analogReadResolution(12);
  analogWriteResolution(12);
  encoderSetup();
  motorDataSetup();
}
void loop() {
  time_ellapsed = millis();
  updateEncoders();
  controller();
   Serial.printf("%.2f,%.2f\n", theta1,theta2);
//  Serial.printf("Present1:%.2f,Past1:%.2f,omega_1:%.2f,cur_1:%.2f\n", theta1, theta1_past, dtheta1,cur_1);
  delay(2);
}
