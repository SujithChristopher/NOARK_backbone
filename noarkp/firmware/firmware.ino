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
  
  controller();
  updateEncoders();
  stream_data();

  // Serial.printf("Encoder_1:%.6f,Encoder_2:%.6f\n", MEnc1,MEnc2);
}
