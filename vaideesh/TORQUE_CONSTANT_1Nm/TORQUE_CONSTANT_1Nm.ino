#include "Variables.h"


int fixed_pwm = 0;   // ← change this value to test different PWMs

void setup() {
  motorDataSetup();
  analogWriteResolution(12);
}

void loop() {
  if (Serial.available() > 0) {
    String input = Serial.readStringUntil('\n');
    input.trim();
    fixed_pwm = input.toInt();
    if (fixed_pwm > 0) {
      digitalWrite(ENABLE_M1, HIGH);
      digitalWrite(CW_M1, HIGH);
      analogWrite(PWM_M1, fixed_pwm);
      Serial.printf("PWM=%d\n", fixed_pwm);
    } else {
      digitalWrite(ENABLE_M1, LOW);
      analogWrite(PWM_M1, 0);
      Serial.println("Motor OFF");
    }
  }
}
//
//void setup() {
//  Serial.begin(115200);
//  analogReadResolution(12);
//  analogWriteResolution(12);
//  motorDataSetup();
//  // Motor starts disabled
//  digitalWrite(ENABLE_M1, LOW);
//  analogWrite(PWM_M1, 0);
// // Wait until Python opens the Serial port
////  while (!Serial);          // ← waits for USB Serial connection
////  delay(500);               // small extra delay to be safe
//  // Signal ready — Python waits for this
//  Serial.println("READY");
//  Serial.send_now();
//}
//void loop() {
//  for (int i = 0; i < NUM_STEPS; i++) {
//    int pwm = PWM_STEPS[i];
//
//    // Apply PWM
//    digitalWrite(ENABLE_M1, HIGH);
//    digitalWrite(CW_M1, HIGH);
//    analogWrite(PWM_M1, pwm);
//
//    // Tell Python which PWM is active
//    Serial.printf("PWM:%d\n", pwm);
//    Serial.send_now();
//
//    // Hold for STEP_DURATION — Python captures during this window
//    delay(STEP_DURATION);
//  }
//
//  // All steps done — stop motor
//  digitalWrite(ENABLE_M1, LOW);
//  analogWrite(PWM_M1, 0);
//  Serial.println("DONE");
//  Serial.send_now();
//
//  while (1);   // stop here — re-upload to run again
//}
