// TORQUE_CONSTANT_1Nm.ino
// Receives a PWM integer over Serial from find_kt.py
// and directly applies it to Motor 1.
// Motor 2 is always disabled.

#include "Variables.h"

int fixed_pwm = 0;   // updated via Serial

void setup() {
  Serial.begin(115200);
  analogWriteResolution(12);
  motorDataSetup();

  // Start with motor stopped
  digitalWrite(ENABLE_M1, HIGH);
  digitalWrite(CW_M1, HIGH);     // fixed CW direction throughout test
  analogWrite(PWM_M1, 0);
  digitalWrite(ENABLE_M2, LOW);  // motor 2 always off
}

void loop() {
  // Read PWM from Python script
  if (Serial.available() > 0) {
    String input = Serial.readStringUntil('\n');
    input.trim();
    int received_pwm = input.toInt();

    // Clamp to safe range
    if (received_pwm < 0)    received_pwm = 0;
    if (received_pwm > MAXPWM) received_pwm = MAXPWM;

    fixed_pwm = received_pwm;
    analogWrite(PWM_M1, fixed_pwm);

    // Confirm back to Python (optional, for debugging)
    Serial.printf("PWM_SET:%d\n", fixed_pwm);
    Serial.send_now();
  }

  delay(5);
}
