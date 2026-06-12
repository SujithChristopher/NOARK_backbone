void controller()
{
  if (Serial.available() > 0) {
    String inputString = Serial.readStringUntil('\n');
    inputString.trim();
//    Encoder reset command
     // Encoder reset command
    if (inputString == "R") {
      encOffsetCount_1 = MotorEnc1.read();
      encOffsetCount_2 = MotorEnc2.read();
      theta1 = 0.0; theta2 = 0.0;
      Serial.println("ENC_RESET");
      return;
    }
      int commaIndex = inputString.indexOf(',');
      if (commaIndex > 0) {
        String firstValue = inputString.substring(0, commaIndex);
        String secondValue = inputString.substring(commaIndex + 1);
        Tou_1_target = firstValue.toFloat();
        Tou_2_target = secondValue.toFloat();
//      Tou_1 = firstValue.toFloat();
//      Tou_2 = secondValue.toFloat();
    }
  }
   // ── time-based slew: change Tou by at most TOU_RATE per second ──
      unsigned long now = micros();
      if (ramp_t_past == 0) ramp_t_past = now;        // init on first pass
      float dt = (now - ramp_t_past) / 1000000.0f;    // seconds since last loop
      ramp_t_past = now;
      float maxStep = TOU_RATE * dt;                  // max change allowed this loop
    
      float d1 = Tou_1_target - Tou_1;
      if (d1 >  maxStep) d1 =  maxStep;
      if (d1 < -maxStep) d1 = -maxStep;
      Tou_1 += d1;
    
      float d2 = Tou_2_target - Tou_2;
      if (d2 >  maxStep) d2 =  maxStep;
      if (d2 < -maxStep) d2 = -maxStep;
      Tou_2 += d2;

  // Motor1
pwm_mot_1 = PWM_Value_1();

if (Tou_1 > 0) {
    digitalWrite(CW_M1, HIGH);
} else {
    digitalWrite(CW_M1, LOW);
}

if (abs(Tou_1) < 0.001) {
    digitalWrite(ENABLE_M1, LOW);
    analogWrite(PWM_M1, 0);
} else {
    digitalWrite(ENABLE_M1, HIGH);
    analogWrite(PWM_M1, pwm_mot_1);
}

  //Motor2
  pwm_mot_2 = PWM_Value_2();
  digitalWrite(ENABLE_M2, HIGH);
  if (Tou_2 > 0) {
    digitalWrite(CW_M2, HIGH);
  } else {
    digitalWrite(CW_M2, LOW);
  }
  if (abs(Tou_2) < 0.001) {
    digitalWrite(ENABLE_M2, LOW);
    analogWrite(PWM_M2, 0);
} else {
    digitalWrite(ENABLE_M2, HIGH);
    analogWrite(PWM_M2, pwm_mot_2);
}
//  analogWrite(PWM_M1, pwm_mot_1);
//  analogWrite(PWM_M2, pwm_mot_2);
}
