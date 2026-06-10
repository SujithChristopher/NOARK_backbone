//void controller()
//{
//  if (Serial.available() > 0) {
//    String inputString = Serial.readStringUntil('\n');
//    int commaIndex = inputString.indexOf(',');
//    if (commaIndex > 0) {
//      String firstValue = inputString.substring(0, commaIndex);
////      String secondValue = inputString.substring(commaIndex + 1);
//      Tou_1 = firstValue.toFloat();
////      Tou_2 = secondValue.toFloat();
//    }
//  }

//  
//  //Motor1
//  pwm_mot_1 = PWM_Value_1();
////  pwm_mot_1 = 600;
//  digitalWrite(ENABLE_M1, HIGH);
//  if (Tou_1 > 0) {
//    digitalWrite(CW_M1, HIGH);
//  } else {
//    digitalWrite(CW_M1, LOW);
//  }
//
//  //Motor2
//  pwm_mot_2 = PWM_Value_2();
//  digitalWrite(ENABLE_M2, HIGH);
//  if (Tou_2 > 0) {
//    digitalWrite(CW_M2, HIGH);
//  } else {
//    digitalWrite(CW_M2, LOW);
//  }
//  analogWrite(PWM_M1, pwm_mot_1);
//  analogWrite(PWM_M2, pwm_mot_2);
//}
