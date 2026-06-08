void motorDataSetup()
{
  //Motor1 control pins
  pinMode(PWM_M1, OUTPUT);
  pinMode(ENABLE_M1, OUTPUT);
  pinMode(CW_M1, OUTPUT);
  digitalWrite(ENABLE_M1, LOW);
}
//  //Motor2 control pins
//  pinMode(PWM_M2, OUTPUT);
//  pinMode(ENABLE_M2, OUTPUT);
//  pinMode(CW_M2, OUTPUT);
//  digitalWrite(ENABLE_M2, LOW);
//}
//int PWM_Value_1() {
////  des_current_1 = Tou_1 / Torque_Const + kd *(dtheta1);
//  tou_1_actual = des_current_1 * Torque_Const;
////  des_current_1 = Tou_1 / Torque_Const;
//  int pwm_1 = map(abs(des_current_1), min_I, max_I, MINPWM, MAXPWM);
//  return pwm_1;
//}
//int PWM_Value_2() {
////  des_current_2 = Tou_2 / Torque_Const + kd *(dtheta2);
////  tou_2_actual = des_current_2 * Torque_Const;
////  des_current_2 = Tou_2 / Torque_Const;
//  int pwm_2 = map(abs(des_current_2), min_I, max_I, MINPWM, MAXPWM);
//  return pwm_2;
//}
//void loop() {
//  if (Serial.available() > 0) {
//    String input = Serial.readStringUntil('\n');
//    input.trim();
//    int received_pwm = input.toInt();
//
//    if (received_pwm < 0)      received_pwm = 0;
//    if (received_pwm > MAXPWM) received_pwm = MAXPWM;
//
//    fixed_pwm = received_pwm;
//
//    if (fixed_pwm == 0) {
//      digitalWrite(ENABLE_M1, LOW);   // disable motor when PWM=0
//      analogWrite(PWM_M1, 0);
//    } else {
//      digitalWrite(ENABLE_M1, HIGH);  // enable only when PWM > 0
//      digitalWrite(CW_M1, HIGH);      // fixed CW direction
//      analogWrite(PWM_M1, fixed_pwm);
//    }
//    Serial.printf("PWM_SET:%d\n", fixed_pwm);
//    Serial.send_now();m
//  }
//  delay(5);
//}
