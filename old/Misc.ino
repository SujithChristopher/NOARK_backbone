void encoderSetup() {
  pinMode(ENC1A, INPUT_PULLUP);
  pinMode(ENC1B, INPUT_PULLUP);
  pinMode(ENC2A, INPUT_PULLUP);
  pinMode(ENC2B, INPUT_PULLUP);

  encOffsetCount_1 = MotorEnc1.read();
  encOffsetCount_2 = MotorEnc2.read();
}
void motorDataSetup()
{
  //Motor1 control pins
  pinMode(PWM_M1, OUTPUT);
  pinMode(ENABLE_M1, OUTPUT);
  pinMode(CW_M1, OUTPUT);
  digitalWrite(ENABLE_M1, LOW);

  //Motor2 control pins
  pinMode(PWM_M2, OUTPUT);
  pinMode(ENABLE_M2, OUTPUT);
  pinMode(CW_M2, OUTPUT);
  digitalWrite(ENABLE_M2, LOW);
}
void updateEncoders() {
  //Store past values
  theta1_past = theta1;
  theta2_past = theta2;

  //Read current angles
  theta1 = read_angle_motor1();
  theta2 = read_angle_motor2();
  
    //Angular Velocity ()
  dtheta1 = (theta1 - theta1_past);
  dtheta2 = (theta2 - theta2_past);

  //Read current time
  //  t_present = time_ellapsed - millis();
  //  dt = (t_present - t_past) / 1000.0 ;

  //  if (dt <= 0) return;
  //  t_past = t_present;
  pwm_1 = PWM_Value_1();
  pwm_2 = PWM_Value_2();
  cur_1 = des_current_1;
  cur_2 = des_current_2; 
}
float read_angle_motor1()
{
  long newPosition = MotorEnc1.read() - encOffsetCount_1;
  return (360.0 * newPosition / (enPPR * 4));
}
float read_angle_motor2()
{
  long newPosition = MotorEnc2.read() - encOffsetCount_2;
  return (360.0 * newPosition / (enPPR * 4));
}
int PWM_Value_1() {
  des_current_1 = Tou_1 / Torque_Const + kd *(dtheta1);
  int pwm_1 = map(abs(des_current_1), min_I, max_I, MINPWM, MAXPWM);
  return pwm_1;
}
int PWM_Value_2() {
  des_current_2 = Tou_2 / Torque_Const + kd *(dtheta2);
  int pwm_2 = map(abs(des_current_2), min_I, max_I, MINPWM, MAXPWM);
  return pwm_2;
}
