#include "Arduino.h"

#define MINPWM          410 //10% of 4095
#define MAXPWM          3686 //90% of 4095


#define CW_M1           13
#define PWM_M1          15
#define ENABLE_M1       14

#define CW_M2           10
#define PWM_M2          12
#define ENABLE_M2       11

#define ENC1A           19
#define ENC1B           18

#define ENC2A           20
#define ENC2B           21

#define ENC1MAXCOUNT    4*4096
#define ENC1COUNT2DEG   0.25f*0.00166f
#define ENC2MAXCOUNT    4*4096
#define ENC2COUNT2DEG   0.25f*0.00166f


Encoder MotorEnc1(ENC1A, ENC1B);
long _enccount1;

Encoder MotorEnc2(ENC2A, ENC2B);
long _enccount2;

float pwm_1,pwm_2;

//Angular velocity
float omega_1 = 0 ;
float omega_2 = 0;
//Encoder angles(degrees)
float theta1 = 0, theta1_past = 0, dtheta1 = 0;
float theta2 = 0, theta2_past = 0, dtheta2 = 0;
//Time 
unsigned long t_present = 0;
unsigned long t_past = 0;
float dt = 0;
float cur_1 = 0 , cur_2 = 0;
int encOffsetCount_1 = 0; //encoder's zero position
int encOffsetCount_2 = 0;
int enPPR = 6400;
float kd = 0.1;
float time_ellapsed;
float des_current_1 = 0.0;
float des_current_2 = 0.0;
float Torque_Const_1 = 0.2136;
float Torque_Const_2 = 0.2064;
float max_I = 5.0; 
float min_I = 0.0;
float Tou_1 = 0.0;
float Tou_2 = 0.0;
float Tou_1_target = 0.0;
float Tou_2_target = 0.0;
const float TOU_RATE = 1.0f;   // Nm/sec — ramps 0→max torque (~0.8 Nm) in <1 s
unsigned long ramp_t_past = 0;
int pwm_mot_1 = 0.0;
int pwm_mot_2 = 0.0;
float current_theta1 = 0.0;
float current_theta2 = 0.0;
