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

//#define PIN_A_M1   19
//#define PIN_B_M1   18
//
//#define PIN_A_M2   21
//#define PIN_B_M2   20
float MEnc1, MEnc2;
int encOffsetCount_1 = 0; //encoder's zero position
int encOffsetCount_2 = 0;
int enPPR = 6400;
float des_current_1 = 0.0;
float des_current_2 = 0.0;
float Torque_Const = 0.231;
float max_I = 5.0;
float min_I = 0.0;
float Tou_1 = 0.0;
float Tou_2 = 0.0;
int pwm_mot_1 = 0.0;
int pwm_mot_2 = 0.0;
unsigned long prevTime = 0;
float prevError = 0.0;
