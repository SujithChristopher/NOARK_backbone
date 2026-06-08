#include "Arduino.h"

#define MINPWM          410 //10% of 4095
#define MAXPWM          3686 //90% of 4095


#define CW_M1           13
#define PWM_M1          15
#define ENABLE_M1       14

#define CW_M2           10
#define PWM_M2          12
#define ENABLE_M2       11

float pwm_1,pwm_2;

float cur_1 = 0 , cur_2 = 0;
float kd = 0.1;
//float Torque_Const = 0.231;
float max_I = 5.0; 
float min_I = 0.0;
int pwm_mot_1 = 0.0;
int pwm_mot_2 = 0.0;
//int fixed_pwm = 0;

// ── PWM steps — must match find_kt.py ──────────────────────
const int PWM_STEPS[]   = {774, 1138, 1502, 1866, 2230, 2594, 2958, 3322, 3686};
const int NUM_STEPS     = 9;
const int STEP_DURATION = 7000;   // ms per step — must match Python STEP_DURATION
const int SETTLE_MS     = 2000;   // ms to settle before Python captures
// ───────────────────────────────────────────────────────────
