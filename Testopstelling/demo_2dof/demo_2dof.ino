#include <AccelStepper.h>
#include <MultiStepper.h>
#include <EEPROM.h>

// Motor 1 is right (as seen from front)
// Motor 2 is top 
// Motor 3 is top

#define PinM0 12
#define PinM1 11
#define PinM2 10
#define Dir1 4
#define Step1 5
#define Dir2 6
#define Step2 7
#define Dir3 8
#define Step3 9

#define LedPin 13

#define RST 2
#define FLT 3

#define max_pitch 250000

#define mm_per_step 0.01

// Global variables
volatile bool FLTflag = false;
long targetpos = -320000; // negative is outward movement of actuators, positive in inward.

AccelStepper rollMotor(1, Step1, Dir1);
AccelStepper pitchMotor1(1, Step2, Dir2);
AccelStepper pitchMotor2(1, Step3, Dir3);

void setup() {
  // Serial monitor
  Serial.begin(9600);
  
  // Configure pins as outputs
  pinMode(PinM0,OUTPUT);
  pinMode(PinM1,OUTPUT);
  pinMode(PinM2,OUTPUT);
  pinMode(Dir1,OUTPUT);
  pinMode(Dir2,OUTPUT);
  pinMode(Dir3,OUTPUT);
  pinMode(Step1,OUTPUT);
  pinMode(Step2,OUTPUT);
  pinMode(Step3,OUTPUT);
  pinMode(LedPin,OUTPUT);
  pinMode(RST, OUTPUT);
  pinMode(FLT, INPUT);

  rollMotor.setMaxSpeed(5000.0);
  rollMotor.setAcceleration(1000.0);
  pitchMotor1.setMaxSpeed(5000.0);
  pitchMotor1.setAcceleration(1000.0);
  pitchMotor2.setMaxSpeed(5000.0);
  pitchMotor2.setAcceleration(1000.0);

  // Create interrupt on FLT pin
  attachInterrupt(digitalPinToInterrupt(FLT), ISR_FLT, FALLING);

  // Set SLEEP HIGH
  digitalWrite(RST, HIGH);

  // Set 1/32 microstepping
  digitalWrite(PinM0,HIGH);
  digitalWrite(PinM1,HIGH);
  digitalWrite(PinM2,HIGH);

  // LED startup sequence
  digitalWrite(LedPin, LOW);
  for(int i=0; i<15; i++){
    digitalWrite(LedPin, HIGH);
    delay(100);
    digitalWrite(LedPin, LOW);
    delay(100);
    }

  // Set zero positions for homing
  rollMotor.setCurrentPosition(0);
  pitchMotor1.setCurrentPosition(0);
  pitchMotor2.setCurrentPosition(0);

  //homeMotors();
  
  //moveMotors(0.0,15.6); // Move to 0 roll angle
  char printline[56];
  sprintf(printline, "%ld \t %ld \t %ld \n", rollMotor.currentPosition(), pitchMotor1.currentPosition(), pitchMotor2.currentPosition());
  Serial.println(printline);
  
  //moveMotors(20.0,0.0); // Move to 0 pitch angle
  sprintf(printline, "%ld \t %ld \t %ld \n", rollMotor.currentPosition(), pitchMotor1.currentPosition(), pitchMotor2.currentPosition());
  Serial.println(printline);
}



void loop() {  
  moveMotors(0.0,10.0);
  moveMotors(0.0,-20.0);
  moveMotors(0.0,10.0);
  finish();
}

void ISR_FLT() {
  FLTflag = true;
}

/************************************************************************
// Retracts rollMotor1 and pitchMotor fully and extends rollMotor2 and sets 
// position as 0.
************************************************************************/
void homeMotors() {

  long homing = 350000;
  
  rollMotor.moveTo(homing);
  pitchMotor1.moveTo(homing);
  pitchMotor2.moveTo(homing);

  bool done = false;
  int counter = 0;
  while(not done) {
    rollMotor.run();
    pitchMotor1.run();
    pitchMotor2.run();
    long dis1tg = rollMotor.distanceToGo();
    long dis2tg = pitchMotor1.distanceToGo();
    long dis3tg = pitchMotor2.distanceToGo();
    if(dis1tg == 0 && dis2tg == 0 && dis3tg == 0) {
      done = true;
    }
    counter++;
    if (counter == 1000) {
      char printline[56];
      sprintf(printline, "%ld \t %ld \t %ld \n", dis1tg, dis2tg, dis3tg);
      Serial.println(printline);
      counter = 0;
    }
  }
  
  // Set position at 0 when finished homing
  rollMotor.setCurrentPosition(0);
  pitchMotor1.setCurrentPosition(0);
  pitchMotor2.setCurrentPosition(0);
}

/************************************************************************
// Calculates number of steps needed (assumes 1/32 microstepping) to achieve 
// the input number of degrees of angle change and executes this number of
// steps on the steppers. 
************************************************************************/
void moveMotors(float pitch, float roll) {
  long pitchSteps = -6685.3 * pitch; // Slope of the pitch trendline
  long rollSteps = -10508.8 * roll; // Slope of the roll trendline

  rollMotor.moveTo(rollMotor.currentPosition()+rollSteps);
  pitchMotor1.moveTo(pitchMotor1.currentPosition()+pitchSteps);
  pitchMotor2.moveTo(pitchMotor1.currentPosition()+pitchSteps);
  
  bool done = false;
  while(not done) {
    rollMotor.run();
    pitchMotor1.run();
    pitchMotor2.run();
    long dis1tg = rollMotor.distanceToGo();
    long dis2tg = pitchMotor1.distanceToGo();
    long dis3tg = pitchMotor2.distanceToGo();
    if(dis1tg == 0 && dis2tg == 0 && dis3tg == 0) {
      done = true;
    }
  }
}

/************************************************************************
// Demo of test setup, moves forward and backward on both DoFs simultaneously
************************************************************************/
void demo(){  
  rollMotor.moveTo(targetpos);
  pitchMotor1.moveTo(targetpos);
  pitchMotor2.moveTo(targetpos);

  int counter = 0;
  bool done = false;
  while(not done) {
    rollMotor.run();
    pitchMotor1.run();
    pitchMotor2.run();
    long dis1tg = rollMotor.distanceToGo();
    long dis2tg = pitchMotor1.distanceToGo();
    long dis3tg = pitchMotor2.distanceToGo();

    counter++;
    if (counter == 1000) {
      char printline[56];
      sprintf(printline, "%ld \t %ld \t %ld \n", dis1tg, dis2tg, dis3tg);
      Serial.println(printline);
      counter = 0;
    }
    
    if(dis1tg == 0 && dis2tg == 0 && dis3tg == 0) {
      done = true;
      if(targetpos == 0) {
        targetpos = -320000;
      }
      else {
        targetpos = 0;
      }
    }
  }
}

/************************************************************************
// Blinks led to indicate script has finished. Blinks forever.
************************************************************************/
void finish() {
  while(true){
    delay(500);
    digitalWrite(LedPin, HIGH);
    delay(500);
    digitalWrite(LedPin, LOW);
  }
}
