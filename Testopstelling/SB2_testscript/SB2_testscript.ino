#define PinM0 12
#define PinM1 11
#define PinM2 10
#define Dir1 4
#define Step1 5
#define Dir2 6
#define Step2 7
#define switch1 A6
#define switch2 A7
#define LedPin 13
#define RST 2
#define FLT 3

// Deg-steps mapping (from Excel trendline)
#define slope_roll 3570.1
#define intercept_roll 39061

// #define slope_pitch 1606.7
// #define intercept_pitch 59340
#define slope_pitch 1624
#define intercept_pitch 57498

// Global variables
volatile bool FLTflag = false;

const int max_steps_roll = 100000; // 32 microstep
const int max_steps_pitch = 100000; // 32 microstep

const unsigned long SECOND = 1000;
const unsigned long MINUTE = 60*SECOND;
const unsigned long HOUR = 60*MINUTE;

const unsigned long INTERVAL = 4*HOUR;

long steps_roll;
long steps_pitch;

void setup() {
// Serial monitor
  Serial.begin(9600);
  
  // Configure pins as outputs
  pinMode(PinM0,OUTPUT);
  pinMode(PinM1,OUTPUT);
  pinMode(PinM2,OUTPUT);
  pinMode(Dir1,OUTPUT);
  pinMode(Dir2,OUTPUT);
  pinMode(Step1,OUTPUT);
  pinMode(Step2,OUTPUT);
  pinMode(LedPin,OUTPUT);
  pinMode(RST, OUTPUT);
  
  pinMode(FLT, INPUT);
  pinMode(switch1, INPUT);
  pinMode(switch2, INPUT);

  // Create interrupt on FLT pin
  attachInterrupt(digitalPinToInterrupt(FLT), ISR_FLT, FALLING);

  // Set RESET HIGH
  digitalWrite(RST, HIGH);

  // Set dirs high
  digitalWrite(Dir1, HIGH); // High dir means moving in direction of 0
  digitalWrite(Dir2, HIGH);

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

  homeMotors();
  movePitchDegrees(0.0);
  delay(1*HOUR);

  while(true) {
    // Move to 0 position
    movePitchDegrees(-10.0);
    delay(4*HOUR);
  
    movePitchDegrees(-5.0);
    delay(3*HOUR);
    
    movePitchDegrees(-2.0);
    delay(3*HOUR);

    movePitchDegrees(-1.0);
    delay(3*HOUR);
  
    movePitchDegrees(0.0);
    delay(3*HOUR);
    
    movePitchDegrees(1.0);
    delay(3*HOUR);
  
    movePitchDegrees(2.0);
    delay(3*HOUR);
  
    movePitchDegrees(5.0);
    delay(3*HOUR);
  
    movePitchDegrees(10.0);
    delay(3*HOUR);
  }
}

void loop() {
  digitalWrite(LedPin, HIGH);
  delay(100);
  digitalWrite(LedPin, LOW);
  delay(100);
}

void movePitchDegrees(float deg) {
  Serial.print("Moving pitch to ");
  Serial.print(deg);
  Serial.println(" degrees");
  long int pitch_step_goal = slope_pitch * deg + intercept_pitch;
  long int pitch_steps_to_go = pitch_step_goal - steps_pitch;
  Serial.print("Steps pitch goal: ");
  Serial.println(pitch_step_goal); 
  
  Serial.print("Steps pitch: ");
  Serial.println(steps_pitch);  
  Serial.print("Pitch steps to go: ");
  Serial.println(pitch_steps_to_go);
  if (pitch_steps_to_go < 0) {
    digitalWrite(Dir2, HIGH);
    pitch_steps_to_go = pitch_steps_to_go * -1;
  }
  else if (pitch_steps_to_go >= 0) {digitalWrite(Dir2, LOW);}
  
  while(pitch_steps_to_go>0) {
    movePitchMotor();
    --pitch_steps_to_go;
  }
  Serial.println("Finished moving pitch");
  return;
}

void homeMotors() {
  // Pitch motor (2)
  Serial.println("Homing pitch backwards...");
  while(analogRead(switch2) < 200) {
    movePitchMotor();
  }

  digitalWrite(Dir2, !digitalRead(Dir2));
  delay(1000);
  Serial.println("Homing pitch forwards...");
  while(analogRead(switch2) > 201) {
    movePitchMotor();
  }
  steps_pitch = 0;

  Serial.println("System has finished homing....");

}

void movePitchMotor() {
    digitalWrite(Step2, HIGH);
    digitalWrite(Step2, LOW);
    if(digitalRead(Dir2)==HIGH) {--steps_pitch;}
    else {steps_pitch++;}
    delayMicroseconds(250);
}

void ISR_FLT() {
  FLTflag = true;
}
