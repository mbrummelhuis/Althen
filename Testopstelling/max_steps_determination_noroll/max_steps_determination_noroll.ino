// Motor 1 is left/roll (as seen from front)
// Motor 2 is top/pitch 

#define PinM0 12
#define PinM1 11
#define PinM2 10

#define Dir2 6
#define Step2 7

#define switch1 A6
#define switch2 A7

#define LedPin 13
#define RST 2
#define FLT 3


#define max_steps_roll 100000 // 32 microstep
#define max_steps_pitch 100000 // 32 microstep

const unsigned long SECOND = 1000;
const unsigned long HOUR = 3600*SECOND;

// Global variables
volatile bool FLTflag = false;

void setup() {
 // Serial monitor
  Serial.begin(9600);
  
  // Configure pins as outputs
  pinMode(PinM0,OUTPUT);
  pinMode(PinM1,OUTPUT);
  pinMode(PinM2,OUTPUT);
  pinMode(Dir2,OUTPUT);
  pinMode(Step2,OUTPUT);
  pinMode(LedPin,OUTPUT);
  pinMode(RST, OUTPUT);
  
  pinMode(FLT, INPUT);
  pinMode(switch2, INPUT);

  // Create interrupt on FLT pin
  attachInterrupt(digitalPinToInterrupt(FLT), ISR_FLT, FALLING);

  // Set RESET HIGH
  digitalWrite(RST, HIGH);

  // Set dirs high
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
  
  unsigned long steps_counter = 0;
  
  while(analogRead(switch2)<200) {
    movePitchMotor();
    steps_counter +=1;
  }
  Serial.println("Number of steps (pitch): \r\n");
  Serial.println(steps_counter);
}

void loop() {
    digitalWrite(LedPin, HIGH);
    delay(100);
    digitalWrite(LedPin, LOW);
    delay(100);

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

  Serial.println("System is finished homing....");
}

void movePitchMotor() {
    digitalWrite(Step2, HIGH);
    digitalWrite(Step2, LOW);
    delayMicroseconds(250);
}

void ISR_FLT() {
  FLTflag = true;
}
