// Motor 1 is left/roll (as seen from front)
// Motor 2 is top/pitch 

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

#define max_steps 30000

// Global variables
volatile bool FLTflag = false;

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
  digitalWrite(Dir1, HIGH);
  digitalWrite(Dir2, HIGH);

  // Set 1/16 microstepping
  digitalWrite(PinM0,LOW);
  digitalWrite(PinM1,LOW);
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
}

void loop() {
  

}

void homeMotors() {
  // Roll motor (1)
  while(analogRead(switch1) < 200) {
    Serial.println(analogRead(switch1));
    moveRollMotor();
  }

  digitalWrite(Dir1, !digitalRead(Dir1));
  delay(1000);
  while(analogRead(switch1) > 201) {
    Serial.println("Switch 1 high");
    moveRollMotor();
  }

  delay(2000);
  // Pitch motor (2)
  while(analogRead(switch2) < 200) {
    Serial.println(analogRead(switch2));
    movePitchMotor();
  }

  digitalWrite(Dir2, !digitalRead(Dir2));
  delay(1000);
  while(analogRead(switch2) > 201) {
    Serial.println(analogRead(switch2));
    movePitchMotor();
  }

  Serial.println("System is finished homing....");
}

void moveRollMotor() {
    digitalWrite(Step1, HIGH);
    digitalWrite(Step1, LOW);
    delayMicroseconds(500);
}
void movePitchMotor() {
    digitalWrite(Step2, HIGH);
    digitalWrite(Step2, LOW);
    delayMicroseconds(500);
}

void ISR_FLT() {
  FLTflag = true;
}
