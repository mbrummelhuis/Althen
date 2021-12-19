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

#define SLEEP 2
#define FLT 3

#define mm_per_step 0.01

void setup() {
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
  pinMode(SLEEP, OUTPUT);
  pinMode(FLT, INPUT);

  // Set SLEEP HIGH
  digitalWrite(SLEEP, HIGH);

  // Set step pins low
  digitalWrite(Step1,LOW);
  digitalWrite(Step2,LOW);
  digitalWrite(Step3,LOW);

  // Set DIR pins low (low is out, high is in)
  digitalWrite(Dir1, LOW);
  digitalWrite(Dir2, LOW);
  digitalWrite(Dir3, LOW);
        
  // LLL Full step
  // HLL half step
  // LHL 1/4 step
  // HHL 1/8 step
  // LLH  1/16 step
  // HLH, LHH, HHH 1/32 step
  // Configure microstepping (low high low = 1/4 step)
  digitalWrite(PinM0,LOW);
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

  // TEST SEQUENCE HEREs
  // Move 10 steps in direction
  for(int i = 0; i<10;i++) {
      digitalWrite(Step1, HIGH);
      delayMicroseconds(100);
      digitalWrite(Step1, LOW);
      delayMicroseconds(100);
 }
    
}

void loop() {
  // put your main code here, to run repeatedly:
  // Move in and out
  while(true) {
    for(int n=0;n < 30000;n++){
      digitalWrite(Step1, HIGH);
      delayMicroseconds(100);
      digitalWrite(Step1, LOW);
      delayMicroseconds(100);
    }
  
    digitalWrite(Dir1, HIGH);
    digitalWrite(LedPin, HIGH);
    delay(1000);
    digitalWrite(LedPin, LOW);
    
    for(int n=0;n < 30000;n++){
      digitalWrite(Step1, HIGH);
      delayMicroseconds(100);
      digitalWrite(Step1, LOW);
      delayMicroseconds(100);
    }
    
    digitalWrite(Dir1, LOW);
    digitalWrite(LedPin, HIGH);
    delay(1000);
    digitalWrite(LedPin, LOW);
  
  }
}
