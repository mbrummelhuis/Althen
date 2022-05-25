
#define switch1 A6

#define switch2 A7

void setup() {
  // put your setup code here, to run once:
  Serial.begin(9600);
  pinMode(switch1, INPUT);
  pinMode(switch2, INPUT);
  
}

void loop() {
  // put your main code here, to run repeatedly:
  Serial.println(digitalRead(switch1));

}
