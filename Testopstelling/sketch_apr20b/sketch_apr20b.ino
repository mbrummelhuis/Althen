#define switch1 A6
void setup() {
  // put your setup code here, to run once:
  Serial.begin(9600);
  pinMode(switch1, INPUT);
}

void loop() {
  // put your main code here, to run repeatedly:
  Serial.println(analogRead(switch1));
}
