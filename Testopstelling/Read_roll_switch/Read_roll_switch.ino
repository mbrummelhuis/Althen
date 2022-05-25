void setup() {
  // put your setup code here, to run once:
  Serial.begin(9600);
  pinMode(A7, INPUT_PULLUP);
}

void loop() {
  // put your main code here, to run repeatedly:
  Serial.println(analogRead(A6));

}
