int received;

void setup() {
  // initialize serial
  Serial.begin(9600);  

}

void loop() {
  if (Serial.available()) {
    received = Serial.parseInt();
    Serial.println(received);
  }

}
