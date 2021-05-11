// Define pin connections & motor's steps per revolution
const int DIR_PIN = 2;
const int STEP_PIN = 3;
// period for succesive steps
const long PERIOD = 2000;

/*
const float STEP_SIZE = 1.8;     // 1    full step
const float STEP_SIZE = 0.9;     // 1/2  microstep
const float STEP_SIZE = 0.45;    // 1/4  microstep
const float STEP_SIZE = 0.225;   // 1/8  microstep
const float STEP_SIZE = 0.1125;  // 1/16 microstep
const float STEP_SIZE = 0.05625; // 1/32 microstep
*/

const float STEP_SIZE = 0.1125;  // 1/16 microstep
int steps;
int angle;
int posCounter = 0;

void setup() {
  // Declare pins as Outputs
  pinMode(STEP_PIN, OUTPUT);
  pinMode(DIR_PIN, OUTPUT);

  // Set motor direction by default
  digitalWrite(DIR_PIN, HIGH);

  // Start serial
  Serial.begin(9600);
}

int angle2steps(int angle) {
  return floor(angle / STEP_SIZE);
}

void loop() {
  if (Serial.available()) {
    // read serial
    angle = Serial.parseInt();

    // query current position
    if (angle == 0) {
      Serial.print("Posición: ");
      Serial.println(posCounter * STEP_SIZE);
    }
    // move motor incrementally angle degrees
    else {
      // change motor direction if angle is negative
      if (angle < 0)
        digitalWrite(DIR_PIN, LOW);
      else {
        digitalWrite(DIR_PIN, HIGH);
      }
      
      steps = angle2steps(angle);
      posCounter += steps;

      // serial output
      Serial.print("angle: ");
      Serial.println(angle);
      Serial.print("Steps: ");
      Serial.println(steps);
      Serial.print("Actual angle: ");
      Serial.println(steps * STEP_SIZE);

      for (int i = 0; i < abs(steps); i++) {
        digitalWrite(STEP_PIN, HIGH);
        delayMicroseconds(PERIOD / 2);
        digitalWrite(STEP_PIN, LOW);
        delayMicroseconds(PERIOD / 2);
      }
    }
  }
}
