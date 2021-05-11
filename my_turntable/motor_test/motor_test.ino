// Define pin connections & motor's steps per revolution
const int DIR_PIN = 2;
const int STEP_PIN = 3;
const int STEPS_PER_REVOLUTION = 200;
const int PERIOD = 2000; // us

void setup()
{
  // Declare pins as Outputs
  pinMode(STEP_PIN, OUTPUT);
  pinMode(DIR_PIN, OUTPUT);
}
void loop()
{
  // Set motor direction clockwise
  digitalWrite(DIR_PIN, HIGH);

  // Spin motor slowly
  for(int i = 0; i < STEPS_PER_REVOLUTION; i++)
  {
    digitalWrite(STEP_PIN, HIGH);
    delayMicroseconds(PERIOD/2);
    digitalWrite(STEP_PIN, LOW);
    delayMicroseconds(PERIOD/2);
  }
  delay(1000); // Wait a second
  
  // Set motor direction counterclockwise
  digitalWrite(DIR_PIN, LOW);

  // Spin motor quickly
  for(int i = 0; i < STEPS_PER_REVOLUTION; i++)
  {
    digitalWrite(STEP_PIN, HIGH);
    delayMicroseconds(PERIOD/4);
    digitalWrite(STEP_PIN, LOW);
    delayMicroseconds(PERIOD/4);
  }
  delay(1000); // Wait a second
}
