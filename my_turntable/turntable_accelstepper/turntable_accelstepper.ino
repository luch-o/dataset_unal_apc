// Include the AccelStepper Library
#include <AccelStepper.h>

// Define pin connections
const int DIR_PIN = 2;
const int STEP_PIN = 3;

/*
const float STEP_SIZE = 1.8;     // 1    full step
const float STEP_SIZE = 0.9;     // 1/2  microstep
const float STEP_SIZE = 0.45;    // 1/4  microstep
const float STEP_SIZE = 0.225;   // 1/8  microstep
const float STEP_SIZE = 0.1125;  // 1/16 microstep
const float STEP_SIZE = 0.05625; // 1/32 microstep
*/
const float STEP_SIZE = 0.1125;  // 1/16 microstep

// Define motor interface type
#define motorInterfaceType 1

// Define variables to control steps via serial
long steps;
float angle;
bool moving = false;

// Creates an instance
AccelStepper turntable(motorInterfaceType, STEP_PIN, DIR_PIN);

void setup() {
  // set the maximum speed, acceleration factor,
  // initial speed and the target position
  turntable.setMaxSpeed(1000);
  turntable.setAcceleration(100);
  turntable.setSpeed(200);

  // initialize serial
  Serial.begin(9600);
}

float steps2angle(int steps) {
  return steps * STEP_SIZE; 
}

void loop() {
  if (turntable.distanceToGo() == 0 && moving) {
    Serial.println("Turntable in position");
    moving = false;
  }
  
  // if a command is sent via serial and turntable is in the desired position
  if (Serial.available()) {    
    // read steps from serial
    steps = Serial.parseInt();
    
    // steps 0 means query current position
    if (steps == 0) {
      Serial.print("Current position: ");
      Serial.println(turntable.currentPosition() * STEP_SIZE);
    } 
    // otherwise move motor by angle degrees from current position
    else {
      moving = true;
      angle = steps2angle(steps);
      
      Serial.print("Steps: ");
      Serial.println(steps);
      Serial.print("Angle: ");
      Serial.println(angle);
      
      turntable.move(steps);
    }
  }
  
  // Move the motor one step
  turntable.run();
}
