#include <Servo.h>

// --- Servo Pins ---
#define PAN_SERVO_PIN 9
#define TILT_SERVO_PIN 10

// --- Servo Objects ---
Servo panServo;
Servo tiltServo;

// --- Servo Angles ---
int panAngle = 90;
int tiltAngle = 90;

// --- Serial Buffer ---
byte incomingData[3];
byte index = 0;

void setup() {
  Serial.begin(9600); // Match Python baud rate
  panServo.attach(PAN_SERVO_PIN);
  tiltServo.attach(TILT_SERVO_PIN);

  // Start at neutral position
  panServo.write(panAngle);
  tiltServo.write(tiltAngle);
}

void loop() {
  if (Serial.available()) {
    byte inByte = Serial.read();

    if (index == 0) {
      // Look for sync byte (255)
      if (inByte == 255) {
        incomingData[index++] = inByte;
      }
    } 
    else {
      // Store servo angle bytes
      incomingData[index++] = inByte;

      if (index == 3) {
        // We have a full packet: [255, pan, tilt]
        panAngle = constrain(incomingData[1], 0, 180);
        tiltAngle = constrain(incomingData[2], 0, 180);

        panServo.write(panAngle);
        tiltServo.write(tiltAngle);

        index = 0; // Reset for next packet
      }
    }
  }
}