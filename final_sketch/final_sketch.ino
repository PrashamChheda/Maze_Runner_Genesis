#include <Servo.h>
// Declare the Servo pin
int servoPin1 = 3;
int servoPin2 = 5;
int i = 0;
// Create a servo object
Servo Servo1;
Servo Servo2;
float theta1,theta4;

void setup()
{
  
  Serial.begin(9600);
  Servo1.attach(servoPin1);
  Servo2.attach(servoPin2);
  Servo1.write(130);
  Servo2.write(50);
  delay(50);
}



void loop()
{
  // Servo1.write(180);
  // Servo2.write(0);
  //  Servo1.write(145);
  //  Servo2.write(165);
  //  delay(2000);
  // Servo1.write(156);
  //  Servo2.write(68);
  //  delay(2000);
  //   Servo1.write(112);
  //  Servo2.write(23);
  //  delay(2000);
  //   Servo1.write(17);
  //  Servo2.write(34);
  //  delay(2000);
    // Servo2.write(0);
  parsePython();
  //Serial.println(theta1);
  //Serial.println(theta4);
  Servo1.write(theta4);
  Servo2.write(theta1);
  delay(50);
  // Serial.println("a");
}

void parsePython()
{
 
  while(!Serial.available());
  //{//Serial.println(i);}
  theta1=Serial.readStringUntil(' ').toFloat();
  theta4 = Serial.readStringUntil('\n').toFloat();
  i++;

}