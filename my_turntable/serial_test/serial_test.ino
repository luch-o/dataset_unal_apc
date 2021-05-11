int data;

void setup()
{
   Serial.begin(9600);
   Serial.setTimeout(50);
}

void loop()
{
   if (Serial.available())
   {
      data = Serial.parseInt();
      Serial.println(data);
   }
}
