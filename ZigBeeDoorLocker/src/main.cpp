#include <Arduino.h>
#include <SoftwareSerial.h>

// 11 RX; 10 TX
SoftwareSerial Zigbee(11, 10); 

#define DELAY_TIME 5000
#define RELAY_PIN 4

const String str_on = "open";
const String str_off = "close";

unsigned long start = 0;

void setup() {
  Zigbee.begin(9600);
  Serial.begin(9600);
  pinMode(RELAY_PIN, OUTPUT);
}

void loop() {

  String mess = Zigbee.readStringUntil('\n');
  mess.trim();

  if (mess == str_on) {
    start = millis();
    Serial.println(mess);
    digitalWrite(RELAY_PIN, HIGH);
    Zigbee.write("OK!\n");
  }
  else if (mess == str_off)
  {
    start = millis();
    Serial.println(mess);
    digitalWrite(RELAY_PIN, LOW);
    Zigbee.write("OK!\n");
  }

  if (millis() - start > DELAY_TIME) {
    digitalWrite(RELAY_PIN, LOW);
  }


}