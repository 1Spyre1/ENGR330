
#include <SoftwareSerial.h>
#include <Wire.h>
#include <Adafruit_Sensor.h>
#include <Adafruit_BME280.h>

// --- HM-10 SERIAL SETUP ---
// Connect HM-10 TX to D2, HM-10 RX to D3
SoftwareSerial bleSerial(2, 3); 

#define EN_PIN 8    // LOW: Driver enabled, HIGH: Driver disabled
#define STEP_PIN 9  // Step on the rising edge
#define DIR_PIN 10  // Set stepping direction


// --- I2C SENSOR SETUP ---
#define BME_I2C_ADDRESS 0x76
Adafruit_BME280 bme; 

// --- ACTUATOR SETUP ---
const int ledPin = LED_BUILTIN; 

// --- TIMING ---
unsigned long previousMillis = 0;
const long interval = 2000; // HM-10 works best with slightly longer intervals (2s)

int noOfSteps = 20000;           // Number of steps to move in each direction
int microSecondsDelay = 1000;  // Delay in microseconds between each step

bool motorFinished = false;

void setup() {
  Serial.begin(9600);   // Hardware Serial for Debugging
  bleSerial.begin(9600); // HM-10 Default Baud Rate
  
  pinMode(ledPin, OUTPUT);
  digitalWrite(ledPin, LOW); 

  pinMode(EN_PIN, OUTPUT);
  pinMode(STEP_PIN, OUTPUT);
  pinMode(DIR_PIN, OUTPUT);
  // Initialize pin states
  digitalWrite(EN_PIN, LOW);   // Enable the driver
  digitalWrite(DIR_PIN, HIGH);  // Set initial direction

  Serial.println("HM-10 Weather Node Initializing...");
  bleSerial.print("HM-10 Weather Node Initializing...");

  // 1. Initialize BME280 Sensor
  if (!bme.begin(BME_I2C_ADDRESS)) {
    Serial.println("Could not find BME280 sensor!");
    bleSerial.print("Could not find BME280 sensor!");
    while (1);
  }
  Serial.println("BME280 Initialized.");
  Serial.println("HM-10 Ready. Connect via BLE Serial App.");
  bleSerial.print("BME280 Initialized.");
  bleSerial.print("HM-10 Ready. Connect via BLE Serial App.");


  Serial.println("Starting Motor Task...");
  moveSteps(noOfSteps);
  digitalWrite(EN_PIN, HIGH);
  motorFinished = true;
  Serial.println("Motor Task Complete.");

  Serial.println("Entering Sensor Monitoring Mode...");
}

void loop() {

  if (!motorFinished) return;
  unsigned long currentMillis = millis();

  // --- 1. TRANSMIT SENSOR DATA ---
  // We send data as a comma-separated string for easy parsing on the phone side
  if (currentMillis - previousMillis >= interval) {
    previousMillis = currentMillis;

    float tempC = bme.readTemperature();
    float humidity = bme.readHumidity();
    float pressureHpa = bme.readPressure() / 100.0F;

    // Format: "T:24.5,H:45.2,P:1013.2"
    bleSerial.print("T:");
    bleSerial.print(tempC);
    bleSerial.print(",H:");
    bleSerial.print(humidity);
    bleSerial.print(",P:");
    bleSerial.println(pressureHpa);

    // Also log to local Serial Monitor for debugging
    Serial.print("Sent to BLE: T:"); Serial.print(tempC); 
    Serial.print(" H:"); Serial.print(humidity); 
    Serial.print(" P:"); Serial.println(pressureHpa);
  }

}

void moveSteps(int steps) {
 for (int i = 0; i < steps; i++) {
   digitalWrite(STEP_PIN, HIGH);
   delayMicroseconds(microSecondsDelay);
   digitalWrite(STEP_PIN, LOW);
   delayMicroseconds(microSecondsDelay);
 }
}
