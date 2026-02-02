#include <ESP8266WiFi.h>
#include <ESP8266HTTPClient.h>
#include <Wire.h>
#include <HTU21D.h>
#include <SDS011.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>
#include <math.h>
#include <EEPROM.h>

const char* ssid = "Vivo V23 Pro";
const char* password = "Yash9130";
const char* serverUrl = "http://10.96.209.108:5000/api/sensor";
const char* nodeId = "node1";

#define SDS_RX 14
#define SDS_TX 12
#define MQ135_PIN A0
#define BUZZER_PIN 13

#define EEPROM_CALIBRATED_FLAG 0
#define EEPROM_R0_ADDR 1
#define EEPROM_CALIB_TIME_ADDR 5
#define EEPROM_SIZE 512

boolean calibrationMode = false;
unsigned long calibrationStartTime = 0;
const unsigned long warmupDuration = 120000;

const float MQ135_RLOAD = 10000.0;
float MQ135_R0 = 7600.0;
const float MQ135_CO2_A = 116.6020682;
const float MQ135_CO2_B = -2.769034857;

HTU21D thSen;
SDS011 sds;

#define SCREEN_WIDTH 128
#define SCREEN_HEIGHT 64
#define OLED_RESET    -1
Adafruit_SSD1306 display(SCREEN_WIDTH, SCREEN_HEIGHT, &Wire, OLED_RESET);

float PM25, PM10;
int error;
float temp, humidity;
int mq135Raw = 0;
float mq135Rs = 0.0;
float mq135CO2ppm = 0.0;
float mq135NO2Index = 0.0;
unsigned long lastSendTime = 0;
const unsigned long sendInterval = 5000;

bool buzzerEnabled = true;
unsigned long lastBuzzerTime = 0;
const unsigned long buzzerInterval = 30000;

boolean isWarmingUp = true;
boolean isCalibrated = false;
float calibratedR0 = 7600.0;

int currentAQI = 0;
const char* aqi_labels[] = {"Good", "Moderate", "USG", "Bad", "V.Bad", "Hazard"};

void initDisplay() {
  if (!display.begin(SSD1306_SWITCHCAPVCC, 0x3C)) {
    Serial.println("SSD1306 allocation failed");
    for (;;) {
      delay(100);
    }
  }
  display.clearDisplay();
  display.setTextColor(SSD1306_WHITE);
  display.setTextSize(1);
  display.display();
}

void showMessage(const String& line1, const String& line2 = "") {
  display.clearDisplay();
  display.setTextSize(1);
  display.setTextColor(SSD1306_WHITE);
  display.setCursor(0, 0);
  display.println(line1);
  if (line2.length() > 0) {
    display.setCursor(0, 12);
    display.println(line2);
  }
  display.display();
}

void setup() {
  Serial.begin(115200);
  delay(100);
  Serial.println("ESP8266 Sensor Reader Started");
  
  EEPROM.begin(EEPROM_SIZE);
  loadCalibrationData();

  pinMode(BUZZER_PIN, OUTPUT);
  digitalWrite(BUZZER_PIN, LOW);
  
  initDisplay();
  showMessage("Initializing...", "Starting sensors");
  
  delay(500);
  thSen.begin();
  sds.begin(SDS_RX, SDS_TX);
  
  // Show calibration status
  displayCalibrationStatus();
  
  calibrationStartTime = millis();
  
  showMessage("WiFi connecting..", "");
  connectToWiFi();
}

void loop() {
  if (Serial.available()) {
    handleSerialCommand();
  }
  
  if (WiFi.status() != WL_CONNECTED) {
    connectToWiFi();
  }
  
  error = sds.read(&PM25, &PM10);
  
  if (!thSen.measure()) {
    Serial.println("HTU21D sensor error");
  }
  
  if (!error && thSen.measure()) {
    temp = thSen.getTemperature();
    humidity = thSen.getHumidity();
    readMQ135();
    
    checkWarmupStatus();
    
    Serial.print("[");
    Serial.print(nodeId);
    Serial.print("] Temp: ");
    Serial.print(temp);
    Serial.print("°C, Humidity: ");
    Serial.print(humidity);
    Serial.print("%, PM2.5: ");
    Serial.print(PM25);
    Serial.print("μg/m³, PM10: ");
    Serial.print(PM10);
    Serial.print("μg/m³, CO2(est): ");
    Serial.print(mq135CO2ppm);
    Serial.print(" ppm, MQ135 raw: ");
    Serial.print(mq135Raw);
    Serial.print(", R0: ");
    Serial.println(MQ135_R0);
    
    displayOnLCD();
    handleBuzzerAlert();
    
    if (isCalibrated) {
      if (millis() - lastSendTime >= sendInterval) {
        sendSensorData();
        lastSendTime = millis();
      }
    }
  }
  
  delay(1000);
}

void connectToWiFi() {
  Serial.print("Connecting to WiFi: ");
  Serial.println(ssid);
  
  WiFi.mode(WIFI_STA);
  WiFi.begin(ssid, password);
  
  int attempts = 0;
  while (WiFi.status() != WL_CONNECTED && attempts < 20) {
    delay(500);
    Serial.print(".");
    attempts++;
  }
  
  if (WiFi.status() == WL_CONNECTED) {
    Serial.println("\nWiFi connected!");
    Serial.print("IP address: ");
    Serial.println(WiFi.localIP());
    showMessage("WiFi OK", "Reading sensors...");
    delay(1500);
    
    // Auto-calibrate if not already calibrated
    if (!isCalibrated) {
      Serial.println(">>> Auto-starting calibration <<<");
      startCalibration();
    }
  } else {
    Serial.println("\nFailed to connect to WiFi");
    showMessage("WiFi FAILED", "Retrying...");
  }
}

void sendSensorData() {
  if (WiFi.status() != WL_CONNECTED) {
    return;
  }
  
  WiFiClient client;
  HTTPClient http;
  
  String jsonPayload = "{\"node_id\":\"" + String(nodeId) + "\"," +
                       "\"temperature\":" + String(temp, 2) + 
                       ",\"humidity\":" + String(humidity, 2) + 
                       ",\"pm25\":" + String(PM25, 2) + 
                       ",\"pm10\":" + String(PM10, 2) + 
                       ",\"co2\":" + String(mq135CO2ppm, 1) + 
                       ",\"mq135_raw\":" + String(mq135Raw) + 
                       "}";
  
  http.begin(client, serverUrl);
  http.addHeader("Content-Type", "application/json");
  
  int httpResponseCode = http.POST(jsonPayload);
  
  if (httpResponseCode > 0) {
    Serial.print("[");
    Serial.print(nodeId);
    Serial.print("] Data sent. Response code: ");
    Serial.println(httpResponseCode);
  } else {
    Serial.print("[");
    Serial.print(nodeId);
    Serial.print("] Error sending data. Code: ");
    Serial.println(httpResponseCode);
  }
  
  http.end();
}

void displayOnLCD() {
  // Calculate AQI and render on OLED
  calculateAQI();
  
  display.clearDisplay();
  display.setTextSize(1);
  display.setTextColor(SSD1306_WHITE);

  // Row 1: AQI and category
  display.setCursor(0, 0);
  display.print("AQI: ");
  display.print(currentAQI);
  display.print(" ");
  display.println(aqi_labels[getAQICategory(currentAQI)]);

  // Row 2: PM metrics
  display.setCursor(0, 12);
  display.print("PM25:");
  display.print((int)PM25);
  display.print(" PM10:");
  display.println((int)PM10);

  // Row 3: Temp/Humidity
  display.setCursor(0, 24);
  display.print("T:");
  display.print(temp, 1);
  display.print("C H:");
  display.print(humidity, 0);
  display.println("%");

  // Row 4: CO2 estimate
  display.setCursor(0, 36);
  display.print("CO2(est): ");
  display.print((int)mq135CO2ppm);
  display.println(" ppm");

  display.display();
}

void buzz(uint8_t times = 2, uint16_t onMs = 100, uint16_t offMs = 120) {
  for (uint8_t i = 0; i < times; i++) {
    digitalWrite(BUZZER_PIN, HIGH);
    delay(onMs);
    digitalWrite(BUZZER_PIN, LOW);
    if (i + 1 < times) {
      delay(offMs);
    }
  }
}

void handleBuzzerAlert() {
  if (!buzzerEnabled) return;
  int category = getAQICategory(currentAQI);
  if (category >= 3) {
    unsigned long now = millis();
    if (now - lastBuzzerTime >= buzzerInterval) {
      buzz(3, 120, 140);
      lastBuzzerTime = now;
    }
  }
}

void readMQ135() {
  mq135Raw = analogRead(MQ135_PIN);
  if (mq135Raw == 0) {
    mq135CO2ppm = 0;
    mq135NO2Index = 0;
    mq135Rs = 0;
    return;
  }
  float vRatio = (1023.0 / (float)mq135Raw) - 1.0;
  mq135Rs = MQ135_RLOAD * vRatio;
  
  float tempFactor = 1.0 + (temp - 20.0) * 0.005;
  mq135Rs = mq135Rs / tempFactor;
  
  float rs_r0 = mq135Rs / MQ135_R0;
  mq135CO2ppm = MQ135_CO2_A * pow(rs_r0, MQ135_CO2_B);
  mq135NO2Index = 1.0 / rs_r0;
}

void calculateAQI() {
  float pm = PM25;
  
  if (pm <= 12.0) {
    currentAQI = (int)(pm * 50.0 / 12.0);
  }
  else if (pm <= 35.4) {
    currentAQI = (int)(50.0 + (pm - 12.1) * 50.0 / 23.3);
  }
  else if (pm <= 55.4) {
    currentAQI = (int)(100.0 + (pm - 35.5) * 50.0 / 19.9);
  }
  else if (pm <= 150.4) {
    currentAQI = (int)(150.0 + (pm - 55.5) * 50.0 / 94.9);
  }
  else if (pm <= 250.4) {
    currentAQI = (int)(200.0 + (pm - 150.5) * 100.0 / 99.9);
  }
  else {
    currentAQI = (int)(300.0 + (pm - 250.5) * 50.0 / 50.0);
  }
  
  if (currentAQI > 500) {
    currentAQI = 500;
  }
}

int getAQICategory(int aqi) {
  if (aqi <= 50) return 0;
  if (aqi <= 100) return 1;
  if (aqi <= 150) return 2;
  if (aqi <= 200) return 3;
  if (aqi <= 300) return 4;
  return 5;
}

void loadCalibrationData() {
  byte calibFlag = EEPROM.read(EEPROM_CALIBRATED_FLAG);
  
  if (calibFlag == 0xFF) {
    isCalibrated = false;
    isWarmingUp = true;
    Serial.println("FIRST BOOT - NO CALIBRATION DATA");
    Serial.println("Sensor needs calibration. Use serial commands:");
    Serial.println("  'CALIB' - Start calibration mode in clean air");
    Serial.println("  'STATUS' - Show calibration status");
  } else {
    EEPROM.get(EEPROM_R0_ADDR, MQ135_R0);
    Serial.print("Loaded R0 from EEPROM: ");
    Serial.println(MQ135_R0);
    isCalibrated = true;
    isWarmingUp = false;
  }
}

void saveCalibrationData() {
  EEPROM.write(EEPROM_CALIBRATED_FLAG, 0x01);
  EEPROM.put(EEPROM_R0_ADDR, MQ135_R0);
  EEPROM.put(EEPROM_CALIB_TIME_ADDR, (uint32_t)millis());
  EEPROM.commit();
  Serial.println("Calibration data saved to EEPROM");
}

void handleSerialCommand() {
  String command = Serial.readStringUntil('\n');
  command.trim();
  command.toUpperCase();
  
  if (command == "CALIB") {
    startCalibration();
  } 
  else if (command == "STATUS") {
    displayCalibrationStatus();
  }
  else if (command.startsWith("R0:")) {
    // Manual R0 input: R0:7800
    String r0Str = command.substring(3);
    float newR0 = r0Str.toFloat();
    if (newR0 > 1000 && newR0 < 100000) {
      MQ135_R0 = newR0;
      saveCalibrationData();
      Serial.print(">>> R0 set to: ");
      Serial.println(MQ135_R0);
    } else {
      Serial.println("Invalid R0 value. Range: 1000-100000");
    }
  }
  else if (command == "RESET") {
    resetCalibration();
  }
  else {
    Serial.println("Unknown command. Available:");
    Serial.println("  CALIB - Start calibration");
    Serial.println("  STATUS - Show status");
    Serial.println("  R0:xxxx - Set R0 manually");
    Serial.println("  RESET - Clear calibration");
  }
}

void startCalibration() {
  Serial.println("STARTING CALIBRATION MODE");
  Serial.println("Instructions:");
  Serial.println("1. Place sensor in CLEAN AIR (outdoor/ventilated area)");
  Serial.println("2. Let it warm up for 2 minutes (MQ135 needs heating)");
  Serial.println("3. Calibration will auto-complete after 2 minutes");
  Serial.println("4. R0 will be calculated and saved");
  
  calibrationMode = true;
  calibrationStartTime = millis();
  isWarmingUp = true;
  isCalibrated = false;

  showMessage("CALIBRATION", "MODE ACTIVE");
}

void checkWarmupStatus() {
  if (!calibrationMode && isWarmingUp) {
    unsigned long warmupTime = millis() - calibrationStartTime;
    
    if (warmupTime >= warmupDuration) {
      isWarmingUp = false;
      isCalibrated = true;
      Serial.println("Warm-up complete. Sensor ready.");
      Serial.println("First calibration assumed. Use 'CALIB' for precise calibration.");
    }
  }
  
  if (calibrationMode) {
    unsigned long calibTime = millis() - calibrationStartTime;
    
    if (calibTime < warmupDuration) {
      int seconds = (warmupDuration - calibTime) / 1000;
      Serial.print("Warming up... ");
      Serial.print(seconds);
      Serial.println(" seconds remaining");

      String line2 = String(seconds) + "s remaining";
      showMessage("Warming up...", line2);
    } 
    else {
      if (mq135Rs > 0) {
        MQ135_R0 = mq135Rs / pow(400.0 / MQ135_CO2_A, 1.0 / MQ135_CO2_B);
        
        saveCalibrationData();
        calibrationMode = false;
        isCalibrated = true;
        
        Serial.println("CALIBRATION COMPLETE");
        Serial.print("Calculated R0: ");
        Serial.println(MQ135_R0);
        Serial.println("Sensor is now ready to use.");
        
        showMessage("CALIBRATION OK", "Ready!");
        
        delay(2000);
      }
    }
  }
}

void displayCalibrationStatus() {
  Serial.println("========== CALIBRATION STATUS ==========");
  Serial.print("Calibrated: ");
  Serial.println(isCalibrated ? "YES" : "NO");
  Serial.print("Warming up: ");
  Serial.println(isWarmingUp ? "YES" : "NO");
  Serial.print("Current R0: ");
  Serial.println(MQ135_R0);
  Serial.print("MQ135 Raw: ");
  Serial.println(mq135Raw);
  Serial.print("MQ135 Rs: ");
  Serial.println(mq135Rs);
  
  if (!isCalibrated) {
    Serial.println("To calibrate, send 'CALIB' command via serial monitor");
  }
  Serial.println("=========================================");
}

void resetCalibration() {
  EEPROM.write(EEPROM_CALIBRATED_FLAG, 0xFF);
  EEPROM.commit();
  
  isCalibrated = false;
  isWarmingUp = true;
  MQ135_R0 = 7600.0;
  calibrationStartTime = millis();
  
  Serial.println("CALIBRATION DATA RESET");
  Serial.println("Sensor needs recalibration.");
  Serial.println("Send 'CALIB' to start calibration process.");
}
