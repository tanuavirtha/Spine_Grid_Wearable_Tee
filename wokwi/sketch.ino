/* ============================================================
   Spine-Grid Posture Tee — Wokwi Simulation Firmware
   ESP32-S3 | 4x MPU6050 (2x I2C buses) | 2x flex pots | 3x LED | buzzer
   ============================================================ */

#include <Wire.h>

// ---------- I2C bus / address map ----------
// Wire  (SDA=8, SCL=9): imu1 = upper back (0x68), imu4 = lumbar (0x69)
// Wire1 (SDA=4, SCL=5): imu3 = right shoulder (0x68), imu2 = left shoulder (0x69)
#define MPU_ADDR_A 0x68
#define MPU_ADDR_B 0x69

// ---------- Pins ----------
const int PIN_POT1  = 2;   // left-to-right cross flex
const int PIN_POT2  = 1;   // right-to-left cross flex
const int PIN_LED_RED    = 10; // led1 - poor posture
const int PIN_LED_YELLOW = 11; // led2 - could be improved
const int PIN_LED_GREEN  = 12; // led3 - good posture
const int PIN_BUZZER = 13;
const int PIN_BUTTON = 14;     // reset / recalibrate

// ---------- Thresholds ----------
const float WARN_DEG     = 8.0;   // deviation that triggers "could be improved"
const float CRITICAL_DEG = 16.0;  // deviation that starts the bad-posture timer
const int   FLEX_WARN     = 400;  // ADC counts deviation (12-bit, 0-4095)
const int   FLEX_CRITICAL = 900;
const unsigned long BAD_POSTURE_HOLD_MS = 5000; // 5s sustained bad posture -> alert

// ---------- State ----------
struct Orientation { float pitch; float roll; };

Orientation baseUpper, baseLumbar, baseLeftSh, baseRightSh;
int basePot1 = 0, basePot2 = 0;
bool calibrated = false;

unsigned long criticalSinceMs = 0;
bool alertActive = false;

// Debounce for reset button
unsigned long lastButtonMs = 0;
const unsigned long DEBOUNCE_MS = 250;

// ---------- I2C helpers ----------
void mpuWake(TwoWire &bus, uint8_t addr) {
  bus.beginTransmission(addr);
  bus.write(0x6B); // PWR_MGMT_1
  bus.write(0x00); // clear sleep bit
  bus.endTransmission(true);
}

bool mpuReadAccel(TwoWire &bus, uint8_t addr, float &ax, float &ay, float &az) {
  bus.beginTransmission(addr);
  bus.write(0x3B); // ACCEL_XOUT_H
  if (bus.endTransmission(false) != 0) return false;
  bus.requestFrom((int)addr, 6, true);
  if (bus.available() < 6) return false;

  int16_t rawX = (bus.read() << 8) | bus.read();
  int16_t rawY = (bus.read() << 8) | bus.read();
  int16_t rawZ = (bus.read() << 8) | bus.read();

  // +-2g range default -> 16384 LSB/g
  ax = rawX / 16384.0;
  ay = rawY / 16384.0;
  az = rawZ / 16384.0;
  return true;
}

Orientation computeOrientation(float ax, float ay, float az) {
  Orientation o;
  o.pitch = atan2(-ax, sqrt(ay * ay + az * az)) * 180.0 / PI;
  o.roll  = atan2(ay, az) * 180.0 / PI;
  return o;
}

Orientation readIMU(TwoWire &bus, uint8_t addr) {
  float ax, ay, az;
  if (!mpuReadAccel(bus, addr, ax, ay, az)) {
    return {0.0, 0.0}; // fail-safe: no deviation reported on read error
  }
  return computeOrientation(ax, ay, az);
}

// ---------- Calibration ----------
void calibrate() {
  baseUpper   = readIMU(Wire, MPU_ADDR_A);   // imu1 upper back
  baseLumbar  = readIMU(Wire, MPU_ADDR_B);   // imu4 lumbar
  baseRightSh = readIMU(Wire1, MPU_ADDR_A);  // imu3 right shoulder
  baseLeftSh  = readIMU(Wire1, MPU_ADDR_B);  // imu2 left shoulder
  basePot1 = analogRead(PIN_POT1);
  basePot2 = analogRead(PIN_POT2);

  criticalSinceMs = 0;
  alertActive = false;
  calibrated = true;

  Serial.println("=== Calibrated: current pose set as neutral posture ===");
}

// ---------- Setup ----------
void setup() {
  Serial.begin(9600);
  delay(200);

  Wire.begin(8, 9);
  Wire1.begin(4, 5);

  mpuWake(Wire, MPU_ADDR_A);   // imu1
  mpuWake(Wire, MPU_ADDR_B);   // imu4
  mpuWake(Wire1, MPU_ADDR_A);  // imu3
  mpuWake(Wire1, MPU_ADDR_B);  // imu2

  pinMode(PIN_LED_RED, OUTPUT);
  pinMode(PIN_LED_YELLOW, OUTPUT);
  pinMode(PIN_LED_GREEN, OUTPUT);
  pinMode(PIN_BUZZER, OUTPUT);
  pinMode(PIN_BUTTON, INPUT_PULLUP);

  analogReadResolution(12);

  delay(300); // let simulated IMUs settle
  calibrate();
}

// ---------- Alert helpers ----------
void setStatusLEDs(int state) {
  // state: 0 = good, 1 = warning, 2 = critical
  digitalWrite(PIN_LED_GREEN,  state == 0 ? HIGH : LOW);
  digitalWrite(PIN_LED_YELLOW, state == 1 ? HIGH : LOW);
  digitalWrite(PIN_LED_RED,    state == 2 ? HIGH : LOW);
}

void buzzerAlert(bool on) {
  if (on) {
    tone(PIN_BUZZER, 2000); // 2kHz alert tone
  } else {
    noTone(PIN_BUZZER);
  }
}

// ---------- Main loop ----------
void loop() {
  // --- Reset button ---
  if (digitalRead(PIN_BUTTON) == LOW && millis() - lastButtonMs > DEBOUNCE_MS) {
    lastButtonMs = millis();
    buzzerAlert(false);
    calibrate();
  }

  // --- Read all sensors ---
  Orientation curUpper   = readIMU(Wire, MPU_ADDR_A);
  Orientation curLumbar  = readIMU(Wire, MPU_ADDR_B);
  Orientation curRightSh = readIMU(Wire1, MPU_ADDR_A);
  Orientation curLeftSh  = readIMU(Wire1, MPU_ADDR_B);
  int pot1 = analogRead(PIN_POT1);
  int pot2 = analogRead(PIN_POT2);

  // --- Deviations from calibrated neutral pose ---
  float thoracicDev = abs(curUpper.pitch - baseUpper.pitch);
  float lumbarDev   = abs(curLumbar.pitch - baseLumbar.pitch);

  float leftRoll  = curLeftSh.roll  - baseLeftSh.roll;
  float rightRoll = curRightSh.roll - baseRightSh.roll;
  float shoulderAsymmetryDev = abs(leftRoll - rightRoll);

  int flexDev = abs(pot1 - basePot1) + abs(pot2 - basePot2);

  // --- Combine into a single worst-case severity ---
  bool anyCritical = (thoracicDev > CRITICAL_DEG) ||
                      (lumbarDev > CRITICAL_DEG) ||
                      (shoulderAsymmetryDev > CRITICAL_DEG) ||
                      (flexDev > FLEX_CRITICAL);

  bool anyWarning = (thoracicDev > WARN_DEG) ||
                     (lumbarDev > WARN_DEG) ||
                     (shoulderAsymmetryDev > WARN_DEG) ||
                     (flexDev > FLEX_WARN);

  // --- 5-second sustained-bad-posture timer ---
  if (anyCritical) {
    if (criticalSinceMs == 0) {
      criticalSinceMs = millis(); // start timer on first critical reading
    }
    if (millis() - criticalSinceMs >= BAD_POSTURE_HOLD_MS) {
      alertActive = true;
    }
  } else {
    criticalSinceMs = 0; // posture recovered before hold time -> cancel timer
    alertActive = false;
  }

  // --- Drive outputs ---
  if (alertActive) {
    setStatusLEDs(2);
    buzzerAlert(true);
  } else if (anyCritical) {
    // critical deviation present but hasn't been sustained 5s yet -> treat as warning
    setStatusLEDs(1);
    buzzerAlert(false);
  } else if (anyWarning) {
    setStatusLEDs(1);
    buzzerAlert(false);
  } else {
    setStatusLEDs(0);
    buzzerAlert(false);
  }

  // --- Debug output ---
  Serial.print("Thoracic: "); Serial.print(thoracicDev, 1);
  Serial.print(" | Lumbar: "); Serial.print(lumbarDev, 1);
  Serial.print(" | ShoulderAsym: "); Serial.print(shoulderAsymmetryDev, 1);
  Serial.print(" | FlexDev: "); Serial.print(flexDev);
  Serial.print(" | Alert: "); Serial.println(alertActive ? "BAD (5s+)" : (anyWarning ? "WARN" : "GOOD"));

  delay(200); // ~5 samples/sec
}
