// ESP32-CAM flaş LED testi — kamera/WiFi/foto çekme yok, sadece
// FLASH_GPIO pinini 1 saniye aralıklarla yakıp söndürür.
#define FLASH_GPIO 4

void setup() {
  pinMode(FLASH_GPIO, OUTPUT);
}

void loop() {
  digitalWrite(FLASH_GPIO, HIGH);
  delay(1000);
  digitalWrite(FLASH_GPIO, LOW);
  delay(1000);
}
