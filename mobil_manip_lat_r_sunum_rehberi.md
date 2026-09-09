# Mobil Manipülatör Robot Geliştirme Rehberi
**Sistem Mimarisi: Tekerlekli Taban (AMR) + Dikey Asansör + Robot Kol**

---

## 1. Sistemin Genel Tanımı ve Çalışma Prensibi
Bu robot tipi endüstride ve lojistikte **"Mobil Manipülatör"** olarak adlandırılır. 
* **Tekerlekli Taban:** Robotun ortamda serbestçe yer değiştirmesini sağlar (Navigasyon).
* **Asansör (Prizmatik Eksen):** Robot kolun erişim yüksekliğini (çalışma hacmini) dikeyde artırır.
* **Robot Kol:** Belirlenen nesneleri hassas şekilde yakalar, taşır veya işler (Manipülasyon).

---

## 2. Mekanik Tasarım ve Fiziksel Temeller

### A. Denge ve Ağırlık Merkezi (Devrilme Güvenliği)
* **Kritik Problem:** Kol ileri uzandığında ve asansör en üst seviyedeyken ağırlık merkezi ($CoG$) yukarı ve dışa doğru kayar.
* **Gereken Bilgi:** 
  * Taban genişliği ve şasi ağırlık oranı doğru ayarlanmalı (aküler ve ağır bileşenler en alta yerleştirilir).
  * Maksimum yük ve maksimum uzanma anındaki devrilme momenti ($M = F \cdot d$) hesaplanmalıdır.

### B. Tahrik ve İletim Mekanizmaları
* **Tekerlek Yapısı:** 
  * *Diferansiyel Sürüş:* 2 motorlu teker + sarhoş tekerlek (basit, güvenilir).
  * *Mecanum/Omni Sürüş:* Her yöne anlık hareket kabiliyeti (dar alanlar için ideal).
* **Asansör Mekanizması:** 
  * Vidalı mil (ball screw) veya kayış-kasnak mekanizması.
  * **Güvenlik Kuralı:** Güç kesildiğinde asansörün kendi ağırlığıyla aşağı düşmemesi için mekanik fren veya kendinden kilitli (otoblokajlı) redüktör kullanımı.
* **Robot Kol Eklemleri:** 
  * Yüksek tork, sıfır boşluk (backlash) sağlayan harmonik/planet redüktörlü servo aktüatörler.

---

## 3. Kinematik ve Hareket Kontrolü

### A. Koordinat Dönüşümleri ve Kinematik
* **İleri Kinematik (Forward Kinematics):** Motorların açı ve pozisyonları bilindiğinde robot kolun ucunun uzaydaki ($X, Y, Z$) yerini bulma.
* **Ters Kinematik (Inverse Kinematics):** Robot kolunun ucunun gitmesi istenen hedef koordinata göre her bir motorun alması gereken açıları hesaplama.
* **Bütünleşik Kinematik Ağaç:** Taban ($X, Y, \theta$) + Asansör ($Z$) + Kol ($\theta_1, \dots, \theta_n$) tek bir kinematik zincir olarak tanımlanır.

### B. Yörünge ve Hareket Profilleri
* Ani duruş/kalkışlarda sarsıntıyı önlemek için ivme ve hız sınırlamaları (Trapezoidal veya S-Curve hız profilleri).
* Motorların hedeflenen pozisyonda tam ve titreşimsiz durması için **PID Kontrol** döngüleri.

---

## 4. Elektronik Donanım ve Güç Yönetimi

### A. Güç Dağıtımı
* **Batarya:** Yüksek anlık akım verebilen LiFePO4 veya Li-Ion batarya paketleri.
* **BMS (Akü Yönetim Sistemi):** Hücre dengeleme, aşırı akım ve kısa devre koruması.
* **Voltaj Regülasyonu:** Motor sürücülerinin yüksek voltajı (örn. 24V/48V) ile bilgisayar/sensörlerin düşük voltajının (5V/12V) DC-DC izole regülatörlerle ayrılması.

### B. Motor ve Sürücü Seçimi
* **Motor Tipleri:**
  * Tekerlekler ve Asansör: Yüksek torklu BLDC (Fırçasız DC) veya Step motorlar.
  * Robot Kol: Entegre tork/pozisyon geri beslemeli akıllı servolar.
* **Geri Besleme Elemanları:** Pozisyon hassasiyeti için mutlak (absolute) veya artımsal (incremental) enkoderler.

### C. Haberleşme Hatları
* Sistem içindeki motorlar ve sürücüler arasında elektriksel gürültüye dayanıklı endüstriyel haberleşme:
  * **CAN Bus** (En yaygın ve güvenilir)
  * **RS485 / Modbus**
  * **Ethernet** (Yüksek veri transferi için)

---

## 5. Yazılım ve Sistem Mimarisi

### A. Alt Seviye Kontrol (Gömülü Katman - MCU / STM32 / ESP32)
* Gerçek zamanlı motor sürme ve hız/pozisyon kontrolü.
* Acil Stop (E-Stop) ve mekanik limit/güvenlik anahtarlarının takibi.
* Enkoder ve alt seviye sensör verilerini üst bilgisayara iletme.

### B. Üst Seviye Kontrol (Mini PC / ROS 2 / Linux)
* **Ortam Algılama ve Navigasyon (SLAM):** 
  * 2D/3D LiDAR ve IMU yardımıyla haritalama, engelleri algılama ve rota oluşturma.
* **Manipülasyon Planlama:** 
  * Kol ve asansörün çevreye veya robotun kendi gövdesine çarpmadan hareketini planlama.
* **Görsel Algılama (Opsiyonel):** 
  * Nesneleri tanımak ve konumlarını hesaplamak için derinlik kameraları (RGB-D).

---

## 6. Proje Aşamaları (Yol Haritası)

1. **Gereksinimlerin Belirlenmesi:** Taşıma kapasitesi (payload), erişim mesafesi, hız ve çalışma ortamı limitleri.
2. **CAD Modelleme ve Ağırlık Analizi:** 3D modelleme, devrilme testleri ve mekanik parça üretimi.
3. **Elektronik Mimari & Pano:** Güç hatlarının, sürücülerin ve güvenlik donanımlarının montajı.
4. **Alt Seviye Entegrasyon:** Motorların tek tek ve koordineli döndürülmesi, güvenlik sınırlarının test edilmesi.
5. **Üst Seviye Yazılım Entegrasyonu:** Kinematik hesaplamalar, navigasyon ve otonom görev zincirinin kurulması.