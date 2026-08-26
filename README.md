# İlaç Tanıma Sistemi

ESP32-CAM ile çekilen ilaç kutusu görüntülerinden derin öğrenme modelleriyle ilaç türünü tanıyan uçtan uca bir sistem. Görüntü toplama, veri artırma, çoklu model eğitimi ve web üzerinden tahmin adımlarını içerir.

## Mimari

```
ESP32-CAM  --(HTTP POST /upload_esp)-->  web/ (FastAPI, tek site — veri toplama + tahmin, Docker)
                                             |
                                             v
                                   veri_arttırma/ (augmentation)
                                             |
                                             v
                                   yapay_zeka/ (model eğitimi)
                                             |
                                             v
                                   web/model/  (eğitilen .h5 + class_names.json buraya kopyalanır)
```

1. **ESP32-CAM**, 5 saniyede bir fotoğraf çekip `web` servisine (`/upload_esp`) yükler.
2. **web** sitesi üç sekmeden oluşur:
   - **📥 Veri Ekleme & Artırma** (`/`) — canlı ESP32 önizlemesi, ilaç adı girip veri setine kaydetme (`web/dataset/<ilaç_adı>/`), veri setindeki klasör/görüntü sayıları ve tek tıkla veri artırma (`web/augmented_dataset/`).
   - **🔬 Veri Tahmini** (`/tahmin`) — `web/model/` içinde eğitilmiş `.h5` dosyaları varsa ESP32'den gelen görüntü üzerinde canlı ensemble tahmin (ortalama, çoğunluk oyu, en iyi model) gösterir.
   - **📊 Toplam Sonuçlar** (`/sonuclar`) — `web/model/model_results.json` içindeki eğitim sonuçlarını accuracy ve parametre sayısı grafikleri olarak gösterir.
3. **veri_arttırma** (veya web sitesindeki "Veri Artırmayı Başlat" butonu), toplanan veri setini döndürme/kaydırma/yakınlaştırma gibi dönüşümlerle çoğaltır (`augmented_dataset/`).
4. **yapay_zeka**, artırılmış veri setiyle 8 farklı modeli (basic/advanced/ensemble CNN, VGG16, ResNet50, MobileNetV2, EfficientNet, Vision Transformer) eğitir, karşılaştırır ve `model_results.json` üretir.
5. Üretilen `.h5`, `class_names*.json` ve `model_results.json` dosyaları `web/model/` altına kopyalanınca **web** servisi bunları otomatik yükler; tahmin ve sonuç grafikleri devreye girer.

## Klasör yapısı

| Klasör | Açıklama |
|---|---|
| `esp_code/sketch_oct22a/` | ESP32-CAM Arduino kodu — WiFi'ye bağlanır, kamerayı başlatır, periyodik olarak `web` servisine görüntü gönderir. |
| `web/` | Birleşik FastAPI sitesi (tek Docker container). ESP32'den gelen görüntüyü alır, veri setine kaydeder VE (model varsa) ensemble tahmin üretir. Eski `sunucu/` ve `tahmin_sitesi/` bu klasörde birleştirildi. |
| `veri_arttırma/` | `veri_seti.py` ile `web/dataset` içindeki görüntülerden data augmentation yapar. |
| `yapay_zeka/` | `app.py` ile modelleri eğitir; `.h5` model dosyaları ve `class_names*.json` sınıf haritalarını üretir (bunları `web/model/` altına kopyalayın). |
| `grafik.py` | Model doğruluk ve parametre sayısı karşılaştırma grafiklerini (`accuracy_C.png`, `params_C.png`) üretir. |

## Kurulum ve çalıştırma

### 1. Web sitesi — veri toplama + tahmin (`web/`, Docker)

Tek komutla ayağa kalkar; hem ESP32'den gelen görüntüleri veri setine kaydeder hem de (eğitilmiş model varsa) tahmin üretir.

```bash
docker compose up --build   # http://0.0.0.0:5050 (host portu; container içinde 5000, macOS'ta AirPlay Receiver 5000'i kullandığı için 5050'ye eşlendi)
```

- `web/dataset/` — etiketlenmiş veri seti buraya kaydedilir (host'ta kalıcı, volume mount).
- `web/augmented_dataset/` — "Veri Artırmayı Başlat" butonuyla üretilen çoğaltılmış görüntüler.
- `web/model/` — eğitilen `.h5` model dosyalarını, `class_names*.json`'ı ve `model_results.json`'ı buraya kopyalayın; container'ı yeniden build etmeden, sadece yeniden başlatınca (`docker compose restart`) yüklenirler.

Docker olmadan çalıştırmak isterseniz:

```bash
cd web
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python main.py   # http://0.0.0.0:5000
```

### 2. ESP32-CAM (`esp_code/sketch_oct22a/`)

WiFi bilgileri koda gömülü değil, `esp_code/sketch_oct22a/secrets.h` içinden gelir (bu dosya `.gitignore` ile hariç tutulur). İlk kurulumda `secrets.h.example` dosyasını `secrets.h` olarak kopyalayıp kendi SSID/şifrenizi girin:

```bash
cp esp_code/sketch_oct22a/secrets.h.example esp_code/sketch_oct22a/secrets.h
# secrets.h içindeki WIFI_SSID / WIFI_PASSWORD değerlerini düzenleyin
```

Ardından Arduino IDE'de açıp yükleyin; yüklemeden önce dosya içindeki `serverName` değerini (web sitesinin çalıştığı bilgisayarın yerel IP adresi + `:5050/upload_esp` — Docker ile çalıştırıyorsanız; `python main.py` ile doğrudan çalıştırıyorsanız `:5000/upload_esp`) kendi ortamınıza göre güncelleyin.

### 3. Veri artırma (`veri_arttırma/`)

Web sitesinin "📥 Veri Ekleme & Artırma" sekmesindeki **Veri Artırmayı Başlat** butonuyla tarayıcıdan tetiklenebilir, ya da bağımsız script olarak:

```bash
cd veri_arttırma
pip install tensorflow scipy
python veri_seti.py   # web/augmented_dataset/ klasörünü oluşturur
```

### 4. Model eğitimi (`yapay_zeka/`)

```bash
cd yapay_zeka
pip install tensorflow numpy
python app.py   # tüm modelleri eğitir, .h5 ve class_names*.json dosyalarını üretir
```

Üretilen `.h5` ve `class_names*.json` dosyalarını `web/model/` altına kopyalayın (bu dosyalar `.gitignore` ile hariç tutulur, bkz. aşağıda) — `web` servisi bunları otomatik algılar.

## Model sonuçları

Bu tablo, web sitesinin **📊 Toplam Sonuçlar** sekmesinde canlı grafik olarak da gösterilir (`web/model/model_results.json` üzerinden — `yapay_zeka/app.py` eğitim sonunda bu dosyayı otomatik üretir).

Mevcut veri setiyle (2 sınıf: `Lansoprol`, `Napren`) elde edilen doğrulama sonuçları (`yapay_zeka/model_results_fixed.txt`):

| Model | Val. Accuracy | Parametre |
|---|---|---|
| ensemble_cnn | 1.0000 | 134,114 |
| mobilenet | 1.0000 | 2,422,210 |
| basic_cnn | 0.9737 | 3,984,706 |
| vgg16 | 0.9211 | 14,846,530 |
| advanced_cnn | 0.5526 | 127,682 |
| resnet50 | 0.5000 | 24,637,826 |
| efficientnet_fixed | 0.5000 | 6,247,762 |
| vision_transformer_fixed | 0.5000 | 1,419,906 |

Veri seti çok küçük olduğundan (sınıf başına ~10 orijinal görüntü) bu sonuçlar sınırlı genelleme gücüne sahiptir; daha fazla ilaç sınıfı ve görüntü ile yeniden eğitim önerilir.

## Notlar

- Eğitilmiş model dosyaları (`*.h5`) ve veri setleri repoya dahil edilmemiştir (bkz. `.gitignore`) — büyük boyutları nedeniyle ayrıca paylaşılmalı veya yerelde yeniden üretilmelidir.
- ESP32 kodundaki WiFi bilgileri `esp_code/sketch_oct22a/secrets.h` içinde tutulur ve `.gitignore` ile repoya dahil edilmez (bkz. `secrets.h.example`).
