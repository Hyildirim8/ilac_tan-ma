# İlaç Tanıma Sistemi

ESP32-CAM ile çekilen ilaç kutusu görüntülerinden derin öğrenme modelleriyle ilaç türünü tanıyan uçtan uca bir sistem. Görüntü toplama, veri artırma, çoklu model eğitimi ve web üzerinden tahmin adımlarını içerir.

## Mimari

```
ESP32-CAM  --(HTTP POST /upload_esp)-->  sunucu/ (FastAPI, veri toplama)
                                             |
                                             v
                                   veri_arttırma/ (augmentation)
                                             |
                                             v
                                   yapay_zeka/ (model eğitimi)
                                             |
                                             v
                                   tahmin_sitesi/ (FastAPI, ensemble tahmin)
```

1. **ESP32-CAM**, 5 saniyede bir fotoğraf çekip `sunucu` servisine yükler.
2. **sunucu**, gelen fotoğrafı gösterir; kullanıcı arayüzden ilaç adını girip veri setine (`sunucu/dataset/<ilaç_adı>/`) kaydeder.
3. **veri_arttırma**, toplanan veri setini döndürme/kaydırma/yakınlaştırma gibi dönüşümlerle çoğaltır (`augmented_dataset/`).
4. **yapay_zeka**, artırılmış veri setiyle 8 farklı modeli (basic/advanced/ensemble CNN, VGG16, ResNet50, MobileNetV2, EfficientNet, Vision Transformer) eğitir ve karşılaştırır.
5. **tahmin_sitesi**, eğitilmiş tüm modelleri yükleyip ESP32'den gelen görüntüler üzerinde ensemble (ortalama + oylama) tahmin yapar ve sonucu web arayüzünde gösterir.

## Klasör yapısı

| Klasör | Açıklama |
|---|---|
| `esp_code/sketch_oct22a/` | ESP32-CAM Arduino kodu — WiFi'ye bağlanır, kamerayı başlatır, periyodik olarak sunucuya görüntü gönderir. |
| `sunucu/` | Veri toplama sunucusu (FastAPI). ESP32'den gelen görüntüyü alır, etiketleyip veri setine kaydeder. |
| `veri_arttırma/` | `veri_seti.py` ile `sunucu/dataset` içindeki görüntülerden data augmentation yapar. |
| `yapay_zeka/` | `app.py` ile modelleri eğitir; `.h5` model dosyaları ve `class_names*.json` sınıf haritalarını üretir. |
| `tahmin_sitesi/` | Tahmin web sitesi (FastAPI). Eğitilmiş tüm modelleri yükler, ensemble tahmin yapar, sonucu gösterir. |
| `grafik.py` | Model doğruluk ve parametre sayısı karşılaştırma grafiklerini (`accuracy_C.png`, `params_C.png`) üretir. |

## Kurulum ve çalıştırma

Her servis için ayrı bir sanal ortam önerilir (Python 3.10+).

### 1. Veri toplama sunucusu (`sunucu/`)

```bash
cd sunucu
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python server.py   # http://0.0.0.0:5000
```

### 2. ESP32-CAM (`esp_code/sketch_oct22a/`)

Arduino IDE'de açıp yükleyin. Yüklemeden önce dosya içindeki `ssid`, `password` ve `serverName` (sunucunun yerel IP adresi) değerlerini kendi ortamınıza göre güncelleyin.

### 3. Veri artırma (`veri_arttırma/`)

```bash
cd veri_arttırma
pip install tensorflow
python veri_seti.py   # augmented_dataset/ klasörünü oluşturur
```

### 4. Model eğitimi (`yapay_zeka/`)

```bash
cd yapay_zeka
pip install tensorflow numpy
python app.py   # tüm modelleri eğitir, .h5 ve class_names*.json dosyalarını üretir
```

### 5. Tahmin sitesi (`tahmin_sitesi/`)

Eğitilen `.h5` ve `class_names*.json` dosyalarını `tahmin_sitesi/model/` altına kopyalayın (bu dosyalar `.gitignore` ile hariç tutulur, bkz. aşağıda).

```bash
cd tahmin_sitesi
pip install fastapi uvicorn tensorflow pillow python-multipart
python main.py   # http://0.0.0.0:5000
```

## Model sonuçları

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
- ESP32 kodundaki WiFi bilgileri örnek amaçlıdır; gerçek dağıtımda repoya işlenmeden önce kaldırılmalı veya bir yapılandırma dosyasına taşınmalıdır.
