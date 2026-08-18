import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models, Model
from tensorflow.keras.applications import VGG16, ResNet50, MobileNetV2, EfficientNetB0
import os
import json
import numpy as np
import datetime

# Dataset dizinleri
dataset_dir = "../veri_arttırma/augmented_dataset"  # augment edilmiş veriler

# Veri hazırlama ve augmentation (train/validation split)
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

# 96x96 için basit modeller
train_gen_96 = datagen.flow_from_directory(
    dataset_dir,
    target_size=(96, 96),
    batch_size=16,
    class_mode='categorical',
    subset='training'
)

val_gen_96 = datagen.flow_from_directory(
    dataset_dir,
    target_size=(96, 96),
    batch_size=16,
    class_mode='categorical',
    subset='validation'
)

# 224x224 için transfer learning modelleri
train_gen_224 = datagen.flow_from_directory(
    dataset_dir,
    target_size=(224, 224),
    batch_size=16,
    class_mode='categorical',
    subset='training'
)

val_gen_224 = datagen.flow_from_directory(
    dataset_dir,
    target_size=(224, 224),
    batch_size=16,
    class_mode='categorical',
    subset='validation'
)

# Sınıf isimlerini yazdır ve kaydet
print("Bulunan sınıflar (ilaç isimleri):")
print(train_gen_96.class_indices)

class_names = {v: k for k, v in train_gen_96.class_indices.items()}
with open("class_names.json", "w", encoding="utf-8") as f:
    json.dump(class_names, f, ensure_ascii=False, indent=2)

print("Sınıf isimleri class_names.json dosyasına kaydedildi")

num_classes = len(train_gen_96.class_indices)

# 1. Basit CNN Modeli (Mevcut)
def create_basic_cnn(num_classes):
    model = models.Sequential([
        layers.Conv2D(32, (3,3), activation='relu', input_shape=(96,96,3)),
        layers.MaxPooling2D(2,2),
        layers.Conv2D(64, (3,3), activation='relu'),
        layers.MaxPooling2D(2,2),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

# 2. Gelişmiş CNN Modeli
def create_advanced_cnn(num_classes):
    model = models.Sequential([
        layers.Conv2D(32, (3,3), activation='relu', input_shape=(96,96,3)),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2,2),
        
        layers.Conv2D(64, (3,3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2,2),
        
        layers.Conv2D(128, (3,3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2,2),
        
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

# 3. Ensemble CNN Modeli
def create_ensemble_cnn(num_classes):
    input_layer = layers.Input(shape=(96, 96, 3))
    
    # Model 1: Küçük kerneller
    x1 = layers.Conv2D(32, (3,3), activation='relu')(input_layer)
    x1 = layers.MaxPooling2D(2,2)(x1)
    x1 = layers.Conv2D(64, (3,3), activation='relu')(x1)
    x1 = layers.GlobalAveragePooling2D()(x1)
    x1 = layers.Dense(64, activation='relu')(x1)
    
    # Model 2: Büyük kerneller
    x2 = layers.Conv2D(32, (5,5), activation='relu')(input_layer)
    x2 = layers.MaxPooling2D(2,2)(x2)
    x2 = layers.Conv2D(64, (5,5), activation='relu')(x2)
    x2 = layers.GlobalAveragePooling2D()(x2)
    x2 = layers.Dense(64, activation='relu')(x2)
    
    # Model 3: Derin model
    x3 = layers.Conv2D(16, (3,3), activation='relu')(input_layer)
    x3 = layers.Conv2D(32, (3,3), activation='relu')(x3)
    x3 = layers.MaxPooling2D(2,2)(x3)
    x3 = layers.Conv2D(64, (3,3), activation='relu')(x3)
    x3 = layers.GlobalAveragePooling2D()(x3)
    x3 = layers.Dense(64, activation='relu')(x3)
    
    # Birleştir
    combined = layers.concatenate([x1, x2, x3])
    combined = layers.Dense(128, activation='relu')(combined)
    combined = layers.Dropout(0.5)(combined)
    output = layers.Dense(num_classes, activation='softmax')(combined)
    
    model = Model(inputs=input_layer, outputs=output)
    return model

# 4. VGG16 Transfer Learning
def create_vgg16_model(num_classes):
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = False
    
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

# 5. ResNet50 Transfer Learning
def create_resnet50_model(num_classes):
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = False
    
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

# 6. MobileNetV2 (Hafif model)
def create_mobilenet_model(num_classes):
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = False
    
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

# 7. EfficientNet (Düzeltilmiş)
def create_efficientnet_model(num_classes):
    try:
        # EfficientNet yerine EfficientNetV2 kullanabiliriz veya basit bir yaklaşım
        from tensorflow.keras.applications import EfficientNetV2B0
        base_model = EfficientNetV2B0(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        base_model.trainable = False
        
        model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.4),
            layers.Dense(num_classes, activation='softmax')
        ])
        return model
    except:
        # EfficientNet yüklenemeIrse alternatif model
        print("EfficientNet yüklenemedi, alternatif model kullanılıyor...")
        base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        base_model.trainable = False
        
        model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.4),
            layers.Dense(num_classes, activation='softmax')
        ])
        return model

# 8. Vision Transformer (Tamamen Düzeltilmiş)
class PatchEmbedding(layers.Layer):
    def __init__(self, patch_size, d_model, **kwargs):
        super(PatchEmbedding, self).__init__(**kwargs)
        self.patch_size = patch_size
        self.d_model = d_model
        
    def build(self, input_shape):
        # Conv2D layer'ı build metodunda oluştur
        self.patch_conv = layers.Conv2D(
            filters=self.d_model,
            kernel_size=self.patch_size,
            strides=self.patch_size,
            padding="valid",
            name="patch_conv"
        )
        super().build(input_shape)
        
    def call(self, images):
        # Conv2D kullanarak patch extraction
        patches = self.patch_conv(images)
        
        # Reshape to (batch_size, num_patches, d_model)
        batch_size = tf.shape(patches)[0]
        patch_height = patches.shape[1]
        patch_width = patches.shape[2]
        num_patches = patch_height * patch_width
        
        patches = tf.reshape(patches, [batch_size, num_patches, self.d_model])
        
        return patches

    def get_config(self):
        config = super().get_config()
        config.update({
            "patch_size": self.patch_size,
            "d_model": self.d_model,
        })
        return config

def create_vision_transformer(num_classes, image_size=96, patch_size=16, num_layers=4, d_model=128, num_heads=4):
    inputs = layers.Input(shape=(image_size, image_size, 3))
    
    # Patch embedding (düzeltilmiş)
    patches = PatchEmbedding(patch_size, d_model)(inputs)
    
    # Patch sayısını hesapla
    num_patches = (image_size // patch_size) ** 2
    
    # Position embedding
    position_embedding = layers.Embedding(
        input_dim=num_patches, 
        output_dim=d_model,
        name="position_embedding"
    )
    
    # Position indices oluştur
    positions = tf.range(start=0, limit=num_patches, delta=1)
    position_embeddings = position_embedding(positions)
    
    # Patch embeddings + position embeddings
    encoded_patches = patches + position_embeddings
    
    # Transformer blocks
    for i in range(num_layers):
        # Multi-head attention
        x1 = layers.LayerNormalization(epsilon=1e-6, name=f"norm1_{i}")(encoded_patches)
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, 
            key_dim=d_model,
            name=f"attention_{i}"
        )(x1, x1)
        x2 = layers.Add(name=f"add1_{i}")([attention_output, encoded_patches])
        
        # MLP
        x3 = layers.LayerNormalization(epsilon=1e-6, name=f"norm2_{i}")(x2)
        x3 = layers.Dense(d_model * 2, activation="gelu", name=f"mlp_dense1_{i}")(x3)
        x3 = layers.Dense(d_model, name=f"mlp_dense2_{i}")(x3)
        encoded_patches = layers.Add(name=f"add2_{i}")([x3, x2])
    
    # Classifier
    representation = layers.LayerNormalization(epsilon=1e-6, name="final_norm")(encoded_patches)
    representation = layers.GlobalAveragePooling1D(name="global_avg_pool")(representation)
    representation = layers.Dropout(0.3, name="final_dropout")(representation)
    outputs = layers.Dense(num_classes, activation="softmax", name="classifier")(representation)
    
    model = Model(inputs=inputs, outputs=outputs, name="vision_transformer")
    return model

# Sonuçları text dosyasına yazma fonksiyonu
def write_results_to_file(results, filename="model_results.txt"):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    with open(filename, "w", encoding="utf-8") as f:
        f.write("="*80 + "\n")
        f.write("İLAÇ TANIMA SİSTEMİ - MODEL EĞİTİM SONUÇLARI\n")
        f.write("="*80 + "\n")
        f.write(f"Tarih: {timestamp}\n")
        f.write(f"Toplam Sınıf Sayısı: {num_classes}\n")
        f.write(f"Sınıflar: {list(class_names.values())}\n\n")
        
        # Başarılı modeller
        valid_results = {k: v for k, v in results.items() if "error" not in v}
        
        if valid_results:
            f.write("BAŞARILI MODELLER:\n")
            f.write("-"*80 + "\n")
            f.write(f"{'Rank':<5} {'Model':<20} {'Accuracy':<12} {'Loss':<12} {'Parametreler':<15}\n")
            f.write("-"*80 + "\n")
            
            # En iyi modeli bul
            best_model = max(valid_results, key=lambda x: valid_results[x]["val_accuracy"])
            
            # Sonuçları accuracy'ye göre sırala
            sorted_results = sorted(valid_results.items(), key=lambda x: x[1]["val_accuracy"], reverse=True)
            
            for i, (model_name, result) in enumerate(sorted_results, 1):
                symbol = "🏆" if model_name == best_model else "  "
                f.write(f"{i:<5} {symbol}{model_name:<18} {result['val_accuracy']:.4f}       {result['val_loss']:.4f}       {result['params']:,}\n")
            
            f.write("\n" + "="*50 + "\n")
            f.write(f"🏆 EN İYİ MODEL: {best_model}\n")
            f.write(f"   Validation Accuracy: {valid_results[best_model]['val_accuracy']:.4f}\n")
            f.write(f"   Validation Loss: {valid_results[best_model]['val_loss']:.4f}\n")
            f.write(f"   Model Parametreleri: {valid_results[best_model]['params']:,}\n")
            f.write("="*50 + "\n\n")
            
            # Detaylı sonuçlar
            f.write("DETAYLI MODEL SONUÇLARI:\n")
            f.write("-"*80 + "\n")
            
            for model_name, result in sorted_results:
                f.write(f"\n{model_name.upper()}:\n")
                f.write(f"  Validation Accuracy: {result['val_accuracy']:.4f}\n")
                f.write(f"  Validation Loss: {result['val_loss']:.4f}\n")
                f.write(f"  Model Parametreleri: {result['params']:,}\n")
                
                # Eğitim geçmişi varsa
                if 'history' in result:
                    history = result['history']
                    if 'accuracy' in history and 'val_accuracy' in history:
                        final_train_acc = history['accuracy'][-1]
                        final_val_acc = history['val_accuracy'][-1]
                        f.write(f"  Son Epoch Train Accuracy: {final_train_acc:.4f}\n")
                        f.write(f"  Son Epoch Val Accuracy: {final_val_acc:.4f}\n")
                        
                        # Overfitting kontrolü
                        if final_train_acc - final_val_acc > 0.1:
                            f.write(f"  ⚠️  Overfitting riski var (fark: {final_train_acc - final_val_acc:.4f})\n")
                        else:
                            f.write(f"  ✅ Overfitting riski düşük\n")
                
                f.write("  " + "-"*50 + "\n")
        
        # Hatalı modeller
        error_models = {k: v for k, v in results.items() if "error" in v}
        if error_models:
            f.write("\n\nHATALI MODELLER:\n")
            f.write("-"*80 + "\n")
            for model_name, error in error_models.items():
                f.write(f"❌ {model_name}: {error['error']}\n")
        
        f.write("\n\nOLUŞTURULAN DOSYALAR:\n")
        f.write("-"*40 + "\n")
        for model_name in valid_results.keys():
            f.write(f"- ilac_model_{model_name}.h5\n")
            f.write(f"- class_names_{model_name}.json\n")
        
        if valid_results:
            f.write(f"- ilac_model.h5 (en iyi model: {best_model})\n")
            f.write(f"- class_names.json (en iyi model için)\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("RAPOR SONU\n")
        f.write("="*80 + "\n")

# Tüm modelleri eğit ve karşılaştır
def train_all_models():
    models_config = {
        "basic_cnn": {
            "func": create_basic_cnn,
            "train_gen": train_gen_96,
            "val_gen": val_gen_96,
            "epochs": 15
        },
        "advanced_cnn": {
            "func": create_advanced_cnn,
            "train_gen": train_gen_96,
            "val_gen": val_gen_96,
            "epochs": 20
        },
        "ensemble_cnn": {
            "func": create_ensemble_cnn,
            "train_gen": train_gen_96,
            "val_gen": val_gen_96,
            "epochs": 25
        },
        "vgg16": {
            "func": create_vgg16_model,
            "train_gen": train_gen_224,
            "val_gen": val_gen_224,
            "epochs": 10
        },
        "resnet50": {
            "func": create_resnet50_model,
            "train_gen": train_gen_224,
            "val_gen": val_gen_224,
            "epochs": 10
        },
        "mobilenet": {
            "func": create_mobilenet_model,
            "train_gen": train_gen_224,
            "val_gen": val_gen_224,
            "epochs": 10
        },
        "efficientnet_fixed": {  # Yeni isim
            "func": create_efficientnet_model,
            "train_gen": train_gen_224,
            "val_gen": val_gen_224,
            "epochs": 12
        },
        "vision_transformer_fixed": {  # Yeni isim
            "func": create_vision_transformer,
            "train_gen": train_gen_96,
            "val_gen": val_gen_96,
            "epochs": 25  # Epoch sayısını azalttık
        }
    }
    
    results = {}
    total_models = len(models_config)
    
    for i, (model_name, config) in enumerate(models_config.items(), 1):
        print(f"\n{'='*50}")
        print(f"[{i}/{total_models}] {model_name.upper()} EĞİTİLİYOR...")
        print(f"{'='*50}")
        
        try:
            # Model oluştur
            model = config["func"](num_classes)
            model.compile(
                optimizer='adam', 
                loss='categorical_crossentropy', 
                metrics=['accuracy']
            )
            
            print(f"Model parametreleri: {model.count_params():,}")
            
            # Model eğit
            history = model.fit(
                config["train_gen"],
                validation_data=config["val_gen"],
                epochs=config["epochs"],
                verbose=1,
                callbacks=[
                    tf.keras.callbacks.EarlyStopping(
                        monitor='val_accuracy',
                        patience=5,
                        restore_best_weights=True
                    )
                ]
            )
            
            # Model değerlendir
            val_loss, val_acc = model.evaluate(config["val_gen"], verbose=0)
            results[model_name] = {
                "val_loss": val_loss,
                "val_accuracy": val_acc,
                "params": model.count_params(),
                "history": history.history
            }
            
            # Model kaydet
            model.save(f"ilac_model_{model_name}.h5")
            
            # Sınıf isimlerini de kaydet
            with open(f"class_names_{model_name}.json", "w", encoding="utf-8") as f:
                json.dump(class_names, f, ensure_ascii=False, indent=2)
            
            print(f"✅ {model_name} - Validation Accuracy: {val_acc:.4f}")
            
        except Exception as e:
            print(f"❌ {model_name} eğitiminde hata: {str(e)}")
            results[model_name] = {"error": str(e)}
            continue
    
    # Sonuçları text dosyasına yaz
    write_results_to_file(results, "model_results_fixed.txt")
    
    # Konsol çıktısı
    print(f"\n{'='*60}")
    print("TÜM MODEL SONUÇLARI (DÜZELTİLMİŞ)")
    print(f"{'='*60}")
    
    valid_results = {k: v for k, v in results.items() if "error" not in v}
    
    if valid_results:
        # En iyi modeli bul
        best_model = max(valid_results, key=lambda x: valid_results[x]["val_accuracy"])
        
        print(f"{'Model':<25} {'Accuracy':<12} {'Loss':<12} {'Parametreler':<15}")
        print("-" * 70)
        
        for model_name, result in sorted(valid_results.items(), key=lambda x: x[1]["val_accuracy"], reverse=True):
            symbol = "🏆" if model_name == best_model else "  "
            print(f"{symbol}{model_name:<23} {result['val_accuracy']:.4f}       {result['val_loss']:.4f}       {result['params']:,}")
        
        print(f"\n🏆 EN İYİ MODEL: {best_model}")
        print(f"   Accuracy: {valid_results[best_model]['val_accuracy']:.4f}")
        print(f"   Parametreler: {valid_results[best_model]['params']:,}")
        
        # En iyi modeli varsayılan olarak kopyala
        os.system(f"cp ilac_model_{best_model}.h5 ilac_model.h5")
        os.system(f"cp class_names_{best_model}.json class_names.json")
        print(f"✅ En iyi model ilac_model.h5 olarak kaydedildi")
    
    # Hata olan modeller
    error_models = {k: v for k, v in results.items() if "error" in v}
    if error_models:
        print(f"\n❌ HATA OLAN MODELLER:")
        for model_name, error in error_models.items():
            print(f"   {model_name}: {error['error']}")
    
    print(f"\n📄 Detaylı sonuçlar 'model_results_fixed.txt' dosyasına kaydedildi")
    
    return results

# Tüm modelleri eğit
if __name__ == "__main__":
    print(f"Toplam {num_classes} sınıf bulundu: {list(class_names.values())}")
    print("\n🚀 8 model eğitiliyor...")
    
    results = train_all_models()
    
    print("\n🎉 Eğitim tamamlandı!")
    print("📁 Oluşturulan dosyalar:")
    print("- model_results.txt (detaylı sonuçlar)")
    print("- ilac_model_[model_name].h5 (her model için)")
    print("- class_names_[model_name].json (her model için)")
    print("- ilac_model.h5 (en iyi model)")
    print("- class_names.json (en iyi model için)")