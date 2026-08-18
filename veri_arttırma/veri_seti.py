import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array, save_img

# Kaynak ve hedef klasörler
dataset_dir = "../sunucu/dataset"       # ilac_A, ilac_B klasörleri
augmented_dir = "augmented_dataset"

# Oluştur, yoksa
if not os.path.exists(augmented_dir):
    os.makedirs(augmented_dir)

# Data augmentation ayarları
datagen = ImageDataGenerator(
    rotation_range=30,      # -30/+30 derece döndür
    width_shift_range=0.1,  # yatay kaydır
    height_shift_range=0.1, # dikey kaydır
    shear_range=0.1,        # kesme
    zoom_range=0.2,         # yakınlaştır
    horizontal_flip=True,   # yatay çevir
    brightness_range=[0.7,1.3],  # parlaklık
    fill_mode='nearest'
)

# Her klasör için
for label in os.listdir(dataset_dir):
    label_path = os.path.join(dataset_dir, label)
    save_path = os.path.join(augmented_dir, label)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    for img_name in os.listdir(label_path):
        img_path = os.path.join(label_path, img_name)
        img = load_img(img_path)                # yükle
        x = img_to_array(img)                   # array’e çevir
        x = x.reshape((1,) + x.shape)           # batch boyutu ekle

        i = 0
        for batch in datagen.flow(x, batch_size=1, save_to_dir=save_path, save_prefix='aug', save_format='jpg'):
            i += 1
            if i >= 10:  # her resimden 10 tane üret
                break
