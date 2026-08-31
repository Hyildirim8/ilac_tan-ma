"""Web arayüzünden tetiklenen model eğitimi.

Model mimarileri yapay_zeka/app.py ile aynıdır; buradaki fark eğitimin
dataset_dir/output_dir parametreleriyle çağrılabilmesi ve her epoch/model
sonunda bir callback ile ilerleme bildirmesidir (web arayüzde canlı log
göstermek için main.py bunu kullanır).
"""

import os
import json
import datetime

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models, Model
from tensorflow.keras.applications import VGG16, ResNet50, MobileNetV2

MODELS_CONFIG = {
    "basic_cnn": {"image_size": 96, "epochs": 15},
    "advanced_cnn": {"image_size": 96, "epochs": 20},
    "ensemble_cnn": {"image_size": 96, "epochs": 25},
    "vgg16": {"image_size": 224, "epochs": 10},
    "resnet50": {"image_size": 224, "epochs": 10},
    "mobilenet": {"image_size": 224, "epochs": 10},
    "efficientnet_fixed": {"image_size": 224, "epochs": 12},
    "vision_transformer_fixed": {"image_size": 96, "epochs": 25},
}


def create_basic_cnn(num_classes):
    return models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(96, 96, 3)),
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D(2, 2),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax'),
    ])


def create_advanced_cnn(num_classes):
    return models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(96, 96, 3)),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2, 2),
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax'),
    ])


def create_ensemble_cnn(num_classes):
    input_layer = layers.Input(shape=(96, 96, 3))

    x1 = layers.Conv2D(32, (3, 3), activation='relu')(input_layer)
    x1 = layers.MaxPooling2D(2, 2)(x1)
    x1 = layers.Conv2D(64, (3, 3), activation='relu')(x1)
    x1 = layers.GlobalAveragePooling2D()(x1)
    x1 = layers.Dense(64, activation='relu')(x1)

    x2 = layers.Conv2D(32, (5, 5), activation='relu')(input_layer)
    x2 = layers.MaxPooling2D(2, 2)(x2)
    x2 = layers.Conv2D(64, (5, 5), activation='relu')(x2)
    x2 = layers.GlobalAveragePooling2D()(x2)
    x2 = layers.Dense(64, activation='relu')(x2)

    x3 = layers.Conv2D(16, (3, 3), activation='relu')(input_layer)
    x3 = layers.Conv2D(32, (3, 3), activation='relu')(x3)
    x3 = layers.MaxPooling2D(2, 2)(x3)
    x3 = layers.Conv2D(64, (3, 3), activation='relu')(x3)
    x3 = layers.GlobalAveragePooling2D()(x3)
    x3 = layers.Dense(64, activation='relu')(x3)

    combined = layers.concatenate([x1, x2, x3])
    combined = layers.Dense(128, activation='relu')(combined)
    combined = layers.Dropout(0.5)(combined)
    output = layers.Dense(num_classes, activation='softmax')(combined)

    return Model(inputs=input_layer, outputs=output)


def create_vgg16_model(num_classes):
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = False
    return models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax'),
    ])


def create_resnet50_model(num_classes):
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = False
    return models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax'),
    ])


def create_mobilenet_model(num_classes):
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = False
    return models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax'),
    ])


def create_efficientnet_model(num_classes):
    try:
        from tensorflow.keras.applications import EfficientNetV2B0
        base_model = EfficientNetV2B0(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    except Exception:
        base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = False
    return models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.4),
        layers.Dense(num_classes, activation='softmax'),
    ])


class PatchEmbedding(layers.Layer):
    def __init__(self, patch_size, d_model, **kwargs):
        super().__init__(**kwargs)
        self.patch_size = patch_size
        self.d_model = d_model

    def build(self, input_shape):
        self.patch_conv = layers.Conv2D(
            filters=self.d_model, kernel_size=self.patch_size,
            strides=self.patch_size, padding="valid", name="patch_conv",
        )
        super().build(input_shape)

    def call(self, images):
        patches = self.patch_conv(images)
        batch_size = tf.shape(patches)[0]
        num_patches = patches.shape[1] * patches.shape[2]
        return tf.reshape(patches, [batch_size, num_patches, self.d_model])

    def get_config(self):
        config = super().get_config()
        config.update({"patch_size": self.patch_size, "d_model": self.d_model})
        return config


def create_vision_transformer(num_classes, image_size=96, patch_size=16, num_layers=4, d_model=128, num_heads=4):
    inputs = layers.Input(shape=(image_size, image_size, 3))
    patches = PatchEmbedding(patch_size, d_model)(inputs)
    num_patches = (image_size // patch_size) ** 2

    position_embedding = layers.Embedding(input_dim=num_patches, output_dim=d_model, name="position_embedding")
    positions = tf.range(start=0, limit=num_patches, delta=1)
    encoded_patches = patches + position_embedding(positions)

    for i in range(num_layers):
        x1 = layers.LayerNormalization(epsilon=1e-6, name=f"norm1_{i}")(encoded_patches)
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=d_model, name=f"attention_{i}",
        )(x1, x1)
        x2 = layers.Add(name=f"add1_{i}")([attention_output, encoded_patches])

        x3 = layers.LayerNormalization(epsilon=1e-6, name=f"norm2_{i}")(x2)
        x3 = layers.Dense(d_model * 2, activation="gelu", name=f"mlp_dense1_{i}")(x3)
        x3 = layers.Dense(d_model, name=f"mlp_dense2_{i}")(x3)
        encoded_patches = layers.Add(name=f"add2_{i}")([x3, x2])

    representation = layers.LayerNormalization(epsilon=1e-6, name="final_norm")(encoded_patches)
    representation = layers.GlobalAveragePooling1D(name="global_avg_pool")(representation)
    representation = layers.Dropout(0.3, name="final_dropout")(representation)
    outputs = layers.Dense(num_classes, activation="softmax", name="classifier")(representation)

    return Model(inputs=inputs, outputs=outputs, name="vision_transformer")


MODEL_BUILDERS = {
    "basic_cnn": lambda n: create_basic_cnn(n),
    "advanced_cnn": lambda n: create_advanced_cnn(n),
    "ensemble_cnn": lambda n: create_ensemble_cnn(n),
    "vgg16": lambda n: create_vgg16_model(n),
    "resnet50": lambda n: create_resnet50_model(n),
    "mobilenet": lambda n: create_mobilenet_model(n),
    "efficientnet_fixed": lambda n: create_efficientnet_model(n),
    "vision_transformer_fixed": lambda n: create_vision_transformer(n),
}


class _ProgressCallback(tf.keras.callbacks.Callback):
    def __init__(self, on_progress, model_name, total_epochs):
        super().__init__()
        self.on_progress = on_progress
        self.model_name = model_name
        self.total_epochs = total_epochs

    def on_epoch_end(self, epoch, logs=None):
        self.on_progress({
            "event": "epoch_end",
            "model": self.model_name,
            "epoch": epoch + 1,
            "total_epochs": self.total_epochs,
            "logs": {k: float(v) for k, v in (logs or {}).items()},
        })


def train_all_models(dataset_dir, output_dir, on_progress=None):
    """dataset_dir altındaki sınıf klasörlerinden tüm modelleri eğitir ve
    sonuçları output_dir içine (ilac_model_<isim>.h5, class_names*.json,
    model_results.json) yazar. on_progress(dict) her önemli adımda çağrılır."""

    def emit(update):
        if on_progress:
            on_progress(update)

    os.makedirs(output_dir, exist_ok=True)

    datagen = ImageDataGenerator(rescale=1. / 255, validation_split=0.2)

    gens = {}
    for size in (96, 224):
        gens[size] = {
            "train": datagen.flow_from_directory(
                dataset_dir, target_size=(size, size), batch_size=16,
                class_mode='categorical', subset='training',
            ),
            "val": datagen.flow_from_directory(
                dataset_dir, target_size=(size, size), batch_size=16,
                class_mode='categorical', subset='validation',
            ),
        }

    train_gen_96 = gens[96]["train"]
    num_classes = len(train_gen_96.class_indices)
    class_names = {v: k for k, v in train_gen_96.class_indices.items()}

    emit({"event": "start", "num_classes": num_classes, "classes": list(class_names.values()),
          "models": list(MODELS_CONFIG.keys())})

    results = {}
    for model_name, cfg in MODELS_CONFIG.items():
        train_gen = gens[cfg["image_size"]]["train"]
        val_gen = gens[cfg["image_size"]]["val"]
        emit({"event": "model_start", "model": model_name, "total_epochs": cfg["epochs"]})

        try:
            model = MODEL_BUILDERS[model_name](num_classes)
            model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

            history = model.fit(
                train_gen,
                validation_data=val_gen,
                epochs=cfg["epochs"],
                verbose=0,
                callbacks=[
                    tf.keras.callbacks.EarlyStopping(
                        monitor='val_accuracy', patience=5, restore_best_weights=True,
                    ),
                    _ProgressCallback(emit, model_name, cfg["epochs"]),
                ],
            )

            val_loss, val_acc = model.evaluate(val_gen, verbose=0)
            results[model_name] = {
                "val_loss": float(val_loss),
                "val_accuracy": float(val_acc),
                "params": int(model.count_params()),
            }

            model.save(os.path.join(output_dir, f"ilac_model_{model_name}.h5"))
            with open(os.path.join(output_dir, f"class_names_{model_name}.json"), "w", encoding="utf-8") as f:
                json.dump(class_names, f, ensure_ascii=False, indent=2)

            emit({"event": "model_end", "model": model_name, **results[model_name]})
        except Exception as e:
            results[model_name] = {"error": str(e)}
            emit({"event": "model_error", "model": model_name, "error": str(e)})

    valid_results = {k: v for k, v in results.items() if "error" not in v}
    best_model = None
    if valid_results:
        best_model = max(valid_results, key=lambda k: valid_results[k]["val_accuracy"])
        import shutil
        shutil.copy(
            os.path.join(output_dir, f"ilac_model_{best_model}.h5"),
            os.path.join(output_dir, "ilac_model.h5"),
        )
        shutil.copy(
            os.path.join(output_dir, f"class_names_{best_model}.json"),
            os.path.join(output_dir, "class_names.json"),
        )

    with open(os.path.join(output_dir, "model_results.json"), "w", encoding="utf-8") as f:
        json.dump(valid_results, f, ensure_ascii=False, indent=2)

    emit({
        "event": "done",
        "best_model": best_model,
        "results": valid_results,
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    })

    return results
