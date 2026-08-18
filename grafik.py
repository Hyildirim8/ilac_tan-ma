import matplotlib.pyplot as plt
import numpy as np

# Model sonuçları (senin txt içeriğine göre)
models = ["basic_cnn", "advanced_cnn", "ensemble_cnn", "vgg16", "resnet50", "mobilenet", "efficientnet_fixed", "vision_transformer"]
accuracies = [0.92, 0.95, 0.96, 0.88, 0.90, 0.91, 0.89, 0.87]
params = [100000, 450000, 780000, 15000000, 23000000, 3500000, 4800000, 950000]

# ---- GRAFİK 1: Accuracy ----
plt.figure(figsize=(10,6))
bars = plt.bar(models, accuracies)
plt.title("Validation Accuracy Comparison", fontsize=14, fontweight="bold")
plt.xlabel("Models")
plt.ylabel("Validation Accuracy")
plt.ylim(0, 1.0)

# Bar üstüne değer yaz
for bar, acc in zip(bars, accuracies):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
             f"{acc:.2f}", ha="center", va="bottom", fontsize=10)

plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig("accuracy_C.png", dpi=300)
plt.close()

# ---- GRAFİK 2: Parametre Sayısı (Log Scale) ----
plt.figure(figsize=(10,6))
bars = plt.bar(models, params)
plt.yscale("log")
plt.title("Model Parameters (Log Scale)", fontsize=14, fontweight="bold")
plt.xlabel("Models")
plt.ylabel("Parameters (log scale)")

# Bar üstüne değer yaz
for bar, val in zip(bars, params):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
             f"{val/1000:.1f}K", ha="center", va="bottom", fontsize=10)

plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig("params_C.png", dpi=300)
plt.close()

print("✅ accuracy_C.png ve params_C.png başarıyla oluşturuldu.")
