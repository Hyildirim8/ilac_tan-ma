import os
import io
import glob
import json
import shutil
import datetime
import threading
from collections import Counter

from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.concurrency import run_in_threadpool
from PIL import Image
import numpy as np
import uvicorn

import model_training

APP_ROOT = os.path.dirname(__file__)
STATIC_DIR = os.path.join(APP_ROOT, "static")
UPLOAD_DIR = os.path.join(APP_ROOT, "uploads")
DATASET_DIR = os.path.join(APP_ROOT, "dataset")
AUGMENTED_DIR = os.path.join(APP_ROOT, "augmented_dataset")
MODEL_DIR = os.path.join(APP_ROOT, "model")
LATEST_PATH = os.path.join(STATIC_DIR, "latest.jpg")

os.makedirs(STATIC_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(DATASET_DIR, exist_ok=True)
os.makedirs(AUGMENTED_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

app = FastAPI()
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")
templates = Jinja2Templates(directory=os.path.join(APP_ROOT, "templates"))

# ---- Model yükleme (tahmin_sitesi'nden birleştirildi) ----

models_ensemble = {}
class_indices = {}
last_prediction = {"class": "Henüz resim yok", "confidence": 0.0, "timestamp": None}


def load_all_models():
    global models_ensemble, class_indices

    print(f"Model klasörü kontrol ediliyor: {MODEL_DIR}")
    model_files = glob.glob(os.path.join(MODEL_DIR, "ilac_model_*.h5"))

    if not model_files:
        print("Model klasöründe .h5 bulunamadı — tahmin devre dışı, sadece veri toplama modu aktif.")
    else:
        for model_file in model_files:
            model_name = os.path.basename(model_file).replace("ilac_model_", "").replace(".h5", "")
            try:
                from tensorflow.keras.models import load_model
                model = load_model(model_file)
                models_ensemble[model_name] = model
                print(f"Yüklendi: {model_name} (parametre: {model.count_params():,})")
            except Exception as e:
                print(f"{model_name} yüklenirken hata: {e}")

    class_files_to_try = [
        os.path.join(MODEL_DIR, "class_names.json"),
        os.path.join(APP_ROOT, "class_names.json"),
    ]
    for class_file in class_files_to_try:
        if os.path.exists(class_file):
            try:
                with open(class_file, "r", encoding="utf-8") as f:
                    loaded = json.load(f)
                class_indices = {int(k): v for k, v in loaded.items()}
                print(f"Sınıf isimleri yüklendi: {class_indices}")
                break
            except Exception as e:
                print(f"{class_file} yüklenirken hata: {e}")

    print(f"Toplam yüklenen model sayısı: {len(models_ensemble)}")


def ensemble_predict(image_path):
    from tensorflow.keras.preprocessing.image import img_to_array, load_img

    predictions = {}
    all_preds = []

    for model_name, model in models_ensemble.items():
        try:
            input_shape = model.input_shape[1:3]
            img = load_img(image_path, target_size=input_shape)
            x = img_to_array(img) / 255.0
            x = np.expand_dims(x, axis=0)

            pred = model.predict(x, verbose=0)
            class_id = int(np.argmax(pred[0]))
            confidence = float(pred[0][class_id])
            class_name = class_indices.get(class_id, f"Unknown_{class_id}")

            predictions[model_name] = {
                "probabilities": pred[0].tolist(),
                "class_id": class_id,
                "class_name": class_name,
                "confidence": confidence,
            }
            all_preds.append(pred[0])
        except Exception as e:
            print(f"{model_name} tahmin hatası: {e}")

    if not all_preds:
        raise Exception("Hiç model tahmin yapamadı")

    ensemble_pred = np.mean(all_preds, axis=0)
    ensemble_class_id = int(np.argmax(ensemble_pred))
    ensemble_class_name = class_indices.get(ensemble_class_id, f"Unknown_{ensemble_class_id}")

    votes = [p["class_id"] for p in predictions.values()]
    vote_counts = Counter(votes)
    voting_class_id = vote_counts.most_common(1)[0][0]
    voting_class_name = class_indices.get(voting_class_id, f"Unknown_{voting_class_id}")

    best_model_name, best_prediction = max(predictions.items(), key=lambda x: x[1]["confidence"])

    return {
        "individual_predictions": predictions,
        "ensemble_average": {
            "class": ensemble_class_name,
            "class_id": ensemble_class_id,
            "confidence": float(ensemble_pred[ensemble_class_id]),
            "method": "average",
        },
        "ensemble_voting": {
            "class": voting_class_name,
            "class_id": voting_class_id,
            "confidence": vote_counts[voting_class_id] / len(votes),
            "votes": {class_indices.get(k, str(k)): v for k, v in vote_counts.items()},
            "method": "voting",
        },
        "best_individual": {
            "class": best_prediction["class_name"],
            "class_id": best_prediction["class_id"],
            "confidence": best_prediction["confidence"],
            "model": best_model_name,
            "method": "best_individual",
        },
        "model_count": len(predictions),
    }


print("Modeller yükleniyor...")
load_all_models()

# ---- Model eğitimi (web sayfasından tetiklenir) ----

training_lock = threading.Lock()
training_status = {
    "running": False,
    "current_model": None,
    "current_epoch": 0,
    "total_epochs": 0,
    "log": [],
    "done": False,
    "error": None,
    "best_model": None,
}


def _pick_training_dataset_dir():
    """Eğitim için augmented_dataset'i tercih eder (yeterli veri varsa),
    yoksa ham dataset'e düşer."""

    def class_counts(base_dir):
        counts = {}
        if os.path.isdir(base_dir):
            for name in os.listdir(base_dir):
                folder = os.path.join(base_dir, name)
                if os.path.isdir(folder):
                    counts[name] = len([f for f in os.listdir(folder) if f.lower().endswith((".jpg", ".jpeg", ".png"))])
        return counts

    augmented_counts = class_counts(AUGMENTED_DIR)
    if sum(1 for c in augmented_counts.values() if c > 0) >= 2:
        return AUGMENTED_DIR, augmented_counts

    dataset_counts = class_counts(DATASET_DIR)
    return DATASET_DIR, dataset_counts


def _training_progress(update):
    event = update.get("event")
    if event == "start":
        training_status["log"].append(
            f"{update['num_classes']} sınıf bulundu: {', '.join(update['classes'])}"
        )
    elif event == "model_start":
        training_status["current_model"] = update["model"]
        training_status["current_epoch"] = 0
        training_status["total_epochs"] = update["total_epochs"]
        training_status["log"].append(f"[{update['model']}] eğitim başladı")
    elif event == "epoch_end":
        training_status["current_model"] = update["model"]
        training_status["current_epoch"] = update["epoch"]
        training_status["total_epochs"] = update["total_epochs"]
        logs = update.get("logs", {})
        acc = logs.get("accuracy")
        val_acc = logs.get("val_accuracy")
        training_status["log"].append(
            f"[{update['model']}] epoch {update['epoch']}/{update['total_epochs']}"
            + (f" - acc: {acc:.4f}" if acc is not None else "")
            + (f" - val_acc: {val_acc:.4f}" if val_acc is not None else "")
        )
    elif event == "model_end":
        training_status["log"].append(
            f"[{update['model']}] tamamlandı - val_accuracy: {update['val_accuracy']:.4f}"
        )
    elif event == "model_error":
        training_status["log"].append(f"[{update['model']}] HATA: {update['error']}")
    elif event == "done":
        training_status["log"].append(f"Eğitim tamamlandı. En iyi model: {update['best_model']}")
        training_status["best_model"] = update["best_model"]

    training_status["log"] = training_status["log"][-200:]


def _run_training_job():
    training_status.update({
        "running": True, "done": False, "error": None, "best_model": None,
        "current_model": None, "current_epoch": 0, "total_epochs": 0, "log": [],
    })
    try:
        dataset_dir, _ = _pick_training_dataset_dir()
        model_training.train_all_models(dataset_dir, MODEL_DIR, on_progress=_training_progress)
        load_all_models()
    except Exception as e:
        training_status["error"] = str(e)
        training_status["log"].append(f"HATA: {e}")
    finally:
        training_status["running"] = False
        training_status["done"] = True


@app.post("/train")
async def start_training():
    if training_status["running"]:
        return JSONResponse({"status": "error", "detail": "Eğitim zaten çalışıyor"}, status_code=409)

    dataset_dir, counts = _pick_training_dataset_dir()
    usable_classes = {k: v for k, v in counts.items() if v > 0}
    if len(usable_classes) < 2:
        return JSONResponse(
            {"status": "error", "detail": "Eğitim için en az 2 sınıfta görüntü gerekli. Önce veri ekleyip veri artırma yapın."},
            status_code=400,
        )

    threading.Thread(target=_run_training_job, daemon=True).start()
    return JSONResponse({"status": "ok", "dataset_dir": os.path.basename(dataset_dir), "classes": usable_classes})


@app.get("/train_status")
async def get_training_status():
    return JSONResponse(training_status)


# ---- Ortak endpoint'ler ----


@app.get("/", response_class=HTMLResponse)
async def veri_sayfasi(request: Request):
    return templates.TemplateResponse(
        "veri.html",
        {"request": request, "active_tab": "veri"},
    )


@app.get("/tahmin", response_class=HTMLResponse)
async def tahmin_sayfasi(request: Request):
    return templates.TemplateResponse(
        "tahmin.html",
        {
            "request": request,
            "active_tab": "tahmin",
            "prediction": last_prediction,
            "model_count": len(models_ensemble),
            "available_models": list(models_ensemble.keys()),
        },
    )


@app.get("/sonuclar", response_class=HTMLResponse)
async def sonuclar_sayfasi(request: Request):
    return templates.TemplateResponse(
        "sonuclar.html",
        {"request": request, "active_tab": "sonuclar"},
    )


# ESP32 buraya POST eder (raw image/jpeg gövde). Görüntüyü kaydeder,
# model varsa tahmin de üretir; yoksa sadece veri toplama moduna düşer.
@app.post("/upload_esp")
async def upload_esp(request: Request):
    global last_prediction
    try:
        contents = await request.body()
        if not contents:
            return JSONResponse({"status": "error", "detail": "Empty request body"}, status_code=400)

        image = Image.open(io.BytesIO(contents)).convert("RGB")
        image.save(LATEST_PATH, format="JPEG", quality=85)

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        upload_path = os.path.join(UPLOAD_DIR, "latest_image.jpg")
        image.save(upload_path, format="JPEG", quality=90)

        result = {"status": "ok", "saved": LATEST_PATH, "timestamp": timestamp}

        if models_ensemble:
            try:
                ensemble_results = ensemble_predict(upload_path)
                best_result = ensemble_results["best_individual"]
                average_result = ensemble_results["ensemble_average"]
                final_result = best_result if best_result["confidence"] > 0.8 else average_result

                last_prediction = {
                    "class": final_result["class"],
                    "confidence": round(final_result["confidence"], 4),
                    "timestamp": timestamp,
                    "image_path": "/uploads/latest_image.jpg",
                    "ensemble_details": ensemble_results,
                    "method": final_result.get("method", "ensemble"),
                }
                result["prediction"] = last_prediction
            except Exception as e:
                print(f"Tahmin hatası: {e}")

        return JSONResponse(result)
    except Exception as e:
        return JSONResponse({"status": "error", "detail": str(e)}, status_code=400)


# Web arayüzden "Kaydet" butonuna basınca en son görüntü veri setine kopyalanır
@app.post("/save")
async def save_image(drug_name: str = Form(...)):
    if not os.path.exists(LATEST_PATH):
        return JSONResponse({"status": "error", "detail": "No latest image"}, status_code=404)
    safe_name = "".join(c for c in drug_name if c.isalnum() or c in (" ", "_", "-")).strip().replace(" ", "_")
    folder = os.path.join(DATASET_DIR, safe_name)
    os.makedirs(folder, exist_ok=True)
    count = len([f for f in os.listdir(folder) if f.lower().endswith((".jpg", ".jpeg", ".png"))])
    dest = os.path.join(folder, f"{count + 1}.jpg")
    shutil.copy(LATEST_PATH, dest)
    return JSONResponse({"status": "ok", "saved_to": dest})


@app.get("/latest.jpg")
async def get_latest():
    if os.path.exists(LATEST_PATH):
        return FileResponse(LATEST_PATH, media_type="image/jpeg")
    return JSONResponse({"status": "error", "detail": "No image"}, status_code=404)


@app.get("/latest_prediction")
async def get_latest_prediction():
    return JSONResponse(content=last_prediction)


@app.get("/api/dataset_summary")
async def dataset_summary():
    classes = {}
    if os.path.isdir(DATASET_DIR):
        for name in sorted(os.listdir(DATASET_DIR)):
            folder = os.path.join(DATASET_DIR, name)
            if os.path.isdir(folder):
                count = len([f for f in os.listdir(folder) if f.lower().endswith((".jpg", ".jpeg", ".png"))])
                classes[name] = count
    return {"classes": classes}


@app.get("/api/model_results")
async def model_results():
    results_path = os.path.join(MODEL_DIR, "model_results.json")
    if not os.path.exists(results_path):
        return {"available": False, "results": {}}
    try:
        with open(results_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return {"available": True, "results": data}
    except Exception as e:
        return JSONResponse({"available": False, "results": {}, "detail": str(e)}, status_code=500)


def _run_augmentation():
    from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array

    datagen = ImageDataGenerator(
        rotation_range=30,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.2,
        horizontal_flip=True,
        brightness_range=[0.7, 1.3],
        fill_mode="nearest",
    )

    per_class = {}
    total_generated = 0

    if not os.path.isdir(DATASET_DIR):
        return {"per_class": per_class, "total_generated": 0}

    for label in sorted(os.listdir(DATASET_DIR)):
        label_path = os.path.join(DATASET_DIR, label)
        if not os.path.isdir(label_path):
            continue
        save_path = os.path.join(AUGMENTED_DIR, label)
        os.makedirs(save_path, exist_ok=True)

        image_files = [f for f in os.listdir(label_path) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        generated = 0
        for img_name in image_files:
            img_path = os.path.join(label_path, img_name)
            img = load_img(img_path)
            x = img_to_array(img)
            x = x.reshape((1,) + x.shape)

            i = 0
            for _ in datagen.flow(x, batch_size=1, save_to_dir=save_path, save_prefix="aug", save_format="jpg"):
                i += 1
                if i >= 10:
                    break
            generated += i

        per_class[label] = {"original": len(image_files), "generated": generated}
        total_generated += generated

    return {"per_class": per_class, "total_generated": total_generated}


# Web arayüzden tetiklenen veri artırma — dataset/ içindeki her görüntüden
# döndürme/kaydırma/yakınlaştırma/parlaklık varyasyonları üretir.
@app.post("/augment")
async def augment():
    try:
        result = await run_in_threadpool(_run_augmentation)
        return JSONResponse({"status": "ok", **result})
    except Exception as e:
        return JSONResponse({"status": "error", "detail": str(e)}, status_code=500)


@app.get("/models_info")
async def get_models_info():
    model_info = {}
    for name, model in models_ensemble.items():
        model_info[name] = {
            "input_shape": [d for d in model.input_shape],
            "params": model.count_params(),
            "layers": len(model.layers),
        }
    return {
        "total_models": len(models_ensemble),
        "model_names": list(models_ensemble.keys()),
        "class_indices": class_indices,
        "model_details": model_info,
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000)
