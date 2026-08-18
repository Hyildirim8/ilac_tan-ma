from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi import Request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
import os
import datetime
import uvicorn
import glob
import json

app = FastAPI()

# Static klasörü varsa mount et
if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

UPLOAD_DIR = "uploads"
MODEL_DIR = "model"  # Model klasörü
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# Templates klasörünü de kontrol et
if not os.path.exists("templates"):
    os.makedirs("templates", exist_ok=True)

templates = Jinja2Templates(directory="templates")

# Tüm modelleri yükle
models_ensemble = {}
class_indices = {}

def load_all_models():
    global models_ensemble, class_indices
    
    print(f"🔍 Model klasörü kontrol ediliyor: {MODEL_DIR}")
    
    # Model klasöründeki tüm .h5 dosyalarını bul
    model_files = glob.glob(os.path.join(MODEL_DIR, "ilac_model_*.h5"))
    
    # Eğer bulunamazsa ana dizini kontrol et
    if not model_files:
        print("❌ Model klasöründe model bulunamadı, ana dizin kontrol ediliyor...")
        model_files = glob.glob("ilac_model_*.h5")
        MODEL_DIR_CURRENT = "."
    else:
        MODEL_DIR_CURRENT = MODEL_DIR
    
    print(f"📁 Bulunan model dosyaları: {len(model_files)}")
    
    if not model_files:
        print("❌ Hiç model dosyası bulunamadı!")
        return
    
    for model_file in model_files:
        model_name = os.path.basename(model_file).replace("ilac_model_", "").replace(".h5", "")
        
        try:
            print(f"🔄 Yükleniyor: {model_file}")
            model = load_model(model_file)
            models_ensemble[model_name] = model
            print(f"✅ {model_name} modeli yüklendi (parametre sayısı: {model.count_params():,})")
        except Exception as e:
            print(f"❌ {model_name} yüklenirken hata: {e}")
            continue
    
    # Sınıf isimlerini yükle - önce model klasöründen, sonra ana dizinden
    class_files_to_try = [
        os.path.join(MODEL_DIR, "class_names.json"),
        "class_names.json"
    ]
    
    class_names_loaded = False
    for class_file in class_files_to_try:
        try:
            if os.path.exists(class_file):
                with open(class_file, "r", encoding="utf-8") as f:
                    class_indices = json.load(f)
                class_indices = {int(k): v for k, v in class_indices.items()}
                print(f"✅ Sınıf isimleri yüklendi ({class_file}): {class_indices}")
                class_names_loaded = True
                break
        except Exception as e:
            print(f"❌ {class_file} yüklenirken hata: {e}")
            continue
    
    if not class_names_loaded:
        print("⚠️ class_names.json bulunamadı, varsayılan isimler kullanılıyor")
        class_indices = {0: 'Lansoprol', 1: 'Napren'}  # Varsayılan
    
    print(f"🎯 Toplam yüklenen model sayısı: {len(models_ensemble)}")
    print(f"📋 Model isimleri: {list(models_ensemble.keys())}")

def ensemble_predict(image_path):
    """Tüm modellerin tahminlerini birleştir"""
    if not models_ensemble:
        raise Exception("Hiç model yüklenmedi")
    
    predictions = {}
    all_preds = []
    successful_models = 0
    
    print(f"\n🔮 Ensemble tahmin başlıyor - {len(models_ensemble)} model kullanılacak")
    
    for model_name, model in models_ensemble.items():
        try:
            # Model giriş boyutunu kontrol et
            input_shape = model.input_shape[1:3]  # (height, width)
            
            # Resmi uygun boyuta getir
            img = load_img(image_path, target_size=input_shape)
            x = img_to_array(img) / 255.0
            x = np.expand_dims(x, axis=0)
            
            # Tahmin yap
            pred = model.predict(x, verbose=0)
            
            class_id = int(np.argmax(pred[0]))
            confidence = float(pred[0][class_id])
            class_name = class_indices.get(class_id, f"Unknown_{class_id}")
            
            predictions[model_name] = {
                "probabilities": pred[0].tolist(),
                "class_id": class_id,
                "class_name": class_name,
                "confidence": confidence,
                "input_size": input_shape
            }
            
            all_preds.append(pred[0])
            successful_models += 1
            
            print(f"   ✅ {model_name}: {class_name} ({confidence:.3f}) [input: {input_shape}]")
            
        except Exception as e:
            print(f"   ❌ {model_name} tahmin hatası: {e}")
            continue
    
    if not all_preds:
        raise Exception("Hiç model tahmin yapamadı")
    
    print(f"📊 Başarılı tahmin: {successful_models}/{len(models_ensemble)} model")
    
    # Ensemble tahmin - ortalama al
    ensemble_pred = np.mean(all_preds, axis=0)
    ensemble_class_id = int(np.argmax(ensemble_pred))
    ensemble_confidence = float(ensemble_pred[ensemble_class_id])
    ensemble_class_name = class_indices.get(ensemble_class_id, f"Unknown_{ensemble_class_id}")
    
    # Voting ensemble - çoğunluk oyu
    votes = [predictions[model]['class_id'] for model in predictions.keys()]
    from collections import Counter
    vote_counts = Counter(votes)
    voting_class_id = vote_counts.most_common(1)[0][0]
    voting_class_name = class_indices.get(voting_class_id, f"Unknown_{voting_class_id}")
    voting_confidence = vote_counts[voting_class_id] / len(votes)
    
    # En yüksek güvenli modeli bul
    best_individual = max(predictions.items(), key=lambda x: x[1]['confidence'])
    best_model_name, best_prediction = best_individual
    
    return {
        "individual_predictions": predictions,
        "ensemble_average": {
            "class": ensemble_class_name,
            "class_id": ensemble_class_id,
            "confidence": ensemble_confidence,
            "method": "average"
        },
        "ensemble_voting": {
            "class": voting_class_name, 
            "class_id": voting_class_id,
            "confidence": voting_confidence,
            "votes": dict(vote_counts),
            "method": "voting"
        },
        "best_individual": {
            "class": best_prediction['class_name'],
            "class_id": best_prediction['class_id'],
            "confidence": best_prediction['confidence'],
            "model": best_model_name,
            "method": "best_individual"
        },
        "model_count": len(predictions),
        "successful_models": successful_models,
        "all_class_probabilities": {
            class_indices[i]: float(ensemble_pred[i]) 
            for i in range(len(ensemble_pred))
        }
    }

# Modelleri yükle
print("🚀 Modeller yükleniyor...")
load_all_models()

# Son tahmin sonucunu saklamak için
last_prediction = {"class": "Henüz resim yok", "confidence": 0.0, "timestamp": None}

def clear_old_images():
    """Eski resimleri sil, sadece son resmi sakla"""
    try:
        image_files = glob.glob(os.path.join(UPLOAD_DIR, "*.jpg"))
        for file_path in image_files:
            if not file_path.endswith("latest_image.jpg"):
                os.remove(file_path)
    except Exception as e:
        print(f"Dosya silme hatası: {e}")

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {
        "request": request, 
        "prediction": last_prediction,
        "model_count": len(models_ensemble),
        "available_models": list(models_ensemble.keys()),
        "class_names": list(class_indices.values())
    })

@app.get("/latest_prediction")
def get_latest_prediction():
    return JSONResponse(content=last_prediction)

@app.get("/models_info")
def get_models_info():
    """Model bilgilerini döndür"""
    model_info = {}
    for name, model in models_ensemble.items():
        model_info[name] = {
            "input_shape": model.input_shape,
            "params": model.count_params(),
            "layers": len(model.layers)
        }
    
    return {
        "total_models": len(models_ensemble),
        "model_names": list(models_ensemble.keys()),
        "class_indices": class_indices,
        "model_details": model_info
    }

@app.post("/upload_esp")
async def predict(request: Request):
    global last_prediction
    
    if not models_ensemble:
        raise HTTPException(status_code=500, detail="Hiç model yüklenmedi")
    
    try:
        content_type = request.headers.get("content-type", "")
        
        if "multipart/form-data" in content_type:
            form = await request.form()
            file = form.get("file")
            if not file:
                raise HTTPException(status_code=400, detail="No file uploaded")
            file_content = await file.read()
        elif "application/octet-stream" in content_type or "image/" in content_type:
            file_content = await request.body()
        else:
            raise HTTPException(status_code=400, detail="Unsupported content type")

        if len(file_content) == 0:
            raise HTTPException(status_code=400, detail="Empty file")

        # Eski resimleri sil
        clear_old_images()

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        latest_path = os.path.join(UPLOAD_DIR, "latest_image.jpg")
        
        with open(latest_path, "wb") as f:
            f.write(file_content)

        print(f"📸 Son resim kaydedildi: {latest_path}")

        # Ensemble tahmin yap
        try:
            ensemble_results = ensemble_predict(latest_path)
            
            # En iyi sonucu seç (en yüksek güvenli model)
            best_result = ensemble_results["best_individual"]
            average_result = ensemble_results["ensemble_average"]
            voting_result = ensemble_results["ensemble_voting"]
            
            print(f"\n🏆 EN İYİ TAHMİN SONUÇLARI:")
            print(f"🥇 En İyi Model ({best_result['model']}): {best_result['class']} ({best_result['confidence']:.3f})")
            print(f"📊 Ortalama Ensemble: {average_result['class']} ({average_result['confidence']:.3f})")
            print(f"🗳️  Çoğunluk Oyu: {voting_result['class']} ({voting_result['confidence']:.3f})")
            print(f"📈 Kullanılan model: {ensemble_results['successful_models']}/{ensemble_results['model_count']}")
            
            # En güvenli sonucu kullan
            final_result = best_result if best_result['confidence'] > 0.8 else average_result
            
            last_prediction = {
                "class": final_result['class'],
                "confidence": round(final_result['confidence'], 4),
                "timestamp": timestamp,
                "image_path": f"/uploads/latest_image.jpg",
                "ensemble_details": ensemble_results,
                "method": final_result.get('method', 'ensemble'),
                "model_used": final_result.get('model', 'ensemble'),
                "all_probabilities": ensemble_results['all_class_probabilities']
            }

            return {
                "class": final_result['class'],
                "confidence": round(final_result['confidence'], 4),
                "status": "success",
                "timestamp": timestamp,
                "ensemble_results": ensemble_results,
                "model_count": ensemble_results['successful_models'],
                "method": final_result.get('method', 'ensemble'),
                "all_probabilities": ensemble_results['all_class_probabilities']
            }
            
        except Exception as prediction_error:
            print(f"❌ Ensemble tahmin hatası: {prediction_error}")
            raise HTTPException(status_code=500, detail=f"Prediction error: {str(prediction_error)}")
    
    except Exception as e:
        error_msg = f"Upload error: {str(e)}"
        print(error_msg)
        raise HTTPException(status_code=500, detail=error_msg)

app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")

if __name__ == "__main__":
    print(f"\n🎉 Sistem Hazır!")
    print(f"🚀 Yüklenen model sayısı: {len(models_ensemble)}")
    print(f"📋 Sınıflar: {list(class_indices.values())}")
    print(f"🌐 Server başlatılıyor: http://0.0.0.0:5000")
    uvicorn.run(app, host="0.0.0.0", port=5000)