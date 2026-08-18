import os
import io
import shutil
from fastapi import FastAPI, File, UploadFile, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from PIL import Image
import uvicorn

APP_ROOT = os.path.dirname(__file__)
STATIC_DIR = os.path.join(APP_ROOT, "static")
DATASET_DIR = os.path.join(APP_ROOT, "dataset")
LATEST_PATH = os.path.join(STATIC_DIR, "latest.jpg")

os.makedirs(STATIC_DIR, exist_ok=True)
os.makedirs(DATASET_DIR, exist_ok=True)

app = FastAPI()
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# ESP32 buraya POST etsin (image/jpeg) => kaydet ve frontend'e hazır olsun
@app.post("/upload_esp")
async def upload_esp(request: Request):
    try:
        # Raw bytes olarak al
        contents = await request.body()
        
        if not contents:
            return JSONResponse({"status": "error", "detail": "Empty request body"}, status_code=400)
            
        # Resmi PIL ile aç ve kaydet
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        image.save(LATEST_PATH, format="JPEG", quality=85)
        return JSONResponse({"status": "ok", "saved": LATEST_PATH})
    except Exception as e:
        return JSONResponse({"status": "error", "detail": str(e)}, status_code=400)

# Web arayüz "Kaydet" butonuna bastığında buraya POST atıyoruz
@app.post("/save")
async def save_image(drug_name: str = Form(...)):
    if not os.path.exists(LATEST_PATH):
        return JSONResponse({"status": "error", "detail": "No latest image"}, status_code=404)
    # temizle ilac ismini güvenli hale getir
    safe_name = "".join(c for c in drug_name if c.isalnum() or c in (" ", "_", "-")).strip().replace(" ", "_")
    folder = os.path.join(DATASET_DIR, safe_name)
    os.makedirs(folder, exist_ok=True)
    # benzersiz isim: klasördeki count+1
    count = len([f for f in os.listdir(folder) if f.lower().endswith((".jpg",".jpeg",".png"))])
    dest = os.path.join(folder, f"{count+1}.jpg")
    shutil.copy(LATEST_PATH, dest)
    return JSONResponse({"status":"ok", "saved_to": dest})

# En son resmi direkt indir / preview için
@app.get("/latest.jpg")
async def get_latest():
    if os.path.exists(LATEST_PATH):
        return FileResponse(LATEST_PATH, media_type="image/jpeg")
    return JSONResponse({"status":"error","detail":"No image"}, status_code=404)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000)
