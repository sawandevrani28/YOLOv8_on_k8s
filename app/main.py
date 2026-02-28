from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from ultralytics import YOLO
from PIL import Image
import io

app = FastAPI(title="YOLOv8 Inference API")

model = YOLO("yolov8n.pt")

@app.get("/healthz")
def health():
    return {"status": "ok"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # Run inference on CPU explicitly
    results = model(image, device="cpu", verbose=False)
    
    detections = []
    for result in results:
        for box in result.boxes:
            detections.append({
                "class": result.names[int(box.cls)],
                "confidence": round(float(box.conf), 3),
                "bbox": [round(x, 1) for x in box.xyxy[0].tolist()]
            })

    return JSONResponse(content={
        "filename": file.filename,
        "detections": detections,
        "count": len(detections)
    })
