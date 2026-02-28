# YOLOv8 Inference on Kubernetes (Local Dev Setup)

Been wanting to get a real ML inference service running on Kubernetes for a while — 
not just a toy app, but something with proper resource limits, health checks, and 
network policies. This is that project.

Runs YOLOv8 nano object detection as a REST API, deployed on Docker Desktop's local 
K8s cluster. No GPU needed — pure CPU inference using ONNX Runtime to keep the image 
lean (~300MB vs the usual 4GB PyTorch bloat).

---

## What it does

POST an image → get back detected objects with confidence scores and bounding boxes.

```json
{
  "filename": "bus.jpg",
  "image_size": { "width": 1280, "height": 720 },
  "detections": [
    { "class": "person", "confidence": 0.892, "bbox": {...} },
    { "class": "bus",    "confidence": 0.871, "bbox": {...} }
  ],
  "count": 3
}
```

---

## Stack

- **YOLOv8n** — nano model, fast enough for CPU inference
- **ONNX Runtime** — no PyTorch/CUDA required in the container
- **FastAPI** — lightweight inference API with health endpoints
- **Kubernetes** — 2-replica deployment with resource limits + network policies
- **Docker Desktop** — local K8s cluster on MacBook Pro M3 Pro

---

## Project Structure

```
yolo-k8s/
├── app/
│   ├── main.py               # FastAPI inference server
│   ├── requirements.txt
│   └── yolov8n.onnx          # exported model (~12MB)
├── Dockerfile
├── k8s/
│   ├── deployment.yaml       # 2 replicas, CPU/memory limits
│   ├── service.yaml          # NodePort on :30800
│   └── networkpolicy.yaml    # restrict ingress/egress
└── test/
    └── test_inference.py
```

---

## Running it locally

**Prerequisites:** Docker Desktop with Kubernetes enabled.

```bash
# 1. Export the ONNX model (one time)
pip install ultralytics
yolo export model=yolov8n.pt format=onnx imgsz=640
mv yolov8n.onnx app/

# 2. Build the image
docker buildx build \
  --platform linux/amd64 \
  -t yolo-inference:v1 \
  --load .

# 3. Deploy
kubectl config use-context docker-desktop
kubectl apply -f k8s/

# 4. Wait for pods to be ready
kubectl get pods -w
```

App will be live at `http://localhost:30800`

---

## Testing

```bash
# Health check
curl http://localhost:30800/healthz

# Run inference
curl -X POST "http://localhost:30800/predict?confidence=0.4" \
  -F "file=@/path/to/image.jpg" | python3 -m json.tool

# Or run the test script
python test/test_inference.py
```

---

## A few things I learned along the way

- Building for `linux/amd64` on Apple Silicon without specifying CPU-only torch
  pulls in gigabytes of Nvidia CUDA packages you'll never use — explicitly install
  from `https://download.pytorch.org/whl/cpu` or just switch to ONNX Runtime
- `python:3.11-slim` doesn't ship with `libGL.so.1` — OpenCV needs it, add `libgl1`
  to your apt installs
- Docker Desktop K8s can access locally built images directly with
  `imagePullPolicy: Never` — no need to push to a registry for local dev
- Default-deny NetworkPolicy breaks DNS — always add an egress rule for port 53

---
