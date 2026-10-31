import os
import glob
import time
import shutil
import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from backend.camera_service import camera_service
from backend.calibration_service import (
    save_reference, save_manual_corners, load_reference, get_calibration_info,
    clear_calibration, check_orientation
)
from pydantic import BaseModel
from typing import List

from backend.model_manager import (
    WORKSPACE_DIR,
    get_available_models,
    get_best_model_path,
    run_detection,
    get_sample_images
)

app = FastAPI(
    title="PCB Defect Detection API",
    description="Vision AI Dashboard for Automated PCB Defect Inspection",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

FRONTEND_DIR = os.path.join(WORKSPACE_DIR, "frontend")

TRAINING_STATE = {
    "is_training": False,
    "progress": 0,
    "epoch": 0,
    "total_epochs": 0,
    "current_map": 0.0,
    "message": "Idle"
}

@app.get("/api/status")
def get_system_status():
    import torch
    cuda_avail = torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0) if cuda_avail else "CPU Only"
    best_m = get_best_model_path()

    return {
        "status": "online",
        "cuda_available": cuda_avail,
        "device": device_name,
        "active_model": os.path.basename(best_m),
        "active_model_path": best_m,
        "training_active": TRAINING_STATE["is_training"]
    }

@app.get("/api/models")
def list_models():
    return {
        "models": get_available_models(),
        "active_model": get_best_model_path()
    }

@app.get("/api/samples")
def list_samples():
    return {
        "samples": get_sample_images(limit=24)
    }

@app.get("/api/sample-image/{filename}")
def get_sample_image(filename: str):
    if ".." in filename or "/" in filename or "\\" in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")

    for sub in ["valid", "train"]:
        img_p = os.path.join(WORKSPACE_DIR, "Datasets", sub, "images", filename)
        if os.path.exists(img_p):
            return FileResponse(img_p, media_type="image/jpeg")

    raise HTTPException(status_code=404, detail="Sample image not found")

@app.get("/api/dataset-stats")
def get_dataset_statistics():
    train_dir = os.path.join(WORKSPACE_DIR, "Datasets", "train", "images")
    valid_dir = os.path.join(WORKSPACE_DIR, "Datasets", "valid", "images")

    n_train = len(os.listdir(train_dir)) if os.path.exists(train_dir) else 0
    n_valid = len(os.listdir(valid_dir)) if os.path.exists(valid_dir) else 0

    return {
        "train_count": n_train,
        "valid_count": n_valid,
        "total_images": n_train + n_valid,
        "classes": ["missing_hole"],
        "class_count": 1,
        "base_model": "YOLO11m (Medium Architecture, ~20M parameters)"
    }

@app.post("/api/detect")
async def detect_defects(
    file: UploadFile = None,
    sample_filename: str = Form(None),
    image_base64: str = Form(None),
    conf_thresh: float = Form(0.50),
    iou_thresh: float = Form(0.45),
    model_path: str = Form(None)
):
    try:
        img_bytes = None
        source_name = "uploaded_pcb.jpg"

        if sample_filename:
            source_name = sample_filename
            found = False
            for sub in ["valid", "train"]:
                p = os.path.join(WORKSPACE_DIR, "Datasets", sub, "images", sample_filename)
                if os.path.exists(p):
                    with open(p, "rb") as fp:
                        img_bytes = fp.read()
                    found = True
                    break
            if not found:
                raise HTTPException(status_code=404, detail="Sample image not found")
        elif image_base64:
            source_name = "live_camera_frame.jpg"
            if "," in image_base64:
                image_base64 = image_base64.split(",", 1)[1]
            import base64
            img_bytes = base64.b64decode(image_base64)
        elif file is not None:
            source_name = file.filename
            img_bytes = await file.read()
        else:
            raise HTTPException(status_code=400, detail="Either file, sample_filename, or image_base64 must be provided")

        nparr = np.frombuffer(img_bytes, np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img_bgr is None:
            raise HTTPException(status_code=400, detail="Could not decode image file")

        result = run_detection(
            img_bgr=img_bgr,
            conf_thresh=conf_thresh,
            iou_thresh=iou_thresh,
            model_path=model_path
        )

        result["filename"] = source_name
        return JSONResponse(content=result)

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

BACKEND_CAMERAS_CACHE = None

@app.get("/api/camera/devices")
def list_backend_cameras():
    return {
        "devices": [
            {
                "id": "directshow_0",
                "index": 0,
                "name": "USB Camera (DirectShow / USB Microscope)"
            }
        ]
    }

@app.post("/api/camera/start")
def start_camera_stream(
    camera_index: int = Form(0),
    conf_thresh: float = Form(0.50),
    iou_thresh: float = Form(0.45),
    model_path: str = Form(None)
):
    camera_service.update_settings(
        conf_thresh=conf_thresh,
        iou_thresh=iou_thresh,
        model_path=model_path,
        auto_detect=True
    )
    ok = camera_service.start(camera_index)
    if not ok:
        raise HTTPException(status_code=500, detail="Failed to initialize USB camera via DirectShow")
    return {"status": "ok", "active": True}

@app.post("/api/camera/stop")
def stop_camera_stream():
    camera_service.stop()
    return {"status": "ok", "active": False}

@app.get("/api/camera/stream")
def get_camera_stream():
    if not camera_service.is_active():
        ok = camera_service.start(0)
        if not ok:
            raise HTTPException(status_code=500, detail="Cannot start camera stream")

    return StreamingResponse(
        camera_service.generate_stream(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

@app.get("/api/camera/telemetry")
def get_camera_telemetry():
    return camera_service.get_telemetry()

@app.post("/api/camera/settings")
def update_camera_settings(
    conf_thresh: float = Form(None),
    iou_thresh: float = Form(None),
    model_path: str = Form(None),
    auto_detect: bool = Form(None),
    brightness: float = Form(None)
):
    camera_service.update_settings(
        conf_thresh=conf_thresh,
        iou_thresh=iou_thresh,
        model_path=model_path,
        auto_detect=auto_detect,
        brightness=brightness
    )
    return {"status": "updated"}

@app.post("/api/camera/snapshot")
def capture_camera_snapshot():
    try:
        result = camera_service.capture_snapshot()
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# =============================================================
# Calibration Endpoints
# =============================================================

class CornersPayload(BaseModel):
    corners: List[List[float]]

@app.post("/api/calibration/save_corners")
def save_calib_corners(payload: CornersPayload):
    """Save 4 corners marked interactively by user on camera view."""
    if not camera_service.is_active():
        raise HTTPException(status_code=400, detail="กรุณาเปิดกล้องก่อนมาร์กจุด")

    raw = None
    with camera_service.lock:
        if camera_service.latest_raw_frame is not None:
            raw = camera_service.latest_raw_frame.copy()

    if raw is None:
        raise HTTPException(status_code=500, detail="ไม่พบสัญญาณภาพจากกล้อง")

    result = save_manual_corners(raw, payload.corners)
    if not result.get("ok"):
        raise HTTPException(status_code=400, detail=result.get("error", "ไม่สามารถบันทึก 4 มุมได้"))

    return result

@app.get("/api/calibration/status")
def get_calib_status():
    return get_calibration_info()

@app.post("/api/calibration/capture")
def capture_calibration_reference():
    """Capture current camera frame as reference TOP side image."""
    if not camera_service.is_active():
        raise HTTPException(status_code=400, detail="Camera is not active")

    raw = None
    with camera_service.lock:
        if camera_service.latest_raw_frame is not None:
            raw = camera_service.latest_raw_frame.copy()

    if raw is None:
        raise HTTPException(status_code=500, detail="No camera frame available")

    result = save_reference(raw)
    if not result.get("ok"):
        raise HTTPException(status_code=400, detail=result.get("error", "Calibration failed"))

    return result

@app.post("/api/calibration/clear")
def clear_calib():
    clear_calibration()
    return {"status": "cleared"}

@app.get("/api/calibration/check")
def check_calib_orientation():
    """Check current camera frame orientation against reference."""
    if not camera_service.is_active():
        return {"orientation": "CAMERA_OFF", "label_th": "กล้องปิดอยู่", "match_score": 0}

    raw = None
    with camera_service.lock:
        if camera_service.latest_raw_frame is not None:
            raw = camera_service.latest_raw_frame.copy()

    if raw is None:
        return {"orientation": "NO_FRAME", "label_th": "ไม่มีภาพจากกล้อง", "match_score": 0}

    return check_orientation(raw)

if os.path.exists(FRONTEND_DIR):
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.server:app", host="127.0.0.1", port=8000, reload=True)
