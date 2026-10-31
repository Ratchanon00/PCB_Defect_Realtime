import os
import glob
import time
import base64
import io
import torch
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

WORKSPACE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_CACHE = {}

def get_available_models():
    models = []

    primary_cand = os.path.join(WORKSPACE_DIR, "yolo11m.pt")
    if os.path.exists(primary_cand):
        models.append({
            "id": primary_cand,
            "name": "YOLO11m (PCB Defect Detector)",
            "path": primary_cand,
            "is_trained": True,
            "size_mb": round(os.path.getsize(primary_cand) / (1024 * 1024), 1)
        })

    return models

def get_best_model_path():
    models = get_available_models()
    if models:
        return models[0]["path"]
    return os.path.join(WORKSPACE_DIR, "yolo11m.pt")

def load_yolo_model(model_path=None):
    if not model_path:
        model_path = get_best_model_path()

    if model_path in MODELS_CACHE:
        return MODELS_CACHE[model_path]

    if not os.path.exists(model_path):
        fallback = os.path.join(WORKSPACE_DIR, "yolo11m.pt")
        if os.path.exists(fallback):
            model_path = fallback
        else:
            raise FileNotFoundError(f"Model path not found: {model_path}")

    model = YOLO(model_path)
    MODELS_CACHE[model_path] = model
    return model

def draw_industrial_annotations(orig_img_bgr, boxes, class_names):
    """
    Draw sleek industrial style annotations with corner brackets, glowing boxes,
    and dark HUD badge chips.
    """
    annotated = orig_img_bgr.copy()
    h, w = annotated.shape[:2]

    colors = [
        (46, 75, 239),
        (34, 197, 94),
        (234, 179, 8),
        (236, 72, 153),
        (59, 130, 246),
    ]

    for idx, b in enumerate(boxes):
        x1, y1, x2, y2 = int(b["x1"]), int(b["y1"]), int(b["x2"]), int(b["y2"])
        conf = b["confidence"]
        c_name = b["class_name"]
        color = colors[b["class_id"] % len(colors)]

        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w - 1, x2), min(h - 1, y2)

        bw = x2 - x1
        bh = y2 - y1

        overlay = annotated.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
        cv2.addWeighted(overlay, 0.15, annotated, 0.85, 0, annotated)

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)

        corner_len = max(6, min(16, min(bw, bh) // 4))
        cv2.line(annotated, (x1, y1), (x1 + corner_len, y1), color, 3, cv2.LINE_AA)
        cv2.line(annotated, (x1, y1), (x1, y1 + corner_len), color, 3, cv2.LINE_AA)
        cv2.line(annotated, (x2, y1), (x2 - corner_len, y1), color, 3, cv2.LINE_AA)
        cv2.line(annotated, (x2, y1), (x2, y1 + corner_len), color, 3, cv2.LINE_AA)
        cv2.line(annotated, (x1, y2), (x1 + corner_len, y2), color, 3, cv2.LINE_AA)
        cv2.line(annotated, (x1, y2), (x1, y2 - corner_len), color, 3, cv2.LINE_AA)
        cv2.line(annotated, (x2, y2), (x2 - corner_len, y2), color, 3, cv2.LINE_AA)
        cv2.line(annotated, (x2, y2), (x2, y2 - corner_len), color, 3, cv2.LINE_AA)

        label_text = f"#{idx+1} {c_name.upper()} {conf:.1%}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.45
        thickness = 1
        (txt_w, txt_h), baseline = cv2.getTextSize(label_text, font, font_scale, thickness)

        tag_x1 = x1
        tag_y1 = max(0, y1 - txt_h - 8)
        tag_x2 = tag_x1 + txt_w + 10
        tag_y2 = tag_y1 + txt_h + 8

        cv2.rectangle(annotated, (tag_x1, tag_y1), (tag_x2, tag_y2), (15, 23, 42), -1)
        cv2.rectangle(annotated, (tag_x1, tag_y1), (tag_x2, tag_y2), color, 1, cv2.LINE_AA)

        dot_center = (tag_x1 + 6, tag_y1 + (tag_y2 - tag_y1) // 2)
        cv2.circle(annotated, dot_center, 3, color, -1, cv2.LINE_AA)

        cv2.putText(
            annotated,
            label_text,
            (tag_x1 + 13, tag_y2 - 5),
            font,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA
        )

    return annotated

def crop_defect_patches(orig_img_bgr, boxes, max_patches=12):
    """Crop magnified patches around each defect for gallery view."""
    h, w = orig_img_bgr.shape[:2]
    patches = []

    for idx, b in enumerate(boxes[:max_patches]):
        x1, y1, x2, y2 = int(b["x1"]), int(b["y1"]), int(b["x2"]), int(b["y2"])
        bw = x2 - x1
        bh = y2 - y1

        pad_x = max(15, int(bw * 0.4))
        pad_y = max(15, int(bh * 0.4))

        crop_x1 = max(0, x1 - pad_x)
        crop_y1 = max(0, y1 - pad_y)
        crop_x2 = min(w, x2 + pad_x)
        crop_y2 = min(h, y2 + pad_y)

        crop = orig_img_bgr[crop_y1:crop_y2, crop_x1:crop_x2].copy()

        rel_x1 = x1 - crop_x1
        rel_y1 = y1 - crop_y1
        rel_x2 = x2 - crop_x1
        rel_y2 = y2 - crop_y1
        cv2.rectangle(crop, (rel_x1, rel_y1), (rel_x2, rel_y2), (46, 75, 239), 2, cv2.LINE_AA)

        target_size = 140
        ch, cw = crop.shape[:2]
        if ch > 0 and cw > 0:
            scale = min(target_size / cw, target_size / ch)
            new_w, new_h = max(1, int(cw * scale)), max(1, int(ch * scale))
            resized = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

            _, buffer = cv2.imencode('.jpg', resized, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
            patch_b64 = base64.b64encode(buffer).decode('utf-8')

            patches.append({
                "id": idx + 1,
                "class_name": b["class_name"],
                "confidence": round(b["confidence"] * 100, 1),
                "box": [x1, y1, x2, y2],
                "area_px": bw * bh,
                "image_b64": f"data:image/jpeg;base64,{patch_b64}"
            })

    return patches

def run_detection(img_bgr, conf_thresh=0.50, iou_thresh=0.45, model_path=None, include_images=True):
    """
    Run YOLO defect detection on an image.
    Returns full analysis dictionary with timing, boxes, images, and crops.
    """
    device = 0 if torch.cuda.is_available() else "cpu"
    model = load_yolo_model(model_path)

    h, w = img_bgr.shape[:2]

    t0 = time.perf_counter()
    results = model.predict(
        source=img_bgr,
        conf=conf_thresh,
        iou=iou_thresh,
        device=device,
        verbose=False
    )
    total_time_ms = round((time.perf_counter() - t0) * 1000, 1)

    res = results[0]
    boxes_out = []

    names = model.names

    if res.boxes is not None and len(res.boxes) > 0:
        for idx, box in enumerate(res.boxes):
            xyxy = box.xyxy[0].tolist()
            conf = float(box.conf[0].item())
            cls_id = int(box.cls[0].item())
            cls_name = names.get(cls_id, f"class_{cls_id}")

            bx1, by1, bx2, by2 = xyxy
            bw = bx2 - bx1
            bh = by2 - by1

            boxes_out.append({
                "id": idx + 1,
                "class_id": cls_id,
                "class_name": cls_name,
                "confidence": round(conf, 4),
                "x1": round(bx1, 1),
                "y1": round(by1, 1),
                "x2": round(bx2, 1),
                "y2": round(by2, 1),
                "cx": round(bx1 + bw / 2, 1),
                "cy": round(by1 + bh / 2, 1),
                "width": round(bw, 1),
                "height": round(bh, 1),
                "area": round(bw * bh, 1),
                "norm_x1": round(bx1 / w, 4),
                "norm_y1": round(by1 / h, 4),
                "norm_x2": round(bx2 / w, 4),
                "norm_y2": round(by2 / h, 4)
            })

    boxes_out.sort(key=lambda x: x["confidence"], reverse=True)
    for idx, b in enumerate(boxes_out):
        b["id"] = idx + 1

    ann_b64 = None
    orig_b64 = None
    patches = []

    if include_images:
        annotated_bgr = draw_industrial_annotations(img_bgr, boxes_out, names)
        _, ann_buffer = cv2.imencode('.jpg', annotated_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        ann_b64 = base64.b64encode(ann_buffer).decode('utf-8')

        _, orig_buffer = cv2.imencode('.jpg', img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        orig_b64 = base64.b64encode(orig_buffer).decode('utf-8')

        patches = crop_defect_patches(img_bgr, boxes_out)

    distribution = {}
    for b in boxes_out:
        distribution[b["class_name"]] = distribution.get(b["class_name"], 0) + 1

    num_defects = len(boxes_out)
    status = "DEFECTIVE" if num_defects > 0 else "PASS"

    inference_speed = {
        "total_ms": total_time_ms,
        "preprocess_ms": round(res.speed.get("preprocess", 0.0), 1),
        "inference_ms": round(res.speed.get("inference", 0.0), 1),
        "postprocess_ms": round(res.speed.get("postprocess", 0.0), 1),
        "fps": round(1000.0 / max(1.0, total_time_ms), 1),
        "device": f"CUDA (GPU: {torch.cuda.get_device_name(0)})" if torch.cuda.is_available() else "CPU"
    }

    return {
        "status": status,
        "is_pass": num_defects == 0,
        "defect_count": num_defects,
        "dimensions": {"width": w, "height": h},
        "model_used": os.path.basename(model_path or get_best_model_path()),
        "timing": inference_speed,
        "defects": boxes_out,
        "distribution": distribution,
        "patches": patches,
        "annotated_image": f"data:image/jpeg;base64,{ann_b64}",
        "original_image": f"data:image/jpeg;base64,{orig_b64}"
    }

def get_sample_images(limit=16):
    """Retrieve sample images from dataset for instant testing."""
    sample_dirs = [
        os.path.join(WORKSPACE_DIR, "Datasets", "valid", "images"),
        os.path.join(WORKSPACE_DIR, "Datasets", "train", "images")
    ]
    samples = []
    seen = set()

    for s_dir in sample_dirs:
        if os.path.exists(s_dir):
            for fname in sorted(os.listdir(s_dir)):
                if fname.lower().endswith(('.jpg', '.jpeg', '.png')) and fname not in seen:
                    seen.add(fname)
                    full_p = os.path.join(s_dir, fname)
                    samples.append({
                        "filename": fname,
                        "path": full_p,
                        "label_type": "Defect Sample" if "hole" in fname else "Normal PCB",
                        "size_kb": round(os.path.getsize(full_p) / 1024, 1)
                    })
                    if len(samples) >= limit:
                        break
        if len(samples) >= limit:
            break

    return samples
