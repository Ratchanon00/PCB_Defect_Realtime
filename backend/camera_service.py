import time
import threading
import base64
import cv2
import numpy as np
from backend.model_manager import (
    run_detection,
    draw_industrial_annotations,
    crop_defect_patches,
    get_best_model_path,
    load_yolo_model
)
from backend.calibration_service import check_orientation, draw_pcb_orientation_overlay

class CameraService:
    def __init__(self):
        self.cap = None
        self.is_running = False
        self.thread = None
        self.lock = threading.Lock()

        self.camera_index = 0
        self.conf_thresh = 0.50
        self.iou_thresh = 0.45
        self.model_path = None
        self.auto_detect = True
        self.brightness = 0.0

        self.latest_raw_frame = None
        self.latest_annotated_jpeg = None
        self.latest_telemetry = {
            "status": "IDLE",
            "is_pass": True,
            "defect_count": 0,
            "defects": [],
            "patches": [],
            "dimensions": {"width": 1280, "height": 720},
            "timing": {"inference_ms": 0, "fps": 0},
            "fps": 0
        }

        self.stream_fps = 0.0
        self.last_inference_time = 0

    def start(self, camera_index=0):
        with self.lock:
            if self.is_running and self.cap and self.cap.isOpened():
                return True

            self.camera_index = camera_index
            self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
            if not self.cap.isOpened():
                # Fallback without CAP_DSHOW
                self.cap = cv2.VideoCapture(self.camera_index)
                if not self.cap.isOpened():
                    self.cap = None
                    return False

            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            self.is_running = True
            self.thread = threading.Thread(target=self._worker_loop, daemon=True)
            self.thread.start()
            return True

    def stop(self):
        with self.lock:
            self.is_running = False
            if self.cap:
                try:
                    self.cap.release()
                except Exception:
                    pass
                self.cap = None
            self.latest_raw_frame = None
            self.latest_annotated_jpeg = None

    def is_active(self):
        return self.is_running and self.cap is not None

    def update_settings(self, conf_thresh=None, iou_thresh=None, model_path=None, auto_detect=None, brightness=None):
        with self.lock:
            if conf_thresh is not None:
                self.conf_thresh = float(conf_thresh)
            if iou_thresh is not None:
                self.iou_thresh = float(iou_thresh)
            if model_path is not None:
                self.model_path = model_path
            if auto_detect is not None:
                self.auto_detect = bool(auto_detect)
            if brightness is not None:
                self.brightness = float(brightness)

    def _worker_loop(self):
        frame_counter = 0
        t_fps_start = time.time()
        last_detect_time = 0

        while self.is_running:
            if not self.cap or not self.cap.isOpened():
                time.sleep(0.05)
                continue

            ret, frame = self.cap.read()
            if not ret or frame is None:
                time.sleep(0.01)
                continue

            # Fast detail sharpening & noise suppression (makes traces and holes razor sharp)
            blurred = cv2.GaussianBlur(frame, (0, 0), 2.0)
            frame = cv2.addWeighted(frame, 1.35, blurred, -0.35, 0)

            # Apply software brightness / anti-glare adjustment if set
            if self.brightness != 0:
                frame = cv2.convertScaleAbs(frame, alpha=1.0, beta=self.brightness)

            frame_counter += 1
            now = time.time()
            if now - t_fps_start >= 1.0:
                self.stream_fps = round(frame_counter / (now - t_fps_start), 1)
                frame_counter = 0
                t_fps_start = now

            h, w = frame.shape[:2]

            with self.lock:
                self.latest_raw_frame = frame.copy()

            # Run detection if auto_detect is ON
            if self.auto_detect and (now - last_detect_time >= 0.03):
                last_detect_time = now
                try:
                    # 1. Detect PCB board orientation & corners using Color Image Processing (HSV)
                    orient_res = check_orientation(frame)
                    corners_list = orient_res.get("corners") if orient_res else None
                    boxes = []

                    # 2. If PCB is present on camera, isolate & warp it to upright 512px view
                    if corners_list and len(corners_list) == 4 and orient_res.get("orientation") not in ("NO_BOARD", "IDLE_GUIDE"):
                        pcb_pts = np.array(corners_list, dtype=np.float32)
                        w_a = np.linalg.norm(pcb_pts[2] - pcb_pts[3])
                        w_b = np.linalg.norm(pcb_pts[1] - pcb_pts[0])
                        max_w = max(64, int(max(w_a, w_b)))
                        h_a = np.linalg.norm(pcb_pts[1] - pcb_pts[2])
                        h_b = np.linalg.norm(pcb_pts[0] - pcb_pts[3])
                        max_h = max(64, int(max(h_a, h_b)))

                        target_w = 512
                        target_h = max(128, min(768, int(round(512 * (max_h / max(1, max_w))))))

                        dst = np.array([
                            [0, 0],
                            [target_w - 1, 0],
                            [target_w - 1, target_h - 1],
                            [0, target_h - 1]
                        ], dtype=np.float32)

                        M_warp = cv2.getPerspectiveTransform(pcb_pts, dst)
                        M_inv = cv2.getPerspectiveTransform(dst, pcb_pts)

                        # Warp PCB to flat, upright rectangle (excluding background table)
                        pcb_crop = cv2.warpPerspective(frame, M_warp, (target_w, target_h))

                        # Run YOLO defect detection on the zoomed-in, straightened PCB surface
                        result = run_detection(
                            img_bgr=pcb_crop,
                            conf_thresh=self.conf_thresh,
                            iou_thresh=self.iou_thresh,
                            model_path=self.model_path,
                            include_images=False
                        )

                        # Remap detected defect boxes back to the live camera coordinate space
                        raw_boxes = result.get("defects", [])
                        remapped_boxes = []
                        for b in raw_boxes:
                            bx1, by1, bx2, by2 = float(b["x1"]), float(b["y1"]), float(b["x2"]), float(b["y2"])
                            b_pts = np.array([[[bx1, by1]], [[bx2, by1]], [[bx2, by2]], [[bx1, by2]]], dtype=np.float32)
                            mapped = cv2.perspectiveTransform(b_pts, M_inv).reshape(-1, 2)
                            ox1 = max(0, float(np.min(mapped[:, 0])))
                            oy1 = max(0, float(np.min(mapped[:, 1])))
                            ox2 = min(w, float(np.max(mapped[:, 0])))
                            oy2 = min(h, float(np.max(mapped[:, 1])))

                            b_copy = dict(b)
                            b_copy["x1"], b_copy["y1"] = ox1, oy1
                            b_copy["x2"], b_copy["y2"] = ox2, oy2
                            b_copy["cx"] = (ox1 + ox2) / 2
                            b_copy["cy"] = (oy1 + oy2) / 2
                            b_copy["width"] = ox2 - ox1
                            b_copy["height"] = oy2 - oy1
                            b_copy["norm_x1"] = ox1 / w
                            b_copy["norm_y1"] = oy1 / h
                            b_copy["norm_x2"] = ox2 / w
                            b_copy["norm_y2"] = oy2 / h
                            remapped_boxes.append(b_copy)

                        result["defects"] = remapped_boxes
                        boxes = remapped_boxes
                    else:
                        # Full frame fallback
                        result = run_detection(
                            img_bgr=frame,
                            conf_thresh=self.conf_thresh,
                            iou_thresh=self.iou_thresh,
                            model_path=self.model_path,
                            include_images=False
                        )
                        boxes = result.get("defects", [])

                    # Draw annotations on display frame
                    model = load_yolo_model(self.model_path)
                    annotated_bgr = draw_industrial_annotations(frame, boxes, model.names)

                    # Draw PCB 4-corner bounding box with top-edge status badge
                    try:
                        annotated_bgr = draw_pcb_orientation_overlay(annotated_bgr, orient_res)
                        result["orientation"] = orient_res
                    except Exception as e_or:
                        pass

                    # Encode to JPEG for stream (quality 88 is crisp, high-detail without noise artifacts)
                    _, buf = cv2.imencode('.jpg', annotated_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
                    jpeg_bytes = buf.tobytes()

                    result["fps"] = self.stream_fps
                    result["timing"]["fps"] = self.stream_fps

                    with self.lock:
                        self.latest_annotated_jpeg = jpeg_bytes
                        self.latest_telemetry = result

                except Exception as e:
                    _, buf = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
                    with self.lock:
                        self.latest_annotated_jpeg = buf.tobytes()
            else:
                display_frame = frame.copy()
                try:
                    orient_res = check_orientation(frame)
                    display_frame = draw_pcb_orientation_overlay(display_frame, orient_res)
                except Exception:
                    pass
                _, buf = cv2.imencode('.jpg', display_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
                with self.lock:
                    self.latest_annotated_jpeg = buf.tobytes()

            time.sleep(0.005)

    def generate_stream(self):
        """Yields MJPEG stream packets."""
        while self.is_running:
            frame_bytes = None
            with self.lock:
                if self.latest_annotated_jpeg:
                    frame_bytes = self.latest_annotated_jpeg

            if frame_bytes:
                yield (
                    b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n'
                )
                time.sleep(0.03) # ~30 FPS limit
            else:
                time.sleep(0.02)

    def get_telemetry(self):
        with self.lock:
            return self.latest_telemetry

    def capture_snapshot(self):
        raw = None
        with self.lock:
            if self.latest_raw_frame is not None:
                raw = self.latest_raw_frame.copy()

        if raw is None:
            # Try a direct read
            if self.cap and self.cap.isOpened():
                ret, frame = self.cap.read()
                if ret:
                    raw = frame

        if raw is None:
            raise RuntimeError("No active camera frame available")

        result = run_detection(
            img_bgr=raw,
            conf_thresh=self.conf_thresh,
            iou_thresh=self.iou_thresh,
            model_path=self.model_path
        )
        result["filename"] = f"Camera #{self.camera_index} Snapshot"
        return result

# Global Singleton Instance
camera_service = CameraService()
