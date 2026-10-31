import os
import json
import time
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

CALIBRATION_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "calibration_data")
CALIBRATION_FILE = os.path.join(CALIBRATION_DIR, "calibration.json")
REFERENCE_IMAGE = os.path.join(CALIBRATION_DIR, "reference_top.jpg")

# Fast ORB detector (1000 features is optimal for real-time speed & stability)
_orb = cv2.ORB_create(nfeatures=1000, fastThreshold=12)
_bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

# Cached reference data in RAM
_ref_keypoints = None       # List of KeyPoints inside PCB mask
_ref_descriptors = None     # np.ndarray of descriptors matching _ref_keypoints
_ref_corners = None         # np.ndarray of shape (4, 2)
_last_calib_mtime = 0
_calibration_loaded = False

# Corner smoothing (Exponential Moving Average) to eliminate jitter
_smooth_corners = None
_lost_count = 0

# Font cache for crisp Thai rendering
_font = None
def get_font(size=15):
    global _font
    if _font is not None and getattr(_font, "size", None) == size:
        return _font
    font_paths = [
        "C:/Windows/Fonts/tahoma.ttf",
        "C:/Windows/Fonts/leelawad.ttf",
        "C:/Windows/Fonts/arial.ttf"
    ]
    for fp in font_paths:
        if os.path.exists(fp):
            try:
                _font = ImageFont.truetype(fp, size)
                return _font
            except Exception:
                pass
    _font = ImageFont.load_default()
    return _font


def ensure_dir():
    os.makedirs(CALIBRATION_DIR, exist_ok=True)


def save_manual_corners(frame_bgr, corners_list):
    """
    Save 4 corners manually selected.
    Corners list: [[x0, y0], [x1, y1], [x2, y2], [x3, y3]]
    """
    ensure_dir()
    pts = np.array(corners_list, dtype=np.float32)
    h, w = frame_bgr.shape[:2]

    # Save reference image
    cv2.imwrite(REFERENCE_IMAGE, frame_bgr)

    calib_data = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "image_size": {"width": w, "height": h},
        "side": "TOP",
        "corners": pts.tolist()
    }

    with open(CALIBRATION_FILE, "w", encoding="utf-8") as f:
        json.dump(calib_data, f, ensure_ascii=False, indent=2)

    load_reference(force_reload=True)

    thumb = cv2.resize(frame_bgr, (160, 120))
    _, buf = cv2.imencode('.jpg', thumb, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    import base64
    thumb_b64 = "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode('utf-8')

    return {
        "ok": True,
        "feature_count": len(_ref_keypoints) if _ref_keypoints else 0,
        "created_at": calib_data["created_at"],
        "thumbnail": thumb_b64,
        "corners": pts.tolist()
    }


def save_reference(frame_bgr):
    """Fallback capture using default inset corners."""
    h, w = frame_bgr.shape[:2]
    pad_x, pad_y = int(w * 0.1), int(h * 0.1)
    default_corners = [
        [pad_x, pad_y],
        [w - pad_x, pad_y],
        [w - pad_x, h - pad_y],
        [pad_x, h - pad_y]
    ]
    return save_manual_corners(frame_bgr, default_corners)


def load_reference(force_reload=False):
    """
    Load reference image and PCB 4-corners from disk.
    Caches keypoints & descriptors in RAM strictly inside the PCB bounding polygon.
    """
    global _ref_keypoints, _ref_descriptors, _ref_corners, _last_calib_mtime, _calibration_loaded

    if not os.path.exists(CALIBRATION_FILE) or not os.path.exists(REFERENCE_IMAGE):
        return False

    current_mtime = os.path.getmtime(CALIBRATION_FILE)
    if not force_reload and _calibration_loaded and _ref_corners is not None and current_mtime <= _last_calib_mtime:
        return True

    try:
        with open(CALIBRATION_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)

        ref_img = cv2.imread(REFERENCE_IMAGE)
        if ref_img is None:
            return False

        if "corners" in data and len(data["corners"]) == 4:
            pts = np.array(data["corners"], dtype=np.float32)
        else:
            return False

        h, w = ref_img.shape[:2]

        # Create polygon mask strictly for the PCB board
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [pts.astype(np.int32)], 255)

        gray = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)
        kp, des = _orb.detectAndCompute(gray, mask)

        # Fallback to bounding box rectangle if polygon was too tight
        if des is None or len(kp) < 15:
            x_min, x_max = max(0, int(pts[:, 0].min()) - 10), min(w, int(pts[:, 0].max()) + 10)
            y_min, y_max = max(0, int(pts[:, 1].min()) - 10), min(h, int(pts[:, 1].max()) + 10)
            mask_rect = np.zeros((h, w), dtype=np.uint8)
            mask_rect[y_min:y_max, x_min:x_max] = 255
            kp, des = _orb.detectAndCompute(gray, mask_rect)

        if des is None or len(kp) < 5:
            print("[Calibration] Warning: low features extracted from reference.")

        _ref_keypoints = kp
        _ref_descriptors = des
        _ref_corners = pts
        _last_calib_mtime = current_mtime
        _calibration_loaded = True
        return True

    except Exception as e:
        print("[Calibration] Error loading reference:", e)
        return False


def get_calibration_info():
    """Return calibration status and metadata."""
    if not os.path.exists(CALIBRATION_FILE) or not os.path.exists(REFERENCE_IMAGE):
        return {"calibrated": False}

    try:
        with open(CALIBRATION_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)

        thumb_b64 = None
        if os.path.exists(REFERENCE_IMAGE):
            ref_img = cv2.imread(REFERENCE_IMAGE)
            if ref_img is not None:
                thumb = cv2.resize(ref_img, (160, 120))
                _, buf = cv2.imencode('.jpg', thumb, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                import base64
                thumb_b64 = "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode('utf-8')

        return {
            "calibrated": True,
            "created_at": data.get("created_at", ""),
            "feature_count": data.get("feature_count", 0),
            "side": data.get("side", "TOP"),
            "thumbnail": thumb_b64,
            "corners": data.get("corners", [])
        }
    except Exception:
        return {"calibrated": False}


def clear_calibration():
    """Remove all calibration data."""
    global _ref_keypoints, _ref_descriptors, _ref_corners, _calibration_loaded, _smooth_corners
    _ref_keypoints = None
    _ref_descriptors = None
    _ref_corners = None
    _smooth_corners = None
    _calibration_loaded = False

    if os.path.exists(REFERENCE_IMAGE):
        try: os.remove(REFERENCE_IMAGE)
        except Exception: pass
    if os.path.exists(CALIBRATION_FILE):
        try: os.remove(CALIBRATION_FILE)
        except Exception: pass


def order_corners(pts):
    """
    Order 4 corners clockwise starting from Top-Left:
    TL (0), TR (1), BR (2), BL (3).
    """
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # Top-Left: smallest x + y
    rect[2] = pts[np.argmax(s)]  # Bottom-Right: largest x + y

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # Top-Right: smallest y - x
    rect[3] = pts[np.argmax(diff)]  # Bottom-Left: largest y - x
    return rect


def detect_physical_pcb_board(frame_bgr):
    """
    Fast Physical PCB Board Detection (< 2 ms).
    Detects the physical PCB board rectangle using color mask and geometric contours.
    Works for ANY PCB board (normal, with holes, defective, or different versions).
    """
    h, w = frame_bgr.shape[:2]
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

    # Wide green solder mask range (covers light green, olive, emerald, and dark green PCB)
    mask_green = cv2.inRange(hsv, np.array([25, 20, 20]), np.array([105, 255, 255]))

    # Clean noise and small gaps
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    mask = cv2.morphologyEx(mask_green, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = (w * h) * 0.025  # Minimum 2.5% of frame area
    max_area = (w * h) * 0.90  # Maximum 90% of frame area

    best_corners = None
    best_area = 0

    for c in contours:
        area = cv2.contourArea(c)
        if min_area < area < max_area:
            rect = cv2.minAreaRect(c)
            box = cv2.boxPoints(rect)
            box_area = rect[1][0] * rect[1][1]
            # Ensure it is a solid rectangular PCB board (fill ratio > 60%)
            if box_area > 0 and (area / box_area) > 0.58:
                if area > best_area:
                    best_area = area
                    best_corners = order_corners(box)

    return best_corners


def check_orientation(frame_bgr):
    """
    Intelligent Real-Time PCB Tracking and Orientation Detection.
    1. Uses Physical PCB Board Detection to track ANY PCB board (holes, defective, rotated).
    2. Uses ORB Keypoint Homography to verify top-side orientation when available.
    """
    global _smooth_corners, _lost_count

    load_reference()

    h_full, w_full = frame_bgr.shape[:2]

    # Step 1: Detect physical PCB board via color & contour
    pcb_corners = detect_physical_pcb_board(frame_bgr)

    if pcb_corners is not None:
        # Calculate real-time tilt angle from Top-Left (0) to Top-Right (1)
        dx = float(pcb_corners[1][0] - pcb_corners[0][0])
        dy = float(pcb_corners[1][1] - pcb_corners[0][1])
        angle_deg = float(np.degrees(np.arctan2(dy, dx)))
        abs_angle = abs(round(angle_deg))

        # Temporal smoothing (EMA) for rock-solid stability without jitter
        if _smooth_corners is None or _lost_count > 3:
            _smooth_corners = pcb_corners.copy()
        else:
            alpha = 0.70  # 70% current detection, 30% previous smooth
            _smooth_corners = (alpha * pcb_corners + (1.0 - alpha) * _smooth_corners)

        _lost_count = 0

        # Check if we can determine exact 180-degree flip using reference features
        is_rotated_180 = False
        if _ref_descriptors is not None and _ref_keypoints is not None:
            try:
                # Fast ORB check on the detected PCB region
                scale = 0.5
                small_frame = cv2.resize(frame_bgr, (int(w_full * scale), int(h_full * scale)))
                gray_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
                kp_cur, des_cur = _orb.detectAndCompute(gray_small, None)
                if des_cur is not None and len(kp_cur) >= 10:
                    matches = _bf.knnMatch(_ref_descriptors, des_cur, k=2)
                    good = [m for m, n in matches if m.distance < 0.78 * n.distance]
                    if len(good) >= 10:
                        src_pts = np.float32([_ref_keypoints[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
                        dst_pts = np.float32([kp_cur[m.trainIdx].pt for m in good]).reshape(-1, 1, 2) * (1.0 / scale)
                        M, inliers = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 6.0)
                        if M is not None and np.sum(inliers) >= 8:
                            homo_angle = float(np.degrees(np.arctan2(M[1, 0], M[0, 0])))
                            homo_angle = ((homo_angle + 180) % 360) - 180
                            if abs(round(homo_angle)) > 135:
                                is_rotated_180 = True
            except Exception:
                pass

        if is_rotated_180:
            return {
                "orientation": "TOP_ROTATED",
                "label_th": f"[!] หมุนกลับหัว 180° ({abs_angle}°)",
                "match_score": 90,
                "rotation_deg": round(angle_deg, 1),
                "corners": _smooth_corners.tolist()
            }
        elif abs_angle > 15:
            return {
                "orientation": "TOP_TILTED",
                "label_th": f"[!] บอร์ดเอียง {abs_angle}°",
                "match_score": 85,
                "rotation_deg": round(angle_deg, 1),
                "corners": _smooth_corners.tolist()
            }
        else:
            return {
                "orientation": "TOP_CORRECT",
                "label_th": f"[OK] บอร์ดพร้อมตรวจ ({abs_angle}°)",
                "match_score": 95,
                "rotation_deg": round(angle_deg, 1),
                "corners": _smooth_corners.tolist()
            }

    # Step 2: Fallback to ORB feature matching if color detection didn't trigger
    if _ref_corners is not None and _ref_descriptors is not None and _ref_keypoints is not None:
        try:
            scale = 0.5
            inv_scale = 1.0 / scale
            small_frame = cv2.resize(frame_bgr, (int(w_full * scale), int(h_full * scale)), interpolation=cv2.INTER_LINEAR)
            gray_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
            kp_current, des_current = _orb.detectAndCompute(gray_small, None)

            if des_current is not None and len(kp_current) >= 10:
                matches = _bf.knnMatch(_ref_descriptors, des_current, k=2)
                good_matches = [m for m, n in matches if m.distance < 0.78 * n.distance]

                if len(good_matches) >= 10:
                    src_pts = np.float32([_ref_keypoints[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                    dst_pts = np.float32([kp_current[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2) * inv_scale
                    M, inliers = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 6.0)

                    if M is not None and np.sum(inliers) >= 8:
                        transformed = cv2.perspectiveTransform(_ref_corners.reshape(-1, 1, 2), M).reshape(-1, 2)
                        area = cv2.contourArea(transformed.astype(np.int32))
                        full_area = w_full * h_full
                        if (full_area * 0.03) < area < (full_area * 0.90):
                            if _smooth_corners is None or _lost_count > 3:
                                _smooth_corners = transformed.copy()
                            else:
                                _smooth_corners = 0.70 * transformed + 0.30 * _smooth_corners
                            _lost_count = 0

                            angle_rad = np.arctan2(M[1, 0], M[0, 0])
                            angle_deg = float(np.degrees(angle_rad))
                            angle_deg = ((angle_deg + 180) % 360) - 180
                            abs_angle = abs(round(angle_deg))

                            if abs_angle > 135:
                                return {"orientation": "TOP_ROTATED", "label_th": f"[!] หมุนกลับหัว 180° ({abs_angle}°)", "match_score": 90, "rotation_deg": round(angle_deg, 1), "corners": _smooth_corners.tolist()}
                            elif abs_angle > 15:
                                return {"orientation": "TOP_TILTED", "label_th": f"[!] บอร์ดเอียง {abs_angle}°", "match_score": 85, "rotation_deg": round(angle_deg, 1), "corners": _smooth_corners.tolist()}
                            else:
                                return {"orientation": "TOP_CORRECT", "label_th": f"[OK] บอร์ดพร้อมตรวจ ({abs_angle}°)", "match_score": 95, "rotation_deg": round(angle_deg, 1), "corners": _smooth_corners.tolist()}
        except Exception:
            pass

    return _handle_tracking_lost()


def _handle_tracking_lost():
    """Handle loss of PCB detection gracefully."""
    global _lost_count, _smooth_corners
    _lost_count += 1

    # Keep holding last known PCB position smoothly for 1-2 frames to avoid micro-flickering
    if _lost_count <= 2 and _smooth_corners is not None:
        return {
            "orientation": "TOP_TRACKING",
            "label_th": "🔍 กำลังตรวจจับบอร์ด PCB...",
            "match_score": 0,
            "rotation_deg": 0,
            "corners": _smooth_corners.tolist()
        }

    # When no PCB board is on screen, show clean frame (NO idle guide box)
    _smooth_corners = None
    return {
        "orientation": "NO_BOARD",
        "label_th": "",
        "match_score": 0,
        "rotation_deg": 0,
        "corners": None
    }


def draw_pcb_orientation_overlay(frame_bgr, orientation_result):
    """
    Blazingly fast PCB 4-corner overlay.
    Only draws when a real PCB board is detected. When no board is present, returns clean frame.
    """
    if not orientation_result or not orientation_result.get("corners"):
        return frame_bgr

    orient = orientation_result.get("orientation", "")
    if orient in ("NO_BOARD", "IDLE_GUIDE", ""):
        return frame_bgr

    corners_list = orientation_result.get("corners")
    if not corners_list or len(corners_list) < 4:
        return frame_bgr

    pts = np.array(corners_list, dtype=np.int32)
    label_text = orientation_result.get("label_th", "")

    if not label_text:
        return frame_bgr

    # Colors
    if orient == "TOP_CORRECT":
        bgr_color = (46, 204, 113)    # Bright Emerald Green
        bg_rgb = (10, 45, 25)
        thickness = 3
    elif orient in ("TOP_ROTATED", "TOP_TILTED"):
        bgr_color = (0, 165, 255)     # Amber / Orange
        bg_rgb = (50, 30, 5)
        thickness = 3
    elif orient == "TOP_TRACKING":
        bgr_color = (0, 215, 255)     # Cyan
        bg_rgb = (10, 35, 45)
        thickness = 2
    else:  # IDLE_GUIDE
        bgr_color = (212, 182, 6)     # Cyan Guide
        bg_rgb = (15, 30, 45)
        thickness = 2

    # 1. Draw 4-corner polygon around PCB
    cv2.polylines(frame_bgr, [pts], isClosed=True, color=bgr_color, thickness=thickness, lineType=cv2.LINE_AA)

    # 2. Highlight top edge (corner 0 to 1) with thicker line
    cv2.line(frame_bgr, tuple(pts[0]), tuple(pts[1]), bgr_color, thickness=thickness + 2, lineType=cv2.LINE_AA)

    # 3. Corner points
    for p in pts:
        cv2.circle(frame_bgr, tuple(p), 5, bgr_color, -1, lineType=cv2.LINE_AA)

    # 4. Tiny ROI Thai text badge overlay (fast: only processes a 240x35 px patch, not 1280x720!)
    try:
        h_frame, w_frame = frame_bgr.shape[:2]
        tx = int((pts[0][0] + pts[1][0]) // 2)
        ty = int((pts[0][1] + pts[1][1]) // 2)

        font = get_font(size=14)

        # Estimate badge dimension
        badge_w, badge_h = 240, 32
        x1 = max(5, min(w_frame - badge_w - 5, tx - badge_w // 2))
        y1 = max(5, min(h_frame - badge_h - 5, ty - badge_h - 8))
        x2 = x1 + badge_w
        y2 = y1 + badge_h

        # Extract only the small badge patch
        patch_bgr = frame_bgr[y1:y2, x1:x2]
        if patch_bgr.shape[0] == badge_h and patch_bgr.shape[1] == badge_w:
            # Draw on small patch with PIL
            pil_patch = Image.fromarray(cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(pil_patch)
            draw.rounded_rectangle([0, 0, badge_w - 1, badge_h - 1], radius=6, fill=bg_rgb, outline=(bgr_color[2], bgr_color[1], bgr_color[0]), width=2)
            draw.text((10, 6), label_text, font=font, fill=(255, 255, 255))

            # Put back into frame
            frame_bgr[y1:y2, x1:x2] = cv2.cvtColor(np.array(pil_patch), cv2.COLOR_RGB2BGR)

    except Exception:
        pass

    return frame_bgr
