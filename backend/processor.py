import os
import uuid
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from skimage import color, filters

PROCESSED_DIR = Path("processed")

OPENCVD_OPERATIONS = {
    "translate", "rotate", "reflect", "crop", "shear",
    "edge_detect", "equalize",
}
PILLOW_OPERATIONS = {
    "brightness", "contrast", "color", "sharpness", "noise_remove",
}
SKIMAGE_OPERATIONS = {
    "grayscale", "hsv", "segment", "blur",
}

YOLO_OPERATIONS = {"detect_objects"}

ALL_OPERATIONS = OPENCVD_OPERATIONS | PILLOW_OPERATIONS | SKIMAGE_OPERATIONS | YOLO_OPERATIONS

MODEL_PATH = os.environ.get("YOLO_MODEL_PATH", "yolov8s.pt")
_standard_model = None
_sahi_model = None


def get_standard_model():
    global _standard_model
    if _standard_model is None:
        try:
            from ultralytics import YOLO
        except ModuleNotFoundError:
            raise ValueError("Object detection requires ultralytics: pip install ultralytics")
        _standard_model = YOLO(MODEL_PATH)
    return _standard_model


def get_sahi_model():
    global _sahi_model
    if _sahi_model is None:
        try:
            from sahi import AutoDetectionModel
        except ModuleNotFoundError:
            raise ValueError("SAHI requires sahi: pip install sahi>=0.11.15")
        _sahi_model = AutoDetectionModel.from_pretrained(
            model_type="ultralytics",
            model_path="yolov8s.pt",
            confidence_threshold=0.15,
            device="cpu",
        )
    return _sahi_model


def process(image_path, operation, params=None):
    params = params or {}
    img_bgr = cv2.imread(str(image_path))
    if img_bgr is None:
        raise ValueError(f"Could not load image: {image_path}")

    h, w = img_bgr.shape[:2]
    original_size = {"width": w, "height": h}

    if operation in OPENCVD_OPERATIONS:
        result = _run_opencv(img_bgr, operation, params)
        result_bgr = result
    elif operation in PILLOW_OPERATIONS:
        pil_img = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        result_pil = _run_pillow(pil_img, operation, params)
        result_bgr = cv2.cvtColor(np.asarray(result_pil), cv2.COLOR_RGB2BGR)
    elif operation in SKIMAGE_OPERATIONS:
        result_rgb = _run_skimage(img_bgr, operation, params)
        result_bgr = result_rgb
    elif operation in YOLO_OPERATIONS:
        annotated_bgr, heatmap_bgr, overlay_bgr, extra = _run_yolo(img_bgr, operation, params)
        result_bgr = annotated_bgr
    else:
        raise ValueError(f"Unknown operation: {operation}")

    output_filename = f"{uuid.uuid4().hex}.png"
    output_path = PROCESSED_DIR / output_filename
    PROCESSED_DIR.mkdir(exist_ok=True)
    cv2.imwrite(str(output_path), result_bgr)

    h2, w2 = result_bgr.shape[:2]
    result = {
        "output_path": str(output_path),
        "original_size": original_size,
        "new_size": {"width": w2, "height": h2},
        "operation": operation,
        "params": params,
    }
    if operation in YOLO_OPERATIONS:
        # Save heatmap and overlay alongside the annotated output
        stem = output_path.stem  # UUID
        heatmap_path = output_path.with_name(f"{stem}_heatmap.jpg")
        overlay_path = output_path.with_name(f"{stem}_overlay.jpg")
        cv2.imwrite(str(heatmap_path), heatmap_bgr)
        cv2.imwrite(str(overlay_path), overlay_bgr)
        extra["heatmap_filename"] = heatmap_path.name
        extra["overlay_filename"] = overlay_path.name
        result.update(extra)
    return result


def pipeline(image_path, operations_list):
    current_path = str(image_path)
    steps = []

    for op_spec in operations_list:
        if isinstance(op_spec, str):
            op_name = op_spec
            op_params = {}
        else:
            op_name = op_spec["operation"]
            op_params = op_spec.get("params", {})

        result = process(current_path, op_name, op_params)
        current_path = result["output_path"]
        steps.append(result)

    return {
        "output_path": current_path,
        "steps": steps,
    }


def _run_opencv(img, operation, params):
    if operation == "translate":
        dx = params.get("dx", 0)
        dy = params.get("dy", 0)
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        return cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))

    if operation == "rotate":
        angle = params.get("angle", 0)
        scale = params.get("scale", 1.0)
        h, w = img.shape[:2]
        center = (w / 2, h / 2)
        M = cv2.getRotationMatrix2D(center, angle, scale)
        return cv2.warpAffine(img, M, (w, h))

    if operation == "reflect":
        axis = params.get("axis", 1)
        return cv2.flip(img, axis)

    if operation == "crop":
        x = params.get("x", 0)
        y = params.get("y", 0)
        width = params.get("width", img.shape[1])
        height = params.get("height", img.shape[0])
        return img[y : y + height, x : x + width]

    if operation == "shear":
        shear_factor = params.get("shear_factor", 0.0)
        h, w = img.shape[:2]
        M = np.float32([[1, shear_factor, 0], [0, 1, 0]])
        return cv2.warpAffine(img, M, (w, h))

    if operation == "edge_detect":
        threshold1 = params.get("threshold1", 100)
        threshold2 = params.get("threshold2", 200)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, threshold1, threshold2)
        return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    if operation == "equalize":
        yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])
        return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)

    raise ValueError(f"Unknown OpenCV operation: {operation}")


def _run_pillow(img, operation, params):
    if operation == "brightness":
        factor = params.get("factor", 1.0)
        return ImageEnhance.Brightness(img).enhance(factor)

    if operation == "contrast":
        factor = params.get("factor", 1.0)
        return ImageEnhance.Contrast(img).enhance(factor)

    if operation == "color":
        factor = params.get("factor", 1.0)
        return ImageEnhance.Color(img).enhance(factor)

    if operation == "sharpness":
        factor = params.get("factor", 1.0)
        return ImageEnhance.Sharpness(img).enhance(factor)

    if operation == "noise_remove":
        size = params.get("size", 3)
        return img.filter(ImageFilter.MedianFilter(size=size))

    raise ValueError(f"Unknown Pillow operation: {operation}")


def _run_skimage(img_bgr, operation, params):
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    if operation == "grayscale":
        gray = color.rgb2gray(rgb)
        gray_8 = (gray * 255).astype(np.uint8)
        return cv2.cvtColor(gray_8, cv2.COLOR_GRAY2BGR)

    if operation == "hsv":
        hsv_img = color.rgb2hsv(rgb)
        hsv_display = (hsv_img * 255).astype(np.uint8)
        return cv2.cvtColor(hsv_display, cv2.COLOR_RGB2BGR)

    if operation == "segment":
        threshold = params.get("threshold", 0.5)
        gray = color.rgb2gray(rgb)
        binary = gray > threshold
        binary_8 = (binary * 255).astype(np.uint8)
        return cv2.cvtColor(binary_8, cv2.COLOR_GRAY2BGR)

    if operation == "blur":
        sigma = params.get("sigma", 1.0)
        blurred = np.zeros_like(rgb, dtype=np.float64)
        for c in range(3):
            blurred[..., c] = filters.gaussian(rgb[..., c].astype(np.float64), sigma=sigma)
        blurred_8 = blurred.astype(np.uint8)
        return cv2.cvtColor(blurred_8, cv2.COLOR_RGB2BGR)

    raise ValueError(f"Unknown skimage operation: {operation}")


def compute_iou(box1, box2):
    xi1 = max(box1[0], box2[0])
    yi1 = max(box1[1], box2[1])
    xi2 = min(box1[2], box2[2])
    yi2 = min(box1[3], box2[3])
    inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0


def _run_yolo(img_bgr, operation, params):
    from scipy.ndimage import gaussian_filter

    conf_threshold = params.get("conf", 0.15)
    h, w = img_bgr.shape[:2]
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    thickness = max(2, int(min(h, w) / 300))
    font_scale = max(0.5, min(h, w) / 800)
    font = cv2.FONT_HERSHEY_SIMPLEX

    annotated = img_bgr.copy()

    # Pre-load both models before inference to avoid cross-contamination
    standard_model = get_standard_model()
    sahi_model = get_sahi_model()

    # ------------------------------------------------------------------
    # PASS 1 — Standard inference (large / medium objects)
    # ------------------------------------------------------------------
    max_dim = max(h, w)
    if max_dim > 1280:
        scale = 1280 / max_dim
        resized_bgr = cv2.resize(img_bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LINEAR)
    else:
        resized_bgr = img_bgr

    results = standard_model.predict(source=resized_bgr, imgsz=1280, conf=conf_threshold, iou=0.45, verbose=False)
    scale_x = w / resized_bgr.shape[1]
    scale_y = h / resized_bgr.shape[0]

    standard_dets = []
    for result in results:
        if result.boxes is None:
            continue
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            standard_dets.append({
                "label": result.names[int(box.cls[0])],
                "confidence": round(float(box.conf[0]), 4),
                "bbox": [int(x1 * scale_x), int(y1 * scale_y), int(x2 * scale_x), int(y2 * scale_y)],
                "source": "standard",
            })

    # ------------------------------------------------------------------
    # PASS 2 — SAHI sliced inference (small / distant objects)
    # ------------------------------------------------------------------
    sahi_dets = []
    try:
        from sahi.predict import get_sliced_prediction

        sahi_result = get_sliced_prediction(
            image=img_rgb,
            detection_model=sahi_model,
            slice_height=512,
            slice_width=512,
            overlap_height_ratio=0.2,
            overlap_width_ratio=0.2,
            perform_standard_pred=False,
            postprocess_type="NMM",
            postprocess_match_threshold=0.5,
            verbose=0,
        )
        for obj in sahi_result.object_prediction_list:
            bbox = obj.bbox.to_xyxy()
            sahi_dets.append({
                "label": obj.category.name,
                "confidence": round(float(obj.score.value), 4),
                "bbox": [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])],
                "source": "sahi",
            })
    except Exception:
        pass

    # ------------------------------------------------------------------
    # MERGE — IoU-based deduplication
    # ------------------------------------------------------------------
    all_dets = standard_dets + sahi_dets
    all_dets.sort(key=lambda d: d["confidence"], reverse=True)

    merged = []
    for d in all_dets:
        dup = False
        for kept in merged:
            if compute_iou(d["bbox"], kept["bbox"]) > 0.5:
                dup = True
                break
        if not dup:
            merged.append(d)

    # ------------------------------------------------------------------
    # DRAW — green for standard, cyan for SAHI
    # ------------------------------------------------------------------
    for d in merged:
        x1, y1, x2, y2 = d["bbox"]
        if d["source"] == "sahi":
            color = (255, 200, 0)
            label_text = f"{d['label']} {d['confidence']:.0%} [S]"
        else:
            color = (80, 200, 0)
            label_text = f"{d['label']} {d['confidence']:.0%}"

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)

        (tw, th), _ = cv2.getTextSize(label_text, font, font_scale, thickness)
        pad = 4
        lx1 = x1
        lx2 = x1 + tw + pad * 2
        if lx2 > x2:
            lx1 = max(0, min(x1, x2 - (tw + pad * 2)))
            lx2 = lx1 + tw + pad * 2
        if lx1 < 0:
            lx1 = 0
            lx2 = min(lx1 + tw + pad * 2, w)
        ly1 = max(0, y1 - th - pad * 2)
        cv2.rectangle(annotated, (lx1, ly1), (lx2, y1), color, -1)
        cv2.putText(annotated, label_text, (lx1 + pad, y1 - pad), font, font_scale, (255, 255, 255), thickness)

    # Legend
    lx, ly = 12, 20
    leg_h = 22
    cv2.rectangle(annotated, (lx - 4, ly - 4), (lx + 220, ly + leg_h * 2 + 8), (30, 30, 30), -1)
    cv2.rectangle(annotated, (lx, ly + 2), (lx + 14, ly + 16), (80, 200, 0), -1)
    cv2.putText(annotated, "Standard", (lx + 20, ly + 14), font, 0.45, (200, 200, 200), 1)
    lx2 = lx + 90
    cv2.rectangle(annotated, (lx2, ly + 2 + leg_h), (lx2 + 14, ly + 16 + leg_h), (255, 200, 0), -1)
    cv2.putText(annotated, "SAHI (small)", (lx2 + 20, ly + 14 + leg_h), font, 0.45, (200, 200, 200), 1)

    # ------------------------------------------------------------------
    # HEATMAP — Gaussian blobs per detection
    # ------------------------------------------------------------------
    heat_raw = np.zeros((h, w), dtype=np.float32)
    for d in merged:
        x1, y1, x2, y2 = d["bbox"]
        conf = d["confidence"]
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        sigma = max(20, (x2 - x1 + y2 - y1) // 8)
        temp = np.zeros_like(heat_raw)
        temp[cy, cx] = conf
        blurred = gaussian_filter(temp, sigma=sigma)
        heat_raw += blurred

    if heat_raw.max() > 0:
        heat_norm = (heat_raw / heat_raw.max() * 255).astype(np.uint8)
    else:
        heat_norm = heat_raw.astype(np.uint8)

    heatmap_bgr = cv2.applyColorMap(heat_norm, cv2.COLORMAP_JET)
    overlay_bgr = cv2.addWeighted(img_bgr, 0.5, heatmap_bgr, 0.5, 0)

    # ------------------------------------------------------------------
    # RESPONSE (pre-merge counts, post-merge details)
    # ------------------------------------------------------------------
    std_count = len(standard_dets)
    sahi_count = len(sahi_dets)

    return annotated, heatmap_bgr, overlay_bgr, {
        "detections": merged,
        "standard_count": std_count,
        "sahi_count": sahi_count,
        "total_after_dedup": len(merged),
        "model": "yolov8s + SAHI",
        "confidence_threshold": conf_threshold,
        "slice_size": "512x512",
    }


process_image = pipeline
