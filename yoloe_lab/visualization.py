
from __future__ import annotations

import cv2
import numpy as np
from PIL import Image


def _annotate_bgr(image: np.ndarray, text: str) -> np.ndarray:
    out = image.copy()
    cv2.rectangle(out, (0, 0), (min(out.shape[1], 900), 34), (0, 0, 0), -1)
    cv2.putText(out, text, (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def _to_pil(bgr: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))


def read_image(path: str) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"参考画像を開けません: {path}")
    return image


def validate_boxes(boxes: np.ndarray, image_shape: tuple[int, int]) -> None:
    h, w = image_shape
    if np.any(boxes[:, 0] < 0) or np.any(boxes[:, 1] < 0):
        raise ValueError("bbox座標は0以上にしてください。")
    if np.any(boxes[:, 2] > w) or np.any(boxes[:, 3] > h):
        raise ValueError(f"bboxが画像範囲を超えています。画像サイズ={w}x{h}")


def letterbox_visual(
    image: np.ndarray,
    boxes: np.ndarray,
    imgsz: int,
) -> tuple[np.ndarray, np.ndarray, float, tuple[int, int]]:
    src_h, src_w = image.shape[:2]
    dst_h = dst_w = int(imgsz)
    gain = min(dst_h / src_h, dst_w / src_w)
    new_w = round(src_w * gain)
    new_h = round(src_h * gain)

    left = round((dst_w - new_w) / 2 - 0.1)
    top = round((dst_h - new_h) / 2 - 0.1)
    right = dst_w - new_w - left
    bottom = dst_h - new_h - top

    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = cv2.copyMakeBorder(
        resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114)
    )

    scaled = boxes.astype(np.float32).copy()
    scaled *= gain
    scaled[:, 0::2] += left
    scaled[:, 1::2] += top
    return canvas, scaled, gain, (left, top)


def rasterize_prompt_masks(
    scaled_boxes: np.ndarray,
    cls_ids: np.ndarray,
    imgsz: int,
    scale_factor: float = 1 / 8,
) -> np.ndarray:
    mask_h = int(int(imgsz) * scale_factor)
    mask_w = int(int(imgsz) * scale_factor)
    xs = np.arange(mask_w, dtype=np.float32)[None, :]
    ys = np.arange(mask_h, dtype=np.float32)[:, None]
    classes = sorted(set(int(x) for x in cls_ids.tolist()))
    visuals = np.zeros((len(classes), mask_h, mask_w), dtype=np.float32)

    for box, cid in zip(scaled_boxes, cls_ids):
        x1, y1, x2, y2 = box * scale_factor
        mask = (xs >= x1) & (xs < x2) & (ys >= y1) & (ys < y2)
        visuals[int(cid)] = np.logical_or(visuals[int(cid)] > 0, mask).astype(np.float32)

    return visuals


def _embedding_image(embedding: np.ndarray, width: int = 1000, row_height: int = 80) -> Image.Image:
    arr = np.asarray(embedding, dtype=np.float32)
    if arr.ndim == 3:
        arr = arr[0]
    if arr.ndim != 2:
        raise ValueError(f"Unexpected embedding shape: {arr.shape}")
    lo, hi = float(arr.min()), float(arr.max())
    if hi - lo < 1e-12:
        norm = np.zeros_like(arr, dtype=np.uint8)
    else:
        norm = ((arr - lo) / (hi - lo) * 255).clip(0, 255).astype(np.uint8)
    h = max(row_height * arr.shape[0], 80)
    heat = cv2.resize(norm, (width, h), interpolation=cv2.INTER_NEAREST)
    heat = cv2.cvtColor(heat, cv2.COLOR_GRAY2BGR)
    heat = _annotate_bgr(heat, f"Exact prompt embedding  shape={tuple(arr.shape)}  min={lo:.4f} max={hi:.4f}")
    return _to_pil(heat)


def _similarity_image(embedding: np.ndarray) -> Image.Image | None:
    arr = np.asarray(embedding, dtype=np.float32)
    if arr.ndim == 3:
        arr = arr[0]
    if arr.ndim != 2 or arr.shape[0] < 2:
        return None
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    normed = arr / np.clip(norms, 1e-12, None)
    sim = normed @ normed.T
    gray = ((sim + 1) * 127.5).clip(0, 255).astype(np.uint8)
    canvas = cv2.resize(gray, (480, 480), interpolation=cv2.INTER_NEAREST)
    canvas = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)
    canvas = _annotate_bgr(canvas, "Cosine similarity between prompt embeddings (-1..1)")
    return _to_pil(canvas)


def build_visual_gallery(
    reference_path: str,
    boxes: np.ndarray,
    cls_ids: np.ndarray,
    imgsz: int,
    embedding: np.ndarray | None = None,
    labels: dict[int, str] | None = None,
) -> tuple[list[tuple[Image.Image, str]], dict]:
    labels = labels or {}
    image = read_image(reference_path)
    validate_boxes(boxes, image.shape[:2])
    gallery: list[tuple[Image.Image, str]] = []

    annotated = image.copy()
    for box, cid in zip(boxes, cls_ids):
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 255, 255), 2)
        txt = f"c{int(cid)}"
        if int(cid) in labels:
            txt += f":{labels[int(cid)]}"
        cv2.putText(annotated, txt, (x1, max(18, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
    gallery.append((_to_pil(_annotate_bgr(annotated, "1. Reference image + user bbox")), "Reference + bbox"))

    for i, (box, cid) in enumerate(zip(boxes, cls_ids)):
        x1, y1, x2, y2 = map(int, box)
        crop = image[y1:y2, x1:x2].copy()
        name = labels.get(int(cid), f"object{int(cid)}")
        gallery.append((_to_pil(_annotate_bgr(crop, f"2. ROI #{i} class={cid} {name}")), f"ROI #{i}"))

    letterboxed, scaled_boxes, gain, pad = letterbox_visual(image, boxes, imgsz)
    stage = letterboxed.copy()
    for box, cid in zip(scaled_boxes, cls_ids):
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(stage, (x1, y1), (x2, y2), (255,255,255), 2)
        cv2.putText(stage, f"c{int(cid)}", (x1, max(18, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
    gallery.append((_to_pil(_annotate_bgr(stage, f"3. Letterbox {imgsz}x{imgsz} gain={gain:.4f} pad={pad}")), "Letterbox + scaled bbox"))

    visuals = rasterize_prompt_masks(scaled_boxes, cls_ids, imgsz)
    for cid in range(visuals.shape[0]):
        mask = (visuals[cid] * 255).astype(np.uint8)
        display = cv2.resize(mask, (imgsz, imgsz), interpolation=cv2.INTER_NEAREST)
        display = cv2.cvtColor(display, cv2.COLOR_GRAY2BGR)
        display = _annotate_bgr(display, f"4. Exact visual prompt tensor class={cid} shape={visuals[cid].shape}")
        gallery.append((_to_pil(display), f"Prompt mask class {cid}"))

    details = {
        "reference_size": {"width": int(image.shape[1]), "height": int(image.shape[0])},
        "letterbox_size": [int(imgsz), int(imgsz)],
        "gain": float(gain),
        "padding_left_top": [int(pad[0]), int(pad[1])],
        "scaled_bboxes_xyxy": scaled_boxes.round(3).tolist(),
        "prompt_tensor_shape": list(visuals.shape),
        "prompt_nonzero_pixels": [int((m > 0).sum()) for m in visuals],
    }

    if embedding is not None:
        gallery.append((_embedding_image(embedding), "Final prompt embedding"))
        sim_img = _similarity_image(embedding)
        if sim_img is not None:
            gallery.append((sim_img, "Embedding cosine similarity"))
        emb = np.asarray(embedding, dtype=np.float32)
        if emb.ndim == 3:
            emb = emb[0]
        details["embedding"] = {
            "shape": list(embedding.shape),
            "per_class_l2_norm": np.linalg.norm(emb, axis=1).round(6).tolist(),
            "mean": float(emb.mean()),
            "std": float(emb.std()),
            "min": float(emb.min()),
            "max": float(emb.max()),
        }

    return gallery, details
