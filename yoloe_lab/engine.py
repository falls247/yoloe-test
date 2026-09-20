
from __future__ import annotations

import tempfile
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLOE
from ultralytics.models.yolo.yoloe import YOLOEVPSegPredictor

PROMPTABLE_MODELS = [
    "yoloe-26n-seg.pt",
    "yoloe-26s-seg.pt",
    "yoloe-26m-seg.pt",
    "yoloe-26l-seg.pt",
    "yoloe-26x-seg.pt",
    "yoloe-11s-seg.pt",
    "yoloe-11m-seg.pt",
    "yoloe-11l-seg.pt",
    "yoloe-11x-seg.pt",
]


def prompt_free_model(promptable_model: str) -> str:
    if not promptable_model.endswith("-seg.pt"):
        raise ValueError(f"Unexpected YOLOE model name: {promptable_model}")
    return promptable_model.replace("-seg.pt", "-seg-pf.pt")


def parse_classes(text: str) -> list[str]:
    items = [x.strip() for x in text.replace("\n", ",").split(",")]
    classes = [x for x in items if x]
    if not classes:
        raise ValueError("Text promptに1つ以上のクラス名を入力してください。")
    if " " in classes:
        raise ValueError("空白1文字だけのクラス名は使用できません。")
    return classes


def parse_visual_boxes(text: str) -> tuple[np.ndarray, np.ndarray, dict[int, str]]:
    """Parse one prompt per line: x1,y1,x2,y2,class_id[,label]."""
    boxes: list[list[float]] = []
    cls_ids: list[int] = []
    labels: dict[int, str] = {}

    for lineno, raw in enumerate(text.splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 5:
            raise ValueError(f"{lineno}行目: x1,y1,x2,y2,class_id[,label] の形式で入力してください。")
        try:
            x1, y1, x2, y2 = map(float, parts[:4])
            class_id = int(parts[4])
        except ValueError as exc:
            raise ValueError(f"{lineno}行目: bboxまたはclass_idを数値として解釈できません。") from exc
        if x2 <= x1 or y2 <= y1:
            raise ValueError(f"{lineno}行目: x2>x1 かつ y2>y1 が必要です。")
        if class_id < 0:
            raise ValueError(f"{lineno}行目: class_idは0以上が必要です。")
        boxes.append([x1, y1, x2, y2])
        cls_ids.append(class_id)
        if len(parts) >= 6 and parts[5]:
            labels[class_id] = parts[5]

    if not boxes:
        raise ValueError("Visual Prompt用bboxを1つ以上入力してください。")

    unique = sorted(set(cls_ids))
    expected = list(range(len(unique)))
    if unique != expected:
        raise ValueError(f"class_idは0から連番にしてください。現在={unique}, 必要={expected}")

    return np.asarray(boxes, dtype=np.float32), np.asarray(cls_ids, dtype=np.int32), labels


def _device_arg(device: str) -> dict:
    device = (device or "auto").strip()
    return {} if device == "auto" else {"device": device}


def load_text_model(model_name: str, classes: list[str], device: str) -> tuple[YOLOE, np.ndarray]:
    model = YOLOE(model_name)
    pe = model.get_text_pe(classes)
    model.set_classes(classes, pe)
    return model, pe.detach().cpu().float().numpy()


def load_visual_model(
    model_name: str,
    reference_path: str,
    boxes: np.ndarray,
    cls_ids: np.ndarray,
    imgsz: int,
    conf: float,
    device: str,
) -> tuple[YOLOE, np.ndarray]:
    model = YOLOE(model_name)
    prompts = {"bboxes": boxes, "cls": cls_ids}
    kwargs = {
        "imgsz": int(imgsz),
        "conf": float(conf),
        "rect": False,
        "verbose": False,
        **_device_arg(device),
    }
    model.predict(
        source=reference_path,
        refer_image=reference_path,
        visual_prompts=prompts,
        predictor=YOLOEVPSegPredictor,
        **kwargs,
    )
    pe = getattr(model.model, "pe", None)
    if pe is None:
        raise RuntimeError("Visual Prompt embedding (model.model.pe) を取得できませんでした。")
    return model, pe.detach().cpu().float().numpy()


def load_prompt_free_model(model_name: str) -> YOLOE:
    return YOLOE(prompt_free_model(model_name))


def process_video(
    model: YOLOE,
    video_path: str,
    *,
    imgsz: int,
    conf: float,
    device: str,
    show_masks: bool,
    max_frames: int = 0,
    label_map: dict[int, str] | None = None,
    progress=None,
) -> tuple[str, dict]:
    if not video_path:
        raise ValueError("動画をアップロードしてください。")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"動画を開けません: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_source_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    if width <= 0 or height <= 0:
        cap.release()
        raise ValueError("動画サイズを取得できません。")

    out_dir = Path(tempfile.mkdtemp(prefix="yoloe_lab_"))
    out_path = out_dir / "result.mp4"
    writer = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        (width, height),
    )
    if not writer.isOpened():
        cap.release()
        raise RuntimeError("出力動画writerを初期化できません。")

    frame_limit = total_source_frames
    if max_frames and max_frames > 0:
        frame_limit = min(frame_limit or max_frames, int(max_frames))

    count = 0
    detections = 0
    class_counts: Counter[str] = Counter()
    inference_ms: list[float] = []
    started = time.perf_counter()
    kwargs = {
        "imgsz": int(imgsz),
        "conf": float(conf),
        "rect": False,
        "verbose": False,
        **_device_arg(device),
    }

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if max_frames and count >= int(max_frames):
                break

            result = model.predict(frame, **kwargs)[0]
            plotted = result.plot(masks=bool(show_masks), boxes=True, labels=True, conf=True)

            if label_map and result.boxes is not None and len(result.boxes):
                for box, cls_t in zip(result.boxes.xyxy.cpu().numpy(), result.boxes.cls.cpu().numpy()):
                    cid = int(cls_t)
                    label = label_map.get(cid)
                    if label:
                        x1, y1, _, _ = map(int, box)
                        cv2.putText(
                            plotted,
                            label,
                            (max(0, x1), max(15, y1 - 18)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.55,
                            (255, 255, 255),
                            2,
                            cv2.LINE_AA,
                        )

            writer.write(plotted)
            count += 1

            if result.boxes is not None and len(result.boxes):
                cls_values = result.boxes.cls.detach().cpu().numpy().astype(int)
                detections += len(cls_values)
                for cid in cls_values:
                    name = result.names.get(cid, str(cid)) if isinstance(result.names, dict) else result.names[cid]
                    if label_map and cid in label_map:
                        name = f"{name} / {label_map[cid]}"
                    class_counts[name] += 1

            if getattr(result, "speed", None) and result.speed.get("inference") is not None:
                inference_ms.append(float(result.speed["inference"]))

            if progress is not None and frame_limit:
                progress(min(count / max(frame_limit, 1), 1.0), desc=f"{count}/{frame_limit} frames")
    finally:
        cap.release()
        writer.release()

    elapsed = time.perf_counter() - started
    summary = {
        "frames_processed": count,
        "source_fps": round(float(fps), 3),
        "wall_time_sec": round(elapsed, 3),
        "effective_fps": round(count / elapsed, 3) if elapsed else None,
        "avg_model_inference_ms": round(float(np.mean(inference_ms)), 3) if inference_ms else None,
        "detections_total": int(detections),
        "class_counts": dict(class_counts),
        "output": str(out_path),
    }
    return str(out_path), summary
