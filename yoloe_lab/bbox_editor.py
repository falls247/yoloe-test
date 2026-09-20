from __future__ import annotations

from typing import Any

import cv2
import numpy as np


def _read_bgr(reference_path: str) -> np.ndarray:
    if not reference_path:
        raise ValueError("参考画像をアップロードしてください。")
    image = cv2.imread(str(reference_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"参考画像を開けません: {reference_path}")
    return image


def render_editor(
    reference_path: str | None,
    items: list[dict[str, Any]] | None,
    pending_point: list[int] | None = None,
) -> np.ndarray | None:
    """Render reference image with all committed boxes and an optional first-corner marker."""
    if not reference_path:
        return None

    image = _read_bgr(reference_path)
    items = items or []

    for i, item in enumerate(items):
        x1, y1, x2, y2 = [int(round(v)) for v in item["bbox"]]
        cid = int(item["class_id"])
        label = str(item.get("label") or "").strip()
        text = f"#{i} c{cid}" + (f" {label}" if label else "")
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 255, 255), 3)
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.62, 2)
        ty = max(th + 6, y1)
        cv2.rectangle(image, (x1, ty - th - 8), (x1 + tw + 8, ty + 4), (0, 0, 0), -1)
        cv2.putText(
            image,
            text,
            (x1 + 4, ty - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    if pending_point is not None:
        x, y = map(int, pending_point)
        length = max(10, min(image.shape[:2]) // 40)
        cv2.line(image, (max(0, x - length), y), (min(image.shape[1] - 1, x + length), y), (255, 255, 255), 2)
        cv2.line(image, (x, max(0, y - length)), (x, min(image.shape[0] - 1, y + length)), (255, 255, 255), 2)
        cv2.circle(image, (x, y), 5, (255, 255, 255), -1)

    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def bbox_rows(items: list[dict[str, Any]] | None) -> list[list[Any]]:
    rows: list[list[Any]] = []
    for i, item in enumerate(items or []):
        x1, y1, x2, y2 = item["bbox"]
        rows.append(
            [
                i,
                int(item["class_id"]),
                str(item.get("label") or ""),
                round(float(x1), 1),
                round(float(y1), 1),
                round(float(x2), 1),
                round(float(y2), 1),
                round(float((x2 - x1) * (y2 - y1)), 1),
            ]
        )
    return rows


def reset_editor(reference_path: str | None):
    if not reference_path:
        return None, [], None, [], "参考画像をアップロードしてください。"
    image = _read_bgr(reference_path)
    h, w = image.shape[:2]
    return (
        cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
        [],
        None,
        [],
        f"画像サイズ: **{w}×{h}**。BBoxの左上と右下を順に2回クリック。",
    )


def add_click(
    reference_path: str,
    items: list[dict[str, Any]] | None,
    pending_point: list[int] | None,
    class_id: int | float,
    label: str | None,
    point: list[int] | tuple[int, int],
):
    image = _read_bgr(reference_path)
    h, w = image.shape[:2]
    x, y = int(point[0]), int(point[1])
    x = min(max(x, 0), w - 1)
    y = min(max(y, 0), h - 1)
    items = list(items or [])

    if pending_point is None:
        pending = [x, y]
        status = (
            f"1点目 **({x}, {y})** を設定。"
            " 対角側の2点目をクリックするとBBoxを確定。"
        )
        return render_editor(reference_path, items, pending), items, pending, bbox_rows(items), status

    px, py = map(int, pending_point)
    x1, x2 = sorted((px, x))
    y1, y2 = sorted((py, y))
    if x2 - x1 < 3 or y2 - y1 < 3:
        status = "BBoxが小さすぎる。1点目を保持したまま、離れた位置をクリック。"
        return render_editor(reference_path, items, pending_point), items, pending_point, bbox_rows(items), status

    cid = int(class_id or 0)
    if cid < 0:
        raise ValueError("class_idは0以上を指定してください。")

    clean_label = str(label or "").strip()
    items.append(
        {
            "bbox": [float(x1), float(y1), float(x2), float(y2)],
            "class_id": cid,
            "label": clean_label,
        }
    )

    area_ratio = ((x2 - x1) * (y2 - y1)) / float(w * h)
    warning = ""
    if area_ratio < 0.01:
        warning = " 画像全体の1%未満。部品全体ではなく一部分だけをPrompt化していないか確認。"

    status = (
        f"BBox #{len(items) - 1} を追加: **({x1}, {y1}) - ({x2}, {y2})** / "
        f"class_id={cid} / 画像占有率={area_ratio * 100:.2f}%"
        + warning
        + " 次のBBoxは再び1点目から指定。"
    )
    return render_editor(reference_path, items), items, None, bbox_rows(items), status


def undo_last(reference_path: str | None, items: list[dict[str, Any]] | None):
    items = list(items or [])
    if items:
        items.pop()
    status = f"BBox数: {len(items)}"
    return render_editor(reference_path, items), items, None, bbox_rows(items), status


def clear_all(reference_path: str | None):
    return render_editor(reference_path, []), [], None, [], "BBoxをすべて削除。"


def state_to_visual_boxes(
    items: list[dict[str, Any]] | None,
) -> tuple[np.ndarray, np.ndarray, dict[int, str]]:
    items = list(items or [])
    if not items:
        raise ValueError("参考画像上でBBoxを1つ以上指定してください。")

    boxes: list[list[float]] = []
    cls_ids: list[int] = []
    labels: dict[int, str] = {}

    for i, item in enumerate(items):
        box = [float(v) for v in item["bbox"]]
        if len(box) != 4 or box[2] <= box[0] or box[3] <= box[1]:
            raise ValueError(f"BBox #{i} が不正です。")
        cid = int(item["class_id"])
        if cid < 0:
            raise ValueError(f"BBox #{i}: class_idは0以上が必要です。")
        label = str(item.get("label") or "").strip()

        boxes.append(box)
        cls_ids.append(cid)

        if label:
            if cid in labels and labels[cid] != label:
                raise ValueError(
                    f"class_id={cid} に異なるlabelが指定されています: "
                    f"'{labels[cid]}' と '{label}'"
                )
            labels[cid] = label

    unique = sorted(set(cls_ids))
    expected = list(range(len(unique)))
    if unique != expected:
        raise ValueError(f"class_idは0から連番にしてください。現在={unique}, 必要={expected}")

    return (
        np.asarray(boxes, dtype=np.float32),
        np.asarray(cls_ids, dtype=np.int32),
        labels,
    )
