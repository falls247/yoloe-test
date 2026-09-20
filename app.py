from __future__ import annotations

from functools import wraps

import gradio as gr

from yoloe_lab.bbox_editor import (
    add_click,
    clear_all,
    reset_editor,
    state_to_visual_boxes,
    undo_last,
)
from yoloe_lab.engine import (
    PROMPTABLE_MODELS,
    load_prompt_free_model,
    load_text_model,
    load_visual_model,
    parse_classes,
    process_video,
    prompt_free_model,
)
from yoloe_lab.visualization import build_visual_gallery

DEFAULT_MODEL = "yoloe-26s-seg.pt"


def _wrap_error(fn):
    @wraps(fn)
    def inner(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except gr.Error:
            raise
        except Exception as exc:
            raise gr.Error(str(exc)) from exc

    return inner


@_wrap_error
def reset_visual_editor(reference_path):
    return reset_editor(reference_path)


@_wrap_error
def add_visual_bbox(reference_path, items, pending_point, class_id, label, evt: gr.SelectData):
    if not reference_path:
        raise ValueError("先に参考画像をアップロードしてください。")
    if not isinstance(evt.index, (list, tuple)) or len(evt.index) < 2:
        raise ValueError("画像上のクリック座標を取得できませんでした。")
    return add_click(
        reference_path,
        items,
        pending_point,
        class_id,
        label,
        evt.index,
    )


@_wrap_error
def undo_visual_bbox(reference_path, items):
    return undo_last(reference_path, items)


@_wrap_error
def clear_visual_bboxes(reference_path):
    return clear_all(reference_path)


@_wrap_error
def preview_visual_prompt(reference_path, bbox_items, imgsz):
    if not reference_path:
        raise ValueError("参考画像をアップロードしてください。")
    boxes, cls_ids, labels = state_to_visual_boxes(bbox_items)
    gallery, details = build_visual_gallery(
        reference_path, boxes, cls_ids, int(imgsz), embedding=None, labels=labels
    )
    return gallery, details


@_wrap_error
def run_text(
    video_path,
    classes_text,
    model_name,
    conf,
    imgsz,
    device,
    show_masks,
    max_frames,
    progress=gr.Progress(),
):
    classes = parse_classes(classes_text)
    progress(0.02, desc="Text prompt embedding生成")
    model, embedding = load_text_model(model_name, classes, device)
    progress(0.05, desc="動画推論開始")
    out, summary = process_video(
        model,
        video_path,
        imgsz=int(imgsz),
        conf=float(conf),
        device=device,
        show_masks=bool(show_masks),
        max_frames=int(max_frames or 0),
        progress=progress,
    )
    summary["mode"] = "text"
    summary["model"] = model_name
    summary["classes"] = classes
    summary["prompt_embedding_shape"] = list(embedding.shape)
    return out, summary


@_wrap_error
def run_visual(
    video_path,
    reference_path,
    bbox_items,
    model_name,
    conf,
    imgsz,
    device,
    show_masks,
    max_frames,
    progress=gr.Progress(),
):
    if not reference_path:
        raise ValueError("参考画像をアップロードしてください。")
    boxes, cls_ids, labels = state_to_visual_boxes(bbox_items)

    progress(0.01, desc="Visual prompt前処理")
    build_visual_gallery(
        reference_path, boxes, cls_ids, int(imgsz), embedding=None, labels=labels
    )

    progress(0.03, desc="SAVPE / Visual Prompt embedding生成")
    model, embedding = load_visual_model(
        model_name,
        reference_path,
        boxes,
        cls_ids,
        int(imgsz),
        float(conf),
        device,
    )
    gallery, details = build_visual_gallery(
        reference_path,
        boxes,
        cls_ids,
        int(imgsz),
        embedding=embedding,
        labels=labels,
    )

    label_map = {
        cid: labels.get(cid, f"object{cid}") for cid in sorted(set(int(x) for x in cls_ids.tolist()))
    }
    progress(0.06, desc="動画推論開始")
    out, summary = process_video(
        model,
        video_path,
        imgsz=int(imgsz),
        conf=float(conf),
        device=device,
        show_masks=bool(show_masks),
        max_frames=int(max_frames or 0),
        label_map=label_map,
        progress=progress,
    )
    summary.update(
        {
            "mode": "visual",
            "model": model_name,
            "visual_prompts": [
                {
                    "bbox_xyxy": [float(v) for v in box],
                    "class_id": int(cid),
                    "label": labels.get(int(cid), f"object{int(cid)}"),
                }
                for box, cid in zip(boxes, cls_ids)
            ],
            "visualization": details,
        }
    )
    return out, gallery, summary


@_wrap_error
def run_prompt_free(
    video_path,
    promptable_model_name,
    conf,
    imgsz,
    device,
    show_masks,
    max_frames,
    progress=gr.Progress(),
):
    pf_name = prompt_free_model(promptable_model_name)
    progress(0.03, desc=f"{pf_name} 読込")
    model = load_prompt_free_model(promptable_model_name)
    progress(0.06, desc="動画推論開始")
    out, summary = process_video(
        model,
        video_path,
        imgsz=int(imgsz),
        conf=float(conf),
        device=device,
        show_masks=bool(show_masks),
        max_frames=int(max_frames or 0),
        progress=progress,
    )
    summary["mode"] = "prompt-free"
    summary["model"] = pf_name
    return out, summary


def common_controls():
    with gr.Row():
        model = gr.Dropdown(
            PROMPTABLE_MODELS,
            value=DEFAULT_MODEL,
            label="YOLOE checkpoint",
            info="Text / Visual は *-seg.pt。Prompt-Free は同スケールの *-seg-pf.pt を自動使用",
        )
        device = gr.Dropdown(
            ["auto", "0", "cpu"],
            value="auto",
            label="Device",
            info="RTX使用を固定する場合は 0",
        )
    with gr.Row():
        conf = gr.Slider(0.01, 0.95, value=0.25, step=0.01, label="Confidence")
        imgsz = gr.Dropdown([320, 480, 640, 768, 960], value=640, label="imgsz")
        max_frames = gr.Number(
            value=0,
            precision=0,
            label="最大フレーム数",
            info="0 = 全フレーム。動作確認は100〜300推奨",
        )
        masks = gr.Checkbox(value=True, label="Segmentation mask表示")
    return model, device, conf, imgsz, max_frames, masks


CSS = """
.gradio-container { max-width: 1500px !important; }
.pipeline-note { border-left: 4px solid #888; padding-left: 12px; }
#bbox-editor img { cursor: crosshair !important; }
"""


with gr.Blocks(title="YOLOE Mode Lab", css=CSS) as demo:
    gr.Markdown(
        """
# YOLOE Mode Lab
YOLOE の **Text Prompt / Visual Prompt / Prompt-Free** を同じ動画で試すローカル実験環境。

Visual Prompt タブでは、Ultralytics が実際に使う処理に合わせて
**参考画像 → ユーザー指定BBox → letterbox → 1/8 prompt mask → SAVPE → prompt embedding → 動画推論**
を可視化する。

> 可視化対象は「SAVPEへの実入力 prompt tensor」と「SAVPE後の最終 prompt embedding」。
> SAVPE内部の各層 feature map は公開API外のため、この版では hook していない。
"""
    )

    with gr.Tab("Text Prompt"):
        gr.Markdown("クラス名をテキストで指定。初回のみ CLIP/text encoder の追加ダウンロードが発生する。")
        text_video = gr.Video(label="入力動画", sources=["upload"])
        text_classes = gr.Textbox(
            value="person, car",
            label="Text prompt",
            lines=2,
            info="カンマまたは改行区切り",
        )
        text_model, text_device, text_conf, text_imgsz, text_max_frames, text_masks = common_controls()
        text_run = gr.Button("Text Promptで実行", variant="primary")
        with gr.Row():
            text_out = gr.Video(label="結果動画")
            text_json = gr.JSON(label="実行結果 / timing")
        text_run.click(
            run_text,
            inputs=[
                text_video,
                text_classes,
                text_model,
                text_conf,
                text_imgsz,
                text_device,
                text_masks,
                text_max_frames,
            ],
            outputs=[text_out, text_json],
        )

    with gr.Tab("Visual Prompt"):
        gr.Markdown(
            """
### BBox指定
1. 参考画像をアップロード
2. 下の **BBoxエディタ画像で左上・右下の2点を順にクリック**
3. 白枠で囲まれた範囲が、そのままVisual Prompt対象

同じ `class_id` の複数BBoxは1枚のprompt maskへOR結合。
`class_id` は 0,1,2... の連番。labelは表示用で、YOLOE内部では `object0`, `object1`... として扱われる。

> 座標の手入力は廃止。表示画像のクリック座標からBBoxを生成する。
"""
        )

        bbox_state = gr.State([])
        pending_point = gr.State(None)

        with gr.Row():
            visual_video = gr.Video(label="入力動画", sources=["upload"])
            reference = gr.Image(
                label="参考画像アップロード",
                type="filepath",
                sources=["upload"],
                height=300,
            )

        with gr.Row():
            bbox_class_id = gr.Number(
                value=0,
                precision=0,
                minimum=0,
                label="追加するBBoxの class_id",
                info="同じ対象の複数見本は同じclass_id",
            )
            bbox_label = gr.Textbox(
                value="target",
                label="表示ラベル",
                info="推論には使わない。画面表示用",
            )

        bbox_canvas = gr.Image(
            label="BBoxエディタ: 対角2点をクリック",
            type="numpy",
            interactive=False,
            elem_id="bbox-editor",
            height=620,
            show_download_button=False,
        )
        bbox_status = gr.Markdown("参考画像をアップロードするとBBox指定を開始できる。")

        bbox_table = gr.Dataframe(
            headers=["#", "class_id", "label", "x1", "y1", "x2", "y2", "area_px"],
            datatype=["number", "number", "str", "number", "number", "number", "number", "number"],
            value=[],
            interactive=False,
            label="登録済みBBox",
            wrap=True,
        )

        with gr.Row():
            bbox_undo = gr.Button("最後のBBoxを削除")
            bbox_clear = gr.Button("BBoxを全削除")

        reference.change(
            reset_visual_editor,
            inputs=[reference],
            outputs=[bbox_canvas, bbox_state, pending_point, bbox_table, bbox_status],
        )
        bbox_canvas.select(
            add_visual_bbox,
            inputs=[reference, bbox_state, pending_point, bbox_class_id, bbox_label],
            outputs=[bbox_canvas, bbox_state, pending_point, bbox_table, bbox_status],
        )
        bbox_undo.click(
            undo_visual_bbox,
            inputs=[reference, bbox_state],
            outputs=[bbox_canvas, bbox_state, pending_point, bbox_table, bbox_status],
        )
        bbox_clear.click(
            clear_visual_bboxes,
            inputs=[reference],
            outputs=[bbox_canvas, bbox_state, pending_point, bbox_table, bbox_status],
        )

        visual_model, visual_device, visual_conf, visual_imgsz, visual_max_frames, visual_masks = common_controls()
        with gr.Row():
            visual_preview = gr.Button("Prompt前処理だけ確認")
            visual_run = gr.Button("Visual Promptで動画実行", variant="primary")

        visual_gallery = gr.Gallery(
            label="Visual Prompt処理可視化",
            columns=3,
            rows=2,
            height="auto",
        )
        visual_json = gr.JSON(label="Prompt tensor / embedding 詳細")
        visual_out = gr.Video(label="結果動画")

        visual_preview.click(
            preview_visual_prompt,
            inputs=[reference, bbox_state, visual_imgsz],
            outputs=[visual_gallery, visual_json],
        )
        visual_run.click(
            run_visual,
            inputs=[
                visual_video,
                reference,
                bbox_state,
                visual_model,
                visual_conf,
                visual_imgsz,
                visual_device,
                visual_masks,
                visual_max_frames,
            ],
            outputs=[visual_out, visual_gallery, visual_json],
        )

    with gr.Tab("Prompt-Free"):
        gr.Markdown(
            """
外部プロンプトなし。選択したモデルと同スケールの `*-seg-pf.pt` を自動ロードし、
内蔵 4,585 語 vocabulary から検出する。
"""
        )
        pf_video = gr.Video(label="入力動画", sources=["upload"])
        pf_model, pf_device, pf_conf, pf_imgsz, pf_max_frames, pf_masks = common_controls()
        pf_run = gr.Button("Prompt-Freeで実行", variant="primary")
        with gr.Row():
            pf_out = gr.Video(label="結果動画")
            pf_json = gr.JSON(label="実行結果 / timing")
        pf_run.click(
            run_prompt_free,
            inputs=[
                pf_video,
                pf_model,
                pf_conf,
                pf_imgsz,
                pf_device,
                pf_masks,
                pf_max_frames,
            ],
            outputs=[pf_out, pf_json],
        )

    with gr.Accordion("処理の読み方", open=False):
        gr.Markdown(
            """
1. **Reference + user BBox**: 画像上でユーザーが2点クリックして確定した参照領域
2. **Selected Prompt Region**: BBox以外を暗くして、SAVPEへ対象として指示する範囲を明示
3. **Letterbox + scaled bbox**: `rect=False` 固定で `imgsz × imgsz` にletterbox。bboxも同じ倍率・paddingで移動
4. **Visual prompt tensor**: bboxを1/8解像度で二値化。同一classのbboxはOR結合
5. **Prompt embedding**: YOLOE/SAVPEが生成し `model.model.pe` に保持する実テンソル
6. **Video inference**: 以後の各フレームは保持済みembeddingと領域特徴を照合して検出・segment

`Selected Prompt Region` は説明用のcropではなく、**実際にPrompt Maskで有効になる領域を元画像上へ重ねて表示**する。
"""
        )

if __name__ == "__main__":
    demo.queue(default_concurrency_limit=1).launch(
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True,
    )
