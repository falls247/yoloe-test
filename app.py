
from __future__ import annotations

from functools import wraps

import gradio as gr

from yoloe_lab.engine import (
    PROMPTABLE_MODELS,
    load_prompt_free_model,
    load_text_model,
    load_visual_model,
    parse_classes,
    parse_visual_boxes,
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
def preview_visual_prompt(reference_path, box_text, imgsz):
    if not reference_path:
        raise ValueError("参考画像をアップロードしてください。")
    boxes, cls_ids, labels = parse_visual_boxes(box_text)
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
    box_text,
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
    boxes, cls_ids, labels = parse_visual_boxes(box_text)

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
"""


with gr.Blocks(title="YOLOE Mode Lab", css=CSS) as demo:
    gr.Markdown(
        """
# YOLOE Mode Lab
YOLOE の **Text Prompt / Visual Prompt / Prompt-Free** を同じ動画で試すローカル実験環境。

Visual Prompt タブでは、Ultralytics が実際に使う処理に合わせて
**参考画像 → letterbox → bbox変換 → 1/8 prompt mask → SAVPE → prompt embedding → 動画推論**
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
参考画像上の対象を bbox で指定。同じ `class_id` の複数 bbox は **1枚の prompt mask に OR 結合**される。
`class_id` は 0,1,2... の連番。ラベル文字列は表示用で、YOLOE内部では `object0`, `object1`... として扱われる。
"""
        )
        with gr.Row():
            visual_video = gr.Video(label="入力動画", sources=["upload"])
            reference = gr.Image(
                label="参考画像",
                type="filepath",
                sources=["upload"],
            )
        boxes_text = gr.Textbox(
            value="50,80,180,260,0,target",
            label="Visual prompt bbox",
            lines=5,
            info="1行 = x1,y1,x2,y2,class_id[,表示ラベル]",
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
            inputs=[reference, boxes_text, visual_imgsz],
            outputs=[visual_gallery, visual_json],
        )
        visual_run.click(
            run_visual,
            inputs=[
                visual_video,
                reference,
                boxes_text,
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
1. **Reference + bbox**: ユーザーが与えた参照領域  
2. **ROI**: 観察用 crop。YOLOEが crop画像だけを直接モデル入力するわけではない  
3. **Letterbox + scaled bbox**: `rect=False` 固定で `imgsz × imgsz` に letterbox。bboxも同じ倍率・paddingで移動  
4. **Visual prompt tensor**: bboxを 1/8 解像度で二値化。同一classのbboxは OR 結合  
5. **Prompt embedding**: YOLOE/SAVPEが生成し `model.model.pe` に保持する実テンソル  
6. **Video inference**: 以後の各フレームは保持済み embedding と領域特徴を照合して検出・segment
"""
        )

if __name__ == "__main__":
    demo.queue(default_concurrency_limit=1).launch(
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True,
    )
