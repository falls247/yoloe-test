# YOLOE Mode Lab

YOLOE の 3 prompting mode を同じ UI で試し、特に Visual Prompt の処理を段階可視化するローカル実験環境。

- Text Prompt: クラス名から prompt embedding を生成して動画推論
- Visual Prompt: 参考画像 + ユーザー指定BBoxから Visual Prompt embedding を生成し動画へ適用
- Prompt-Free: `*-seg-pf.pt` の内蔵 vocabulary を使ってプロンプトなし推論
- 動画出力: bbox / instance mask / confidence
- 計測: 処理フレーム数、wall time、effective FPS、平均 model inference time、検出数
- Visual Prompt可視化:
  - 参考画像 + ユーザー指定BBox
  - Prompt対象領域オーバーレイ
  - letterbox と変換後 bbox
  - YOLOE が SAVPE へ渡す 1/8 解像度 visual prompt tensor
  - SAVPE 後に `model.model.pe` へ保持される prompt embedding
  - 複数 class 時の embedding cosine similarity

## YOLOEの3モード

| mode | checkpoint | 入力prompt | 用途 |
|---|---|---|---|
| Text | `yoloe-*-seg.pt` | クラス名 | 言語化できる対象 |
| Visual | `yoloe-*-seg.pt` | 参考画像 + bbox | 特定部品、ロゴ、欠陥など |
| Prompt-Free | `yoloe-*-seg-pf.pt` | なし | 何があるか探索 |

デフォルトは `yoloe-26s-seg.pt`。Prompt-Free タブでは `yoloe-26s-seg-pf.pt` のように同スケールへ自動変換する。

## セットアップ（uv）

Python 3.12 を使用。リポジトリには `.python-version` と `pyproject.toml` を含めている。

### 1. uv をインストール

すでに `uv --version` が通る場合はスキップ。

WSL / Linux:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Windows PowerShell:

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### 2. Python 3.12 と仮想環境を作成

```bash
uv python install 3.12
uv sync
```

`uv sync` でプロジェクト直下に `.venv` が自動作成され、`pyproject.toml` の依存関係がインストールされる。仮想環境の手動 activate は不要。

### 3. GPU確認

```bash
uv run python -c "import torch; print(torch.__version__); print('CUDA:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

CUDA が `False` の場合、PyTorch のインストール内容と NVIDIA driver / CUDA 環境を確認。

### 4. 起動

```bash
uv run python app.py
```

または起動スクリプト:

WSL / Linux:

```bash
bash run.sh
```

Windows PowerShell:

```powershell
.\run.ps1
```

`run.sh` / `run.ps1` は内部で `uv sync` を実行してからアプリを起動する。

ブラウザ:

```text
http://localhost:7860
```

### 最短起動

uv インストール済みなら以下だけで起動可能。

```bash
git clone https://github.com/falls247/yoloe-test.git
cd yoloe-test
uv python install 3.12
uv sync
uv run python app.py
```

YOLOE-26 は `ultralytics>=8.4.0` が必要。Text Prompt の初回処理では text encoder が追加ダウンロードされる。Visual Prompt / Prompt-Free は text encoder 不要。

`requirements.txt` は pip を使う場合の互換用として残している。通常は `uv sync` を使用する。

## Visual Prompt入力

BBox座標のテキスト手入力は廃止。参考画像をアップロード後、画像上で任意の範囲を指定する。

### BBox設定手順

1. Visual Prompt タブで参考画像をアップロード
2. `class_id` と表示用 `label` を設定
3. BBoxエディタ画像上で **対角2点を順にクリック**
   - 1回目: BBoxの1つ目の角
   - 2回目: 対角側の角
4. 白枠でBBoxが確定
5. 必要なら複数BBoxを追加
6. `Prompt前処理だけ確認` または `Visual Promptで動画実行`

座標はクリック位置から元画像pixel座標で自動取得するため、座標入力は不要。

### 複数BBox / class_id

- `class_id` は `0,1,2...` の連番
- 同じ対象の別見本は同じ `class_id` を指定
- 同じ `class_id` の複数BBoxは1つの visual prompt tensorへ OR 結合
- 異なる対象を同時に探す場合は `class_id=0`, `class_id=1` のように分ける
- 表示ラベルは UI/結果動画用。YOLOE 内部の Visual Prompt class 名は `object0`, `object1`...
- `最後のBBoxを削除` と `BBoxを全削除` で修正可能

BBox面積が画像全体の1%未満の場合、部品全体ではなく部分特徴だけをPrompt化していないか警告を表示する。

## Visual Promptの可視化が示すもの

アプリは推論時に `rect=False` を固定し、参考画像を `imgsz × imgsz` の正方形へ letterbox する。BBoxも同じ gain/padding で座標変換する。

その後、Ultralytics YOLOE の Visual Prompt predictor と同じ意味になるよう bbox を 1/8 解像度へ縮小し、二値 prompt mask へ rasterize。同一 class の bbox は OR 結合される。

```text
reference image
  └─ user selected bbox
      └─ selected prompt region
          └─ letterbox + bbox座標変換
              └─ 1/8 binary visual prompt tensor
                  └─ SAVPE
                      └─ visual prompt embedding (model.model.pe)
                          └─ target video frames
```

### 可視化項目

1. **Reference + user BBox**  
   ユーザーが画像上で指定した実BBox。

2. **Selected Prompt Region**  
   BBox外を暗くし、Promptとして有効にした領域を明示。これはcrop画像入力ではなく、元画像とPrompt Maskの関係を説明する表示。

3. **Letterbox + scaled bbox**  
   モデル入力サイズへ変換後の参考画像とBBox。

4. **Exact visual prompt tensor**  
   SAVPEへ渡る1/8解像度の二値Prompt Mask。

5. **Final prompt embedding**  
   SAVPE後に `model.model.pe` へ保持される実Visual Prompt embedding。

### 可視化していない範囲

SAVPE内部の semantic / activation branch の各中間 feature map。これらは公開 API の返り値ではないため、現状は forward hook を入れていない。Ultralytics の内部 module 名へ依存する hook はバージョン更新で壊れやすい。

必要なら次段階で `experimental/internal-hooks` として分離実装するのが安全。

## 動画の試し方

最初は `最大フレーム数=100` 程度で確認し、prompt / bbox / confidence を詰めてから `0`（全フレーム）へ変更するのが効率的。

RTX GPU を明示する場合 Device=`0`。CPU確認は Device=`cpu`。

## 既知の注意点

- 初回は model weight を自動ダウンロード
- Text Prompt は text encoder も追加ダウンロード
- Prompt-Free は別 checkpoint
- Visual Prompt result の内部 class 名は `object0...`
- 出力動画は OpenCV `mp4v`。ブラウザ環境によっては再生互換性に差が出る
- 長尺動画は Gradio request 中に同期処理するため、検証用途向け
- Ultralytics のライセンス条件は利用形態に応じて確認が必要

## 想定した「各動作モード」

このリポジトリでは「各動作モード」を YOLOE の prompting mode 3種 **Text / Visual / Prompt-Free** と解釈。

もし意図が `predict / track / export / val` などの operation mode まで含む場合、別タブとして追加する余地あり。
