import os
import json
import base64
from PIL import Image
import io
from labelme import utils
import numpy as np
# from pathlib import Path

from set_seed import set_seed
set_seed(42)

def generate_multiclass_segmentation_masks_from_json(seg_json_dir, seg_mask_dir):
    #seg_json_dir = Path(seg_json_dir)
    #seg_mask_dir = Path(seg_mask_dir)

    # 処理対象のファイル一覧を取得
    seg_json_files = [f for f in os.listdir(seg_json_dir) if f.endswith('.json')]
    #seg_json_files = list(seg_json_dir.glob("*.json"))

    # マスク画像を生成して保存
    for seg_json_file in seg_json_files:
        seg_json_path = os.path.join(seg_json_dir, seg_json_file)

        with open(seg_json_path, 'r') as f:
            data = json.load(f)

    #for seg_json_path in seg_json_files:
        #with open(seg_json_path, 'r') as f:
            #data = json.load(f)

        # 画像データを復元
        imageData = base64.b64decode(data['imageData'])
        image = Image.open(io.BytesIO(imageData))
        image_np = np.array(image)

        # ラベル定義（背景0, 欠陥1〜7）
        label_name_to_value = {'_background_': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7}

        # shapesの中から、ラベルが定義されているものだけ抽出
        valid_shapes = [s for s in data['shapes'] if s['label'] in label_name_to_value]

        # マスク作成
        label, _ = utils.shapes_to_label(image_np.shape, valid_shapes, label_name_to_value)

        # 保存ファイル名を生成（.json → .png）
        mask_filename = os.path.splitext(seg_json_file)[0] + '_mask.png'
        mask_path = os.path.join(seg_mask_dir, mask_filename)
        #mask_filename = seg_json_path.stem + '_mask.png'
        #mask_path = seg_mask_dir / mask_filename

        # 保存（ラベル値そのまま: 0=背景, 1〜7=欠陥種別）
        Image.fromarray(label.astype(np.uint8)).save(mask_path)


    print(f"セグメンテーションマスクを {len(seg_json_files)} 枚生成し、{seg_mask_dir} に保存しました。")
