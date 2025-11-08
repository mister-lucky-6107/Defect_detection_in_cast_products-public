import os
import json
import base64
from PIL import Image
import io
from labelme import utils
import numpy as np

from set_seed import set_seed
set_seed(42)

def generate_annotation_masks_from_json(json_dir, mask_dir):

    # 処理対象のファイル一覧を取得
    json_files = [f for f in os.listdir(json_dir) if f.endswith('.json')]

    # マスク画像を生成して保存
    for json_file in json_files:
        json_path = os.path.join(json_dir, json_file)

        with open(json_path, 'r') as f:
            data = json.load(f)

        # 画像データを復元
        imageData = base64.b64decode(data['imageData'])
        image = Image.open(io.BytesIO(imageData))
        image_np = np.array(image)

        # ラベル定義（背景0, 欠陥1）
        label_name_to_value = {'_background_': 0}

        # 欠陥には一律に1を割り当て
        for shape in data['shapes']:
            label_name_to_value[shape['label']] = 1

        # マスク作成
        label, _ = utils.shapes_to_label(image_np.shape, data['shapes'], label_name_to_value)

        # 保存ファイル名を生成（.json → .png）
        mask_filename = os.path.splitext(json_file)[0] + '_mask.png'
        mask_path = os.path.join(mask_dir, mask_filename)

        # 保存（0=黒, 255=白の二値画像）
        Image.fromarray((label * 255).astype(np.uint8)).save(mask_path)

    print(f"マスク画像を {len(json_files)} 枚生成し、{mask_dir} に保存しました。")
