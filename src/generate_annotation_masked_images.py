import os
from PIL import Image
import numpy as np

from set_seed import set_seed
set_seed(42)

def generate_annotation_masked_images(train_data_dir, mask_dir, masked_dir):

    # マスク画像に合わせて処理（ファイル名ベースで一致）
    for mask_file in os.listdir(mask_dir):
        if not mask_file.endswith('_mask.png'):
            continue

        base_name = mask_file.replace('_mask.png', '')
        image_path = os.path.join(train_data_dir, base_name + '.jpeg')
        mask_path = os.path.join(mask_dir, mask_file)
        output_path = os.path.join(masked_dir, base_name + '_masked.png')

        if not os.path.exists(image_path):
            print(f"スキップ（元画像が存在しません）：{image_path}")
            continue

        # 読み込み
        image = Image.open(image_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')

        image_np = np.array(image)
        mask_np = np.array(mask) / 255  # 白：1, 黒：0

        # 欠陥部分だけ残す（背景を黒に）
        masked_np = image_np * mask_np[..., np.newaxis]
        masked_img = Image.fromarray(masked_np.astype(np.uint8))

        # 保存
        masked_img.save(output_path)

    print(f'アノテーションマスク適用画像を {len(os.listdir(masked_dir))} 枚出力しました：{masked_dir}')

