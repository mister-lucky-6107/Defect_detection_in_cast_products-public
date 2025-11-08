notebook_name=

import numpy as np
import random
import torch
from PIL import Image
from pathlib import Path
import re
import albumentations as A
from albumentations.pytorch import ToTensorV2
import csv
import time

from path_config import MODEL_DIR, RAW_TEST_DIR
from set_seed import set_seed
from MultiTaskEfficientNet_FPN import MultiTaskEfficientNet_FPN

set_seed(42)

# 各ワーカーのシード固定　DataLoaderの引数に渡す
def seed_worker(worker_id):
    worker_seed = 42 + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)

# 乱数生成器
g = torch.Generator()
g.manual_seed(42)


# 画像一覧のリスト取得
list_t = list(Path(RAW_TEST_DIR).iterdir())

# テスト画像の保存場所
t_image_dir = Path(RAW_TEST_DIR)

# テスト用の画像データのリスト
data_t = []

# 画像データ読込
for image_path_t in list_t:
    if image_path_t.is_file():  # ファイルのみ対象
        image_t = Image.open(image_path_t)  # 画像ファイルの読み込み　PILのImageオブジェクト
        data_t.append(image_t)  # リストに追加


# 推論時に必要な前処理のみ
transform = A.Compose([
    A.Resize(224, 224),
    A.Normalize(mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])


# 画像データのテンソルへの変換
t_data_tensors = []
for image_t in data_t:
    image_tensor_t = transform(image_t)
    t_data_tensors.append((image_tensor_t))


# リスト内のテンソルを結合
t_data_tensors = torch.stack(t_data_tensors).to('cuda')



main_notebook_number = re.search(r'nb(\d+)', notebook_name).group(1)
print(main_notebook_number)
NET_PATH = MODEL_DIR / f'model_fold2_nb{main_notebook_number}.pth'

# モデルの読み込み
net = MultiTaskEfficientNet_FPN()
net.load_state_dict(torch.load(NET_PATH, map_location='cuda'))
net.eval().cuda()


# 推測の開始時間の記録
start_time = time.time()

# 予測値の算出
with torch.no_grad():
    output, _ = net(t_data_tensors)
predicted_labels = torch.argmax(output, dim=1)

# 推測の時間測定
end_time = time.time()
execution_time = end_time - start_time
print(f'\n推測の実行時間 : {execution_time:.4f}秒')

# ファイル名と予測ラベルをcsvファイルに書き込み
with open('output.csv', 'w',newline='') as f:
    writer = csv.writer(f)
    for filename, label in zip(list_t, predicted_labels):
        writer.writerow([filename, label.item()])