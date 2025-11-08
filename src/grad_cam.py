from torchvision import transforms
from torchcam.methods import GradCAM
from torchcam.utils import overlay_mask                  # 活性化マップと元画像を重ねて表示する関数
import time
import torch
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os

from set_seed import set_seed
set_seed(42)

def grad_cam(net, t_image_dir, list_t, test_transform):

    print('	対応するラベル:正常品（ラベル1）、欠陥品（ラベル0）')

    # Grad-CAM用にNormalizeなしのtransformも用意
    gradcam_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
    ])

    # Grad-CAMエクストラクターの準備
    cam_extractor = GradCAM(net, target_layer="features.7")

    # 結果を保存するリスト
    results = []

    # 推測の開始時間の記録
    start_time = time.time()

    # モデルの読み込み
    net.load_state_dict(torch.load('net.pth'))
    net.eval()
    for param in net.parameters():
        param.requires_grad = True

    net.to('cuda')

    # 各テスト画像に対してGrad-CAMを実行
    for filename in list_t: # list_tは画像一覧のリスト
        image_path_t = os.path.join(t_image_dir, filename)
        original_image = Image.open(image_path_t).convert("RGB") # 元画像　重ね合わせに使用
        input_tensor = test_transform(original_image).unsqueeze(0).to('cuda')
        input_tensor.requires_grad_() # input_tensor から出る出力に対して Grad-CAM の hook を張る

        # 推論
    #   output = net(input_tensor)
    #  predicted_label = torch.argmax(output, dim=1).item()
    # prediction_score = torch.softmax(output, dim=1)[0, predicted_label].item()

        y_cls, y_seg = net(input_tensor)  # 出力を分解
        predicted_label = torch.argmax(y_cls, dim=1).item()
        prediction_score = torch.softmax(y_cls, dim=1)[0, predicted_label].item()


        # 勾配計算のため backward 実行
        net.zero_grad()
        y_cls[0, predicted_label].backward()  # ← y_cls の特定クラスに対して勾配を計算


        # Grad-CAMの計算
        activation_map = cam_extractor(predicted_label, input_tensor)


        # Grad-CAMマップを画像化するための前処理
            # 1枚目のCAMを取り出して、CPU＆NumPy化
        cam = activation_map[0]  # リスト -> Tensor
        if isinstance(cam, torch.Tensor):
            cam = cam.squeeze().cpu().numpy()  # shape: [H, W]
        cam = (cam - cam.min()) / (cam.max() - cam.min())  # 正規化（0〜1）
        cam = np.uint8(255 * cam)  # 0〜255にスケーリング
        cam_img = Image.fromarray(cam).resize(original_image.size)  # PIL Imageに変換＆元画像と同サイズに


        # 元の画像にGrad-CAMを重ね合わせ
        result = overlay_mask(original_image, cam_img, alpha=0.5)

        # 可視化
        plt.imshow(result)
        plt.title(f"File: {filename}, Predicted: {predicted_label}, Score: {prediction_score:.4f}")
        plt.axis('off')
        plt.show()

        results.append([filename, predicted_label])

    # 推測の時間測定
    end_time = time.time()
    execution_time = end_time - start_time
    print(f'\n推測の実行時間 : {execution_time:.4f}秒')