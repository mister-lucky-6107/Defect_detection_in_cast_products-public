from torchvision.transforms.functional import to_pil_image # テンソル形式の画像をPILのImage形式に戻す関数
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from set_seed import set_seed
set_seed(42)

def visualize_grad_cam_and_pred_mask(original_image, cam_tensor, pred_mask_tensor, alpha=0.5):

    # サイズ取得
    W, H = original_image.size

    # 各TensorをPIL形式に変換（0〜1のfloat想定）
    cam_img = to_pil_image(cam_tensor)
    pred_img = to_pil_image(pred_mask_tensor)

    # Resize（念のため）
    cam_img = cam_img.resize((W, H))
    pred_img = pred_img.resize((W, H))

    # ヒートマップにカラーマップ（jet）を適用（OpenCV使用）
    cam_np = np.array(cam_img)
    cam_color = cv2.applyColorMap(np.uint8(255 * cam_np), cv2.COLORMAP_JET)  # BGR
    cam_color = cv2.cvtColor(cam_color, cv2.COLOR_BGR2RGB)  # RGBへ変換
    cam_color = Image.fromarray(cam_color)

    # 予測マスク → 赤色化（Rのみ強調）
    pred_np = np.array(pred_img)
    pred_red = np.zeros((H, W, 3), dtype=np.uint8)
    pred_red[..., 0] = np.uint8(255 * pred_np)  # 赤チャンネルだけ使う
    pred_color = Image.fromarray(pred_red)

    # 元画像のRGBに重ねる
    base = original_image.convert("RGB")

    blended = Image.blend(base, cam_color, alpha=alpha)  # Grad-CAMをブレンド
    blended = Image.blend(blended, pred_color, alpha=alpha)  # 予測マスクをブレンド

    # 表示
    plt.figure(figsize=(6, 6))
    plt.imshow(blended)
    plt.axis("off")
    plt.title("Grad-CAM + Pred Mask")
    plt.show()