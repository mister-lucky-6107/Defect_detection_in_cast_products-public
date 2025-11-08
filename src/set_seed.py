import random
import numpy as np
import torch

def set_seed(seed=42):
    random.seed(seed)                         # Python標準のランダム
    np.random.seed(seed)                      # NumPyのランダム
    torch.manual_seed(seed)                   # PyTorchのCPU上のランダム
    torch.cuda.manual_seed(seed)              # PyTorchのGPU上のランダム（あれば）
    torch.cuda.manual_seed_all(seed)          # GPUが複数ある場合
    torch.backends.cudnn.deterministic = True # 同じ演算を保証
    torch.backends.cudnn.benchmark = False    # 高速化のランダム性を無効化
