from torch import nn
from torchvision import models

from set_seed import set_seed
set_seed(42)

class MultiTaskEfficientNet_FPN(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        base_model = models.efficientnet_b0(pretrained=True)

        # EfficientNetの各段階特徴抽出
        self.stem = nn.Sequential(base_model.features[0])       # 112x112
        self.block1 = nn.Sequential(base_model.features[1])     # 56x56
        self.block2 = nn.Sequential(base_model.features[2])     # 28x28
        self.block3 = nn.Sequential(base_model.features[3])     # 14x14
        self.block4 = nn.Sequential(*base_model.features[4:])   # 7x7

        # 各段階の特徴を同じチャネル数に変換
        self.lateral_stem = nn.Conv2d(32, 64, 1)
        self.lateral1 = nn.Conv2d(16, 64, 1)
        self.lateral2 = nn.Conv2d(24, 64, 1)
        self.lateral3 = nn.Conv2d(40, 64, 1)
        self.lateral4 = nn.Conv2d(1280, 64, 1)

        # 分類
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(1280, num_classes)
        )

        # FPN融合後のセグメンテーションヘッド
        self.seg_head = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),  # 112→224
            #nn.Conv2d(32, 7, kernel_size=1),
            nn.Conv2d(32, 8, kernel_size=1),
        )

        # seg_head の重み初期化
        for m in self.seg_head.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        # 各段階の特徴抽出
        x_stem = self.stem(x)      # (B, 32, 112, 112)
        x1 = self.block1(x_stem)   # (B, 16, 56, 56)
        x2 = self.block2(x1)       # (B, 24, 28, 28)
        x3 = self.block3(x2)       # (B, 40, 14, 14)
        x4 = self.block4(x3)       # (B, 1280, 7, 7)

        # 分類
        out_cls = self.avgpool(x4)
        out_cls = out_cls.view(out_cls.size(0), -1)
        out_cls = self.classifier(out_cls)

        # FPN風融合
        fpn4 = self.lateral4(x4)                       # (B, 64, 7, 7)
        fpn3 = self.lateral3(x3)                       # (B, 64, 14, 14)
        fpn2 = self.lateral2(x2)                       # (B, 64, 28, 28)
        fpn1 = self.lateral1(x1)                       # (B, 64, 56, 56)
        fpn_stem = self.lateral_stem(x_stem)           # (B, 64, 112, 112)

        # 上位特徴をUpsampleして加算
        fpn3 = fpn3 + nn.functional.interpolate(fpn4, size=fpn3.shape[2:], mode='bilinear', align_corners=False)
        fpn2 = fpn2 + nn.functional.interpolate(fpn3, size=fpn2.shape[2:], mode='bilinear', align_corners=False)
        fpn1 = fpn1 + nn.functional.interpolate(fpn2, size=fpn1.shape[2:], mode='bilinear', align_corners=False)
        fpn_stem = fpn_stem + nn.functional.interpolate(fpn1, size=fpn_stem.shape[2:], mode='bilinear', align_corners=False)

        # 最終特徴マップをセグメンテーションヘッドへ
        out_seg = self.seg_head(fpn_stem)              # (B, 8, 224, 224)

        return out_cls, out_seg