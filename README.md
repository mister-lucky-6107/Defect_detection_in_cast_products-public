# マルチタスク学習による鋳造製品の欠陥検出
## Multi-Task Learning for Cast Product Defect Detection

画像分類とセグメンテーションを同時に学習するマルチタスク学習による鋳造製品の欠陥検出AIです。

## 概要

過去にSIGNATEで公開されていた練習用コンペ「鋳造製品の欠陥検出」の解答コードです。
このコンペは、画像分類でデータ前処理を工夫すれば、初心者でも比較的簡単にリーダーボードでトップタイの精度に到達することが可能でした。
しかし実際には、画像データのノイズによって簡単に高い精度が出る仕組みになっており、解釈面での改善が必要でした。
また、データの質だけでなく、量も学習では画像枚数が250枚に限られており、内容としても、欠陥がその種類ごとに面積差がありすぎるなどの問題がありました。

このGitHubのコードでは、画像分類モデルにセグメンテーションも同時に学習させる、マルチタスク学習という手法を用いています。モデルはセグメンテーションタスクで欠陥位置に注目するため、分類タスクでも欠陥位置に注目しつつ分類するようになっています。

### 主な特徴
- **マルチタスク学習**: 画像分類とセグメンテーションを同時に学習
- **効率的なアーキテクチャ**: EfficientNet-B0 + FPNによる高精度な特徴抽出
- **クラス不均衡対応**: 重み付き損失関数による欠陥クラス不均衡の解決
- **交差検証**: 5-fold交差検証による堅牢なモデル評価
- **可視化機能**: SHAPによるモデル解釈性の向上


## 技術スタック

### 深層学習フレームワーク
- **PyTorch**: メインの深層学習フレームワーク
- **torchvision**: 事前学習済みモデル（EfficientNet-B0）

### データ処理・可視化
- **PILLOW**: 画像処理
- **Albumentations**: データ拡張
- **Matplotlib**: 可視化
- **SHAP**: モデル解釈

### 評価・分析
- **scikit-learn**: 機械学習ユーティリティ
- **NumPy**: 数値計算


## タスク詳細

### 1. 画像分類タスク
- **目的**: 鋳造製品に欠陥があるかどうかの二値分類
- **クラス**: 正常品（1）、欠陥品（0） ※注意　欠陥品の方が0であることに注意してください。

### 2. セグメンテーションタスク
- **目的**: 分類タスクでモデルの注目を欠陥領域に誘導すること。


## アーキテクチャ

### MultiTaskEfficientNet_FPN
```
EfficientNet-B0 (Backbone)
├── Stem Layer (32 channels)
├── Block1 (16 channels)
├── Block2 (24 channels)
├── Block3 (40 channels)
└── Block4 (1280 channels)

Feature Pyramid Network (FPN)
├── Lateral Connections (64 channels)
├── Top-down Pathway
└── Feature Fusion

Task Heads
├── Classification Head
│   ├── Global Average Pooling
│   ├── Dropout (0.3)
│   └── Linear Layer (1280 → 2)
└── Segmentation Head
    ├── Conv2d (64 → 32)
    ├── ReLU
    ├── Upsample (2x)
    └── Conv2d (32 → 8)
```

## プロジェクト構造

```
Defect_detection_in_cast_products-public/
├── src/                          # ソースコード
│   ├── MultiTaskEfficientNet_FPN.py    # メインモデル（EfficientNet-B0 + FPN）
│   ├── MultiTaskDataset_2.py           # マルチタスク用データセットクラス
│   ├── train_one_fold.py               # 交差検証学習スクリプト
│   ├── train_one_fold_early_stopping.py # 早期停止付き交差検証学習
│   ├── retrain.py                      # 全データ再学習スクリプト
│   ├── train_fnc_2.py                  # 学習関数（損失計算・最適化）
│   ├── validate_fnc_epoc.py            # エポック単位の検証関数
│   ├── validate_fnc_fold.py            # フォールド単位の検証関数
│   ├── load_and_preprocess_images_3.py # 画像読み込み・前処理
│   ├── grad_cam.py                     # Grad-CAM可視化
│   ├── visualize_grad_cam_and_pred_mask.py # Grad-CAMと予測マスクの可視化
│   ├── generate_annotation_masks_from_json.py # JSONからアノテーションマスク生成
│   ├── generate_annotation_masked_images.py # アノテーションマスク適用画像生成
│   ├── generate_multiclass_segmentation_masks_from_json.py # JSONからセグメンテーションマスク生成
│   ├── generate_multiclass_segmentation_masked_images.py # セグメンテーションマスク適用画像生成
│   ├── plot_accuracy_precision_recall.py # 精度・適合率・再現率の可視化
│   ├── plot_confusion_matrix.py        # 混同行列の可視化
│   ├── plot_loss_curves.py             # 損失曲線の可視化
│   ├── plot_roc_curve.py               # ROC曲線の可視化
│   ├── segmentation_metrics.py         # セグメンテーション評価指標計算
│   ├── make_spatial_weight_tensor.py   # 空間重みテンソル生成
│   ├── set_seed.py                     # 乱数シード設定
│   └── path_config.py                  # パス設定
├── notebooks/                     # Jupyter Notebooks
│   ├── 68_鋳造製品_EfficientNet-B0_106.ipynb  # メイン学習
│   └── 71_SHAP_28_nb68_fold_2.ipynb          # SHAP分析
├── data/                         # データディレクトリ
│   ├── raw/                      # 生データ（非公開）
│   └── processed/                # 前処理済みデータ
├── outputs/                      # 出力結果
│   ├── models/                   # 学習済みモデル
│   ├── predicted_labels/         # 予測ラベル
│   └── predicted_segmentation_masks/  # 予測マスク
└── scripts/                      # 推論用スクリプト
```

## 使用方法について

notebookをGoogleColabratoryで実行します。
68_鋳造製品_EfficientNet-B0_106.ipynbを実行すると、環境設定と学習、推論までを行います。
モデルの解釈は71_SHAP_28_nb68_fold_2.ipynbで行います。
**ただし、現在、画像データがSIGNATEによって非公開にされていることと、SIGNATEの規約により、セグメンテーションマスクを公開できないことにより、このコードをそのまま実行して頂くことはできません。**


## ソースコード詳細

### コアモデル・データセット
- **`MultiTaskEfficientNet_FPN.py`**: EfficientNet-B0をベースにFPNを組み合わせたマルチタスク学習モデル。分類とセグメンテーションを同時に実行
- **`MultiTaskDataset_2.py`**: 画像、ラベル、セグメンテーションマスクを統合したマルチタスク用データセットクラス

### 学習・検証
- **`train_one_fold.py`**: 5-fold交差検証による学習スクリプト。指定したフォールドのみを学習
- **`train_one_fold_early_stopping.py`**: 早期停止機能付きの交差検証学習。過学習を防ぐ
- **`retrain.py`**: 全データを使用した再学習スクリプト。交差検証後の最終モデル作成用
- **`train_fnc_2.py`**: 学習関数。分類損失とセグメンテーション損失の重み付き和を計算
- **`validate_fnc_epoc.py`**: エポック単位の検証関数。精度、適合率、再現率を計算
- **`validate_fnc_fold.py`**: フォールド単位の検証関数。混同行列、ROC曲線用データを収集

### データ処理
- **`load_and_preprocess_images_3.py`**: 画像の読み込み、前処理、データ拡張を実行
- **`generate_annotation_masks_from_json.py`**: LabelMeのJSONファイルから二値アノテーションマスクを生成
- **`generate_annotation_masked_images.py`**: アノテーションマスクを適用した画像を生成
- **`generate_multiclass_segmentation_masks_from_json.py`**: JSONファイルから多クラスセグメンテーションマスクを生成
- **`generate_multiclass_segmentation_masked_images.py`**: セグメンテーションマスクを適用した画像を生成

### 可視化・分析
- **`grad_cam.py`**: Grad-CAMによるモデルの注意領域可視化
- **`visualize_grad_cam_and_pred_mask.py`**: Grad-CAMと予測マスクを重ね合わせて表示
- **`plot_accuracy_precision_recall.py`**: 精度、適合率、再現率の学習曲線を可視化
- **`plot_confusion_matrix.py`**: 混同行列をヒートマップで表示
- **`plot_loss_curves.py`**: 学習・検証損失の曲線を可視化
- **`plot_roc_curve.py`**: ROC曲線とAUC値を可視化

### 評価・ユーティリティ
- **`segmentation_metrics.py`**: IoU、Dice係数などのセグメンテーション評価指標を計算
- **`make_spatial_weight_tensor.py`**: 欠陥領域を膨張させた空間重みテンソルを生成
- **`set_seed.py`**: 再現性確保のための乱数シード設定
- **`path_config.py`**: プロジェクト全体のパス設定を管理


## 評価指標

### 分類タスク
- **Accuracy**: 正解率
- **Precision**: 適応率
- **Recall**: 再現率
- **F1-Score**: F1スコア


## モデル解釈性

### Grad-CAM
欠陥検出におけるモデルの注意領域を可視化

### SHAP
各ピクセルが予測に与える影響を定量的に分析

## 結果

- **分類精度**: 交差検証で100％

作成中にSIGNATEのコンペ自体が無くなり、LBで精度を確認することはできません。

## データセットについて

**重要**: このリポジトリには学習に使用したデータセットは含まれていません。

詳細については [DATASET_NOTICE.md](DATASET_NOTICE.md) をご確認ください。

## 開発環境

- **実行環境**: Google Colab Pro
- **GPU**: Tesla T4 / V100　/ A100
- **Python**: 3.12


## ライセンス

このプロジェクトのコードはMITライセンスの下で公開されています。詳細は [LICENSE](LICENSE) をご確認ください。


---

**注意**: このプロジェクトは学習目的で作成されたものです。商用利用の際は、適切なデータセットの取得とライセンスの確認を行ってください。
