from pathlib import Path

cwd = Path.cwd()

# このファイルから2階層上（プロジェクトルート）を取得
if 'content' in str(cwd.parts):
    # Google Colabなどの環境
    PROJECT_ROOT = Path('/content/drive/Othercomputers/マイ パソコン/Defect_detection_in_cast_products')
else:
    # ローカル
    PROJECT_ROOT = Path.cwd().parents[0]

# 各ディレクトリのパスを定義

DATA_DIR = PROJECT_ROOT / 'data'
RAW_DATA_DIR = DATA_DIR / 'raw'
PROCESSED_DATA_DIR = DATA_DIR / 'processed'

RAW_TRAIN_DIR = RAW_DATA_DIR / 'train_data'
RAW_TEST_DIR = RAW_DATA_DIR /  'test_data'
TRAIN_CSV = RAW_DATA_DIR / 'train.csv'

ANNO_MASK_JSON_DIR = PROCESSED_DATA_DIR / 'annotation_mask_json'
ANNO_MASK_DIR = PROCESSED_DATA_DIR / 'annotation_masks'
ANNO_MASKED_IMAGES_DIR = PROCESSED_DATA_DIR / 'annotation_masked_images'

SEG_MASK_JSON_DIR = PROCESSED_DATA_DIR  / 'segmentation_mask_json' / 'integrated'
SEG_MASK_DIR = PROCESSED_DATA_DIR  / 'segmentation_masks' / 'integrated'
SEG_MASKED_IMAGES_DIR = PROCESSED_DATA_DIR  / 'segmentation_masked_images'

OUTPUT_DIR = PROJECT_ROOT / 'outputs'
PREDICTED_LABELS_DIR = OUTPUT_DIR / 'predicted_labels'
PREDICTED_SEG_MASKS_DIR = OUTPUT_DIR / 'predicted_segmentation_masks'
MODEL_DIR = OUTPUT_DIR / 'models'
SHAP_DIR = OUTPUT_DIR / 'shap'

__all__  = [
    'PROJECT_ROOT',
    'DATA_DIR',
    'RAW_DATA_DIR',
    'PROCESSED_DATA_DIR',
    'RAW_TRAIN_DIR',
    'RAW_TEST_DIR',
    'TRAIN_CSV',
    'ANNO_MASK_JSON_DIR',
    'ANNO_MASK_DIR',
    'ANNO_MASKED_IMAGES_DIR',
    'SEG_MASK_JSON_DIR',
    'SEG_MASK_DIR',
    'SEG_MASKED_IMAGES_DIR',
    'OUTPUT_DIR',
    'MODEL_DIR',
    'PREDICTED_LABELS_DIR',
    'PREDICTED_SEG_MASKS_DIR',
    'SHAP_DIR'
]